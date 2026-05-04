#include <fmt/core.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include "debug/web_debugger.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

namespace {

constexpr unsigned char kCtrlQ = 17;

class TerminalRawMode {
public:
  bool enable() {
    if (!isatty(STDIN_FILENO))
      return false;

    if (tcgetattr(STDIN_FILENO, &original_) != 0)
      return false;

    termios raw = original_;
    raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    raw.c_iflag &= static_cast<tcflag_t>(~(IXON | IXOFF));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;

    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0)
      return false;

    enabled_ = true;
    return true;
  }

  void restore() {
    if (!enabled_)
      return;
    tcsetattr(STDIN_FILENO, TCSANOW, &original_);
    enabled_ = false;
  }

  ~TerminalRawMode() { restore(); }

private:
  termios original_{};
  bool enabled_ = false;
};

void stop_gimbal(io::Gimbal &gimbal) {
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
}

void cleanup_auto_aim_processes() {
  const auto self_pid = static_cast<long>(getpid());
  const auto cmd = fmt::format(
      "sh -c '"
      "for signal in TERM KILL; do "
      "for exe in /proc/[0-9]*/exe; do "
      "path=$(readlink \"$exe\" 2>/dev/null) || continue; "
      "name=${{path##*/}}; pid=${{exe#/proc/}}; pid=${{pid%/exe}}; "
      "case \"$name\" in "
      "auto_aim_debug_mpc|auto_aim_orin_debug|mt_auto_aim_debug|auto_aim_test|standard_mpc) "
      "[ \"$pid\" != \"$1\" ] && kill -$signal \"$pid\" 2>/dev/null ;; "
      "esac; "
      "done; "
      "if [ \"$signal\" = TERM ]; then sleep 0.2; fi; "
      "done' sh {}",
      self_pid);

  const auto ret = std::system(cmd.c_str());
  if (ret != 0) {
    tools::logger()->warn("auto aim process cleanup command returned {}", ret);
  }
}

std::thread start_ctrl_q_thread(std::atomic<bool> &quit,
                                std::atomic<bool> &app_running,
                                io::Gimbal &gimbal,
                                TerminalRawMode &terminal) {
  return std::thread([&]() {
    if (!terminal.enable()) {
      tools::logger()->warn("stdin is not a TTY, Ctrl+Q emergency exit disabled.");
      return;
    }

    tools::logger()->info("Press Ctrl+Q to force quit auto aim processes.");

    while (app_running && !quit) {
      unsigned char key = 0;
      const auto n = read(STDIN_FILENO, &key, 1);
      if (n == 1 && key == kCtrlQ) {
        tools::logger()->warn("Ctrl+Q received, force quitting auto aim.");
        quit = true;
        stop_gimbal(gimbal);
        cleanup_auto_aim_processes();

        std::this_thread::sleep_for(500ms);
        if (app_running) {
          terminal.restore();
          std::_Exit(0);
        }
        return;
      }

      if (n < 0 && errno != EINTR && errno != EAGAIN) {
        tools::logger()->warn("stdin read failed, Ctrl+Q emergency exit disabled.");
        return;
      }
    }
  });
}

} // namespace

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/standard3_tensorrt.yaml| "
    "位置参数，yaml配置文件路径 }";

int main(int argc, char *argv[]) {
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = YAML::LoadFile(config_path);
  plotter.configure(config_path);
  const bool display_requested = yaml["enable_imshow"].as<bool>(false);
  const bool enable_web_debug = yaml["enable_web_debug"].as<bool>(false);
  const int web_debug_port = yaml["web_debug_port"].as<int>(8080);
  const bool has_display = std::getenv("DISPLAY") != nullptr ||
                           std::getenv("WAYLAND_DISPLAY") != nullptr;
  const bool enable_imshow = display_requested && has_display;
  const bool yolo_debug = yaml["yolo_debug"].as<bool>(false) && enable_imshow;

  if (display_requested && !has_display) {
    tools::logger()->warn("enable_imshow=true but no display detected, "
                          "disabling imshow for headless run.");
  }
  if (yaml["yolo_debug"].as<bool>(false) && !enable_imshow) {
    tools::logger()->warn("yolo_debug requested but imshow is disabled, "
                          "forcing yolo_debug=false.");
  }

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, yolo_debug);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  std::unique_ptr<debug::WebDebugger> debugger;
  if (enable_web_debug) {
    debugger = std::make_unique<debug::WebDebugger>(web_debug_port);
    debugger->start();
  }

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<bool> app_running = true;
  TerminalRawMode terminal;
  auto ctrl_q_thread = start_ctrl_q_thread(quit, app_running, gimbal, terminal);

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      auto t_front_start = std::chrono::steady_clock::now();
      auto target = target_queue.front();
      auto t_front_end = std::chrono::steady_clock::now();
      double front_ms =
          std::chrono::duration<double, std::milli>(t_front_end - t_front_start)
              .count();
      if (front_ms > 5.0)
        tools::logger()->warn("[plan] front() blocked {:.1f}ms", front_ms);
      auto gs = gimbal.state();
      auto plan = planner.plan(target, gs.bullet_speed);

      gimbal.send(plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc,
                  plan.pitch, plan.pitch_vel, plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["gimbal_yaw"] = gs.yaw;
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;

      data["plan_yaw"] = plan.yaw;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;

      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];  // z
        data["target_vz"] = target->ekf_x()[5]; // vz
      }

      if (target.has_value()) {
        data["w"] = target->ekf_x()[7];
      } else {
        data["w"] = 0.0;
      }

      plotter.plot(data);

      std::this_thread::sleep_for(10ms);
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  int frame_count = 0;

  while (!exiter.exit() && !quit) {
    auto t_frame_start = std::chrono::steady_clock::now();

    camera.read(img, t);
    auto q = gimbal.q(t);

    solver.set_R_gimbal2world(q);

    auto t_detect_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto t_detect_end = std::chrono::steady_clock::now();

    auto targets = tracker.track(armors, t);

    auto t_push_start = std::chrono::steady_clock::now();
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);
    auto t_push_end = std::chrono::steady_clock::now();

    double detect_ms =
        std::chrono::duration<double, std::milli>(t_detect_end - t_detect_start)
            .count();
    double push_ms =
        std::chrono::duration<double, std::milli>(t_push_end - t_push_start)
            .count();
    double frame_ms =
        std::chrono::duration<double, std::milli>(t_push_end - t_frame_start)
            .count();
    // tools::logger()->info("[main] detect={:.1f}ms push_wait={:.1f}ms
    // frame={:.1f}ms",detect_ms, push_ms, frame_ms);

    cv::Mat display;
    if (enable_imshow || debugger) {
      cv::cvtColor(img, display, cv::COLOR_RGB2BGR);
    }

    if (enable_imshow && !targets.empty()) {
      auto target = targets.front();

      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d &xyza : armor_xyza_list) {
        auto image_points = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(display, image_points, {0, 255, 0});
      }

      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points = solver.reproject_armor(
          aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(display, image_points, {0, 0, 255});
    }

    if (enable_imshow) {
      cv::resize(display, display, {}, 0.5, 0.5);
      cv::imshow("reprojection", display);
      auto key = cv::waitKey(1);
      if (key == 'q') {
        quit = true;
        break;
      }
    }

    if (debugger) {
      std::vector<debug::DetectionData> web_dets;
      std::vector<debug::ReprojectionData> web_reprojs;
      for (const auto &armor : armors) {
        debug::DetectionData d;
        d.pts = armor.points;
        d.color = static_cast<int>(armor.color);
        d.number = static_cast<int>(armor.name);
        d.conf = armor.confidence;
        web_dets.push_back(d);
      }
      if (!targets.empty()) {
        auto target = targets.front();
        for (const Eigen::Vector4d &xyza : target.armor_xyza_list()) {
          debug::ReprojectionData r;
          r.pts = solver.reproject_armor(xyza.head(3), xyza[3],
                                         target.armor_type, target.name);
          web_reprojs.push_back(r);
        }
      }
      debugger->push(display, web_dets, web_reprojs, frame_ms);
    }

    ++frame_count;
  }

  quit = true;
  if (plan_thread.joinable())
    plan_thread.join();
  app_running = false;
  if (ctrl_q_thread.joinable())
    ctrl_q_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}