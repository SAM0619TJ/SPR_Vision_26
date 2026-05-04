#include <atomic>
#include <chrono>
#include <optional>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

using namespace std::chrono_literals;

const std::string keys =
    "{help h usage ? |                        | 输出命令行参数说明}"
    "{@config-path   | configs/standard3_tensorrt.yaml| "
    "位置参数，yaml配置文件路径 }";

int main(int argc, char *argv[]) {
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};

  auto plan_thread = std::thread([&]() {
    while (!quit) {
      const auto current_mode = mode.load();
      if (current_mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);
        gimbal.send(plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc,
                    plan.pitch, plan.pitch_vel, plan.pitch_acc);
        std::this_thread::sleep_for(8ms);
      } else {
        // 比赛版暂时关闭打符和其它模式，确保非自瞄状态不发控制/开火。
        gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
        std::this_thread::sleep_for(20ms);
      }
    }
  });

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  int frame_count = 0;
  auto last_mode = io::GimbalMode::IDLE;

  while (!exiter.exit()) {
    const auto current_mode = gimbal.mode();
    mode = current_mode;
    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", gimbal.str(current_mode));
      if (current_mode == io::GimbalMode::SMALL_BUFF ||
          current_mode == io::GimbalMode::BIG_BUFF) {
        tools::logger()->warn("Buff mode is disabled in standard_mpc.");
      }
      last_mode = current_mode;
    }

    camera.read(img, t);
    auto q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    if (current_mode == io::GimbalMode::AUTO_AIM) {
      auto armors = yolo.detect(img, frame_count);
      auto targets = tracker.track(armors, t);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);
    } else {
      target_queue.push(std::nullopt);
    }

    ++frame_count;
  }

  quit = true;
  target_queue.push(std::nullopt);
  if (plan_thread.joinable())
    plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
