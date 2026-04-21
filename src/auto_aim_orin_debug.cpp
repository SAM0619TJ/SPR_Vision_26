#include <fmt/core.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>

#include "debug/web_debugger.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                              | 输出命令行参数说明 }"
  "{config-path c  | ../configs/vtune_test.yaml   | yaml配置文件的路径（相对于build目录）}"
  "{use-camera     | true                         | 使用真实相机而非视频文件（默认启用）}"
  "{start-index s  | 0                            | 视频起始帧下标    }"
  "{end-index e    | 0                            | 视频结束帧下标    }"
  "{@input-path    | ../assets/demo/demo          | avi和txt文件的路径（相对于build目录）}";

static std::string resolve_config_path(const std::string & raw_path)
{
  namespace fs = std::filesystem;
  fs::path input(raw_path);
  if (input.is_absolute() && fs::exists(input)) {
    return input.string();
  }

  std::vector<fs::path> candidates = {
    fs::current_path() / input,
    fs::current_path() / "build" / input,
    fs::current_path() / "configs" / input.filename(),
    fs::current_path() / ".." / "configs" / input.filename(),
  };
  for (const auto & candidate : candidates) {
    if (fs::exists(candidate)) {
      return fs::weakly_canonical(candidate).string();
    }
  }
  return raw_path;
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto use_camera = cli.get<bool>("use-camera");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");

  const auto has_explicit_config_arg = [argc, argv]() {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-c" || arg == "--config-path") {
        return true;
      }
      if (arg.rfind("-c=", 0) == 0 || arg.rfind("--config-path=", 0) == 0) {
        return true;
      }
    }
    return false;
  };

  if (!has_explicit_config_arg()) {
    const auto has_suffix = [](const std::string & s, const std::string & suffix) {
      return s.size() >= suffix.size() &&
        s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    const bool positional_is_yaml = has_suffix(input_path, ".yaml") || has_suffix(input_path, ".yml");
    if (positional_is_yaml) {
      config_path = input_path;
      input_path = "../assets/demo/demo";
    }
  }

  config_path = resolve_config_path(config_path);
  if (!std::filesystem::exists(config_path)) {
    tools::logger()->error("Config file not found: {}", config_path);
    tools::logger()->error("Use --config-path configs/vtune_test.yaml or run from build directory.");
    return -1;
  }

  auto yaml = YAML::LoadFile(config_path);
  const bool yolo_debug = yaml["yolo_debug"].as<bool>(false);
  const bool enable_web_debug = yaml["enable_web_debug"].as<bool>(false);
  const int web_debug_port = yaml["web_debug_port"].as<int>(8080);

  tools::Plotter plotter;
  tools::Exiter exiter;
  std::unique_ptr<debug::WebDebugger> debugger;
  if (enable_web_debug) {
    debugger = std::make_unique<debug::WebDebugger>(web_debug_port);
    debugger->start();
    tools::logger()->info("[OrinDebug] Web debugger enabled on port {}", web_debug_port);
  } else {
    tools::logger()->info("[OrinDebug] Web debugger disabled by config");
  }

  std::unique_ptr<io::Camera> camera;
  std::unique_ptr<io::Gimbal> gimbal;
  cv::VideoCapture video;
  std::ifstream text;

  if (use_camera) {
    tools::logger()->info("[OrinDebug] Using real camera input");
    camera = std::make_unique<io::Camera>(config_path);
    gimbal = std::make_unique<io::Gimbal>(config_path);
  } else {
    tools::logger()->info("[OrinDebug] Using video file: {}", input_path);
    auto video_path = fmt::format("{}.avi", input_path);
    auto text_path = fmt::format("{}.txt", input_path);
    video.open(video_path);
    text.open(text_path);

    if (!video.isOpened()) {
      tools::logger()->error("Failed to open video: {}", video_path);
      return -1;
    }

    video.set(cv::CAP_PROP_POS_FRAMES, start_index);
    for (int i = 0; i < start_index; i++) {
      double t, w, x, y, z;
      text >> t >> w >> x >> y >> z;
    }
  }

  auto_aim::YOLO yolo(config_path, yolo_debug);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  cv::Mat img;
  auto t0 = std::chrono::steady_clock::now();
  auto_aim::Target last_target;
  io::Command last_command{false, false, 0, 0, 0};
  double last_t = -1;
  int web_debug_div = 0;

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    std::chrono::steady_clock::time_point timestamp;
    Eigen::Quaterniond q;

    if (use_camera) {
      camera->read(img, timestamp);
      if (img.empty()) {
        tools::logger()->warn("Failed to read from camera");
        continue;
      }

      q = gimbal->q(timestamp - std::chrono::milliseconds(1));
      solver.set_R_gimbal2world(q);
    } else {
      video.read(img);
      if (img.empty()) break;

      double t, w, x, y, z;
      text >> t >> w >> x >> y >> z;
      timestamp = t0 + std::chrono::microseconds(int(t * 1e6));
      q = {w, x, y, z};
      solver.set_R_gimbal2world(q);
    }

    auto gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    auto gs = use_camera ? gimbal->state() : io::GimbalState{0, 0, 0, 0, 27, 0};

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto command = aimer.aim(targets, timestamp, use_camera ? gs.bullet_speed : 27.0, use_camera);
    if (use_camera) {
      command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);
      gimbal->send(
        command.control, command.shoot, static_cast<float>(command.yaw), 0.f, 0.f,
        static_cast<float>(command.pitch), 0.f, 0.f);
    }

    if (command.control) last_command = command;

    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

    nlohmann::json data;
    data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
    data["armor_num"] = armors.size();
    data["frame"] = frame_count;

    data["gimbal_yaw"] = gimbal_pos[0] * 57.3;
    data["gimbal_pitch"] = gimbal_pos[1] * 57.3;
    data["bullet_speed"] = use_camera ? gs.bullet_speed : 27.0;

    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    data["shoot"] = command.shoot;

    if (!armors.empty()) {
      const auto & armor = armors.front();
      tools::draw_points(img, armor.points, {255, 255, 255}, 2);
      tools::draw_text(img, "Detection", armor.points[0] - cv::Point2f(0, 10), {255, 255, 255});

      auto reproject_raw = solver.reproject_armor(
        armor.xyz_in_world, armor.yaw_raw, armor.type, armor.name);
      auto reproject_opt = solver.reproject_armor(
        armor.xyz_in_world, armor.ypr_in_world[0], armor.type, armor.name);

      tools::draw_points(img, reproject_opt, {0, 255, 0});
      tools::draw_text(img, "PnP Optimized", reproject_opt[0] - cv::Point2f(0, 40), {0, 255, 0});

      double error_raw = 0.0, error_opt = 0.0;
      for (int i = 0; i < 4; i++) {
        error_raw += cv::norm(armor.points[i] - reproject_raw[i]);
        error_opt += cv::norm(armor.points[i] - reproject_opt[i]);
      }

      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["reproj_error_raw"] = error_raw;
      data["reproj_error_optimized"] = error_opt;

      if (!targets.empty()) {
        auto target = targets.front();

        if (last_t == -1) {
          last_target = target;
          last_t = use_camera ? tools::delta_time(timestamp, t0) : frame_count;
        }

        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d & xyza : armor_xyza_list) {
          auto image_points = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, image_points, {0, 255, 0});
        }

        auto aim_point = aimer.debug_aim_point;
        Eigen::Vector4d aim_xyza = aim_point.xyza;
        auto image_points = solver.reproject_armor(
          aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});
        else tools::draw_points(img, image_points, {255, 0, 0});

        Eigen::VectorXd x = target.ekf_x();
        data["x"] = x[0];
        data["vx"] = x[1];
        data["y"] = x[2];
        data["vy"] = x[3];
        data["z"] = x[4];
        data["vz"] = x[5];
        data["a"] = x[6] * 57.3;
        data["w"] = x[7];
        data["r"] = x[8];
        data["l"] = x[9];
        data["h"] = x[10];
        data["last_id"] = target.last_id;

        data["residual_yaw"] = target.ekf().data.at("residual_yaw");
        data["residual_pitch"] = target.ekf().data.at("residual_pitch");
        data["residual_distance"] = target.ekf().data.at("residual_distance");
        data["residual_angle"] = target.ekf().data.at("residual_angle");
        data["nis"] = target.ekf().data.at("nis");
        data["nees"] = target.ekf().data.at("nees");
        data["nis_fail"] = target.ekf().data.at("nis_fail");
        data["nees_fail"] = target.ekf().data.at("nees_fail");
        data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
      }
    }

    if (++web_debug_div >= 3) {
      web_debug_div = 0;
      std::vector<debug::DetectionData> web_dets;
      std::vector<debug::ReprojectionData> web_reprojs;
      web_dets.reserve(armors.size());
      web_reprojs.reserve(armors.size());

      for (const auto & armor : armors) {
        debug::DetectionData d;
        d.pts = armor.points;
        d.color = static_cast<int>(armor.color);
        d.number = static_cast<int>(armor.name);
        d.conf = armor.confidence;
        web_dets.push_back(std::move(d));

        debug::ReprojectionData r;
        r.pts = solver.reproject_armor(armor.xyz_in_world, armor.ypr_in_world[0], armor.type, armor.name);
        web_reprojs.push_back(std::move(r));
      }

      double latency = tools::delta_time(std::chrono::steady_clock::now(), t0) * 1000.0;
      if (debugger) {
        debugger->push(img, web_dets, web_reprojs, latency);
      }
    }

    data["yolo_ms"] = tools::delta_time(tracker_start, yolo_start) * 1e3;
    data["tracker_ms"] = tools::delta_time(std::chrono::steady_clock::now(), tracker_start) * 1e3;
    data["last_command_yaw"] = last_command.yaw * 57.3;
    data["last_command_pitch"] = last_command.pitch * 57.3;

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}