#pragma once
// web_debugger.hpp — 轻量 HTTP + WebSocket 调试服务器
// 依赖：POSIX socket（Linux/Jetson 原生支持），OpenCV，nlohmann_json
// 用法：
//   WebDebugger dbg(8080);
//   dbg.start();
//   // 每帧推送
//   dbg.push(frame, detections, reprojections, latency_ms);

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace debug
{

struct DetectionData {
  std::vector<cv::Point2f> pts;   // 4 个关键点（图像坐标）
  int color;                       // 0=blue 1=red
  int number;                      // 0=guard 1-5 6=outpost 7-8=base
  float conf;
};

struct ReprojectionData {
  std::vector<cv::Point2f> pts;   // 3D 点投影到图像的坐标
};

class WebDebugger
{
public:
  explicit WebDebugger(int port = 8080);
  ~WebDebugger();

  // 启动后台服务线程（非阻塞）
  void start();
  void stop();

  // 每帧调用，线程安全
  void push(
    const cv::Mat & frame,
    const std::vector<DetectionData> & detections,
    const std::vector<ReprojectionData> & reprojections,
    double latency_ms);

  bool is_running() const { return running_; }
  int port() const { return port_; }

private:
  int port_;
  std::atomic<bool> running_{false};
  std::thread server_thread_;

  // 最新帧数据（加锁保护）
  std::mutex data_mutex_;
  std::string latest_json_;   // 序列化后的 JSON 字符串

  // WebSocket 客户端 fd 列表
  std::mutex clients_mutex_;
  std::vector<int> ws_clients_;

  void serve();
  void handle_http(int client_fd);
  void handle_websocket(int client_fd);
  void broadcast(const std::string & msg);

  // 工具函数
  static std::string mat_to_base64_jpg(const cv::Mat & img, int quality = 60);
  static std::string build_json(
    const cv::Mat & frame,
    const std::vector<DetectionData> & detections,
    const std::vector<ReprojectionData> & reprojections,
    double latency_ms);
  static std::string ws_handshake_response(const std::string & key);
  static std::string ws_frame(const std::string & payload);
  static bool parse_ws_frame(const std::vector<uint8_t> & buf, std::string & out);
};

}  // namespace debug
