// debug_usage_example.cpp — 展示如何在主循环中接入 WebDebugger
// 这不是独立可执行文件，是集成示例，参考后把相关代码加入 standard.cpp

#include "debug/web_debugger.hpp"
#include "tasks/auto_aim/armor.hpp"

// ── 在 main() 或 AutoAim 初始化时启动调试器 ──────────────────────────────────
//
//   debug::WebDebugger debugger(8080);
//   debugger.start();
//   // 浏览器访问 http://<jetson-ip>:8080

// ── 每帧检测完成后推送 ────────────────────────────────────────────────────────
//
// void push_debug(
//   debug::WebDebugger & dbg,
//   const cv::Mat & frame,
//   const std::list<auto_aim::Armor> & armors,
//   const std::vector<std::vector<cv::Point2f>> & reprojected_pts,
//   double latency_ms)
// {
//   std::vector<debug::DetectionData> dets;
//   for (const auto & armor : armors) {
//     debug::DetectionData d;
//     d.pts    = armor.points;                          // std::vector<cv::Point2f>
//     d.color  = static_cast<int>(armor.color);         // 0=blue 1=red
//     d.number = static_cast<int>(armor.name);          // 0=guard 1-5 6=outpost...
//     d.conf   = armor.confidence;
//     dets.push_back(d);
//   }
//
//   std::vector<debug::ReprojectionData> reproj;
//   for (const auto & pts : reprojected_pts) {
//     debug::ReprojectionData r;
//     r.pts = pts;
//     reproj.push_back(r);
//   }
//
//   dbg.push(frame, dets, reproj, latency_ms);
// }
