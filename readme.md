# SPR_Vision_26

RoboMaster 视觉系统，针对 **Nvidia Jetson Xavier NX 8G**（JetPack 5.x，TRT 8.5）优化。
从 sp_vision_25-TRT（Orin NX，TRT 10.x）移植，集成 RobotDetectionModel（0708.onnx）权重。

---

## 硬件目标

| 项目 | 规格 |
| --- | --- |
| 平台 | Jetson Xavier NX 8G |
| JetPack | 5.x |
| CUDA | 11.4 |
| TensorRT | 8.5 |
| cuDNN | 8.6 |
| CUDA 架构 | SM 7.2（Volta） |

---

## 依赖

JetPack 5.x 已包含：CUDA 11.4、TensorRT 8.5、cuDNN 8.6、OpenCV 4.5（with CUDA）

额外安装：

```bash
sudo apt install libeigen3-dev      # EKF
sudo apt install libyaml-cpp-dev    # 配置文件
sudo apt install libfmt-dev         # 格式化输出
sudo apt install libspdlog-dev      # 日志
sudo apt install nlohmann-json3-dev # JSON（CMake 查找用）
```

CMake 最低版本：**3.16.3**

---

## 构建

```bash
cd SPR_Vision_26
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j6
```

Xavier NX 编译标志（CMakeLists.txt 自动设置）：

```cmake
set(CMAKE_CUDA_ARCHITECTURES "72")          # Xavier NX = SM 7.2
set(CMAKE_CXX_FLAGS "-O3 -march=armv8.2-a+fp16 -mtune=cortex-a57")
```

---

## 架构

```text
SPR_Vision_26/
├── src/
│   └── standard.cpp              # 主循环入口
├── tasks/
│   └── auto_aim/
│       ├── armor.hpp/cpp         # 装甲板数据结构
│       ├── detector.hpp/cpp      # 检测器接口
│       ├── tracker.hpp/cpp       # EKF 跟踪器
│       ├── solver.hpp/cpp        # PnP 位姿解算
│       ├── aimer.hpp/cpp         # 瞄准逻辑
│       ├── shooter.hpp/cpp       # 射击决策
│       ├── classifier.hpp/cpp    # 装甲板分类
│       ├── voter.hpp/cpp         # 目标投票
│       ├── yolo.hpp/cpp          # YOLO 统一接口
│       ├── yolos/
│       │   ├── trt_compat.hpp    # TRT 8.5 / 10.x 兼容层（关键）
│       │   ├── yolov5.hpp/cpp    # YOLOv5 TRT 推理
│       │   ├── yolov8.hpp/cpp    # YOLOv8 TRT 推理
│       │   ├── yolo11.hpp/cpp    # YOLO11 TRT 推理
│       │   ├── yolo26.hpp/cpp    # YOLO26 TRT 推理（动态 shape）
│       │   ├── cuda_preprocess.hpp/cu  # CUDA letterbox 预处理
│       │   └── tensorrt_logger.cpp     # TRT 日志
│       ├── multithread/
│       │   ├── mt_detector.hpp/cpp     # 多线程检测器
│       │   └── commandgener.hpp/cpp    # 指令生成
│       └── planner/
│           ├── planner.hpp/cpp         # MPC 规划器
│           └── tinympc/                # TinyMPC 求解器
├── tasks/auto_buff/              # 能量机关
├── tasks/omniperception/         # 全局感知
├── debug/
│   ├── web_debugger.hpp/cpp      # POSIX socket HTTP + WebSocket 服务器
│   ├── static/index.html         # 前端可视化页面
│   └── usage_example.cpp         # 集成示例
├── io/                           # 相机、串口、云台驱动
├── tools/                        # EKF、轨迹、日志等工具
├── librm/                        # RM 协议库
├── calibration/                  # 标定工具
├── tests/                        # 测试程序
└── configs/                      # YAML 配置文件
```

---

## TRT 兼容层（trt_compat.hpp）

`tasks/auto_aim/yolos/trt_compat.hpp` 通过编译期宏桥接 TRT 8.5 和 TRT 10.x 的 API 差异，
同一份源码可在两个平台编译：

| TRT 10.x API | TRT 8.5 等价 |
| --- | --- |
| `setMemoryPoolLimit` | `setMaxWorkspaceSize` |
| `getIOTensorName` / `setTensorAddress` | `getBindingName` / `executeV2` |
| `enqueueV3` | `enqueueV2` |
| `getTensorShape` | `getBindingDimensions` |

---

## Web 调试器

`debug/web_debugger` 是纯 POSIX socket 实现的 HTTP + WebSocket 服务器（无外部依赖）：

- WebSocket 握手使用内联 SHA-1（RFC 3174）+ Base64，不依赖 OpenSSL
- 每帧推送：base64 JPEG 帧 + 检测框 + 重投影点 + 延迟
- 前端自动重连，支持切换显示关键点/重投影/标签

**集成方式**（参考 `debug/usage_example.cpp`）：

```cpp
#include "debug/web_debugger.hpp"

debug::WebDebugger debugger(8080);
debugger.start();
// 浏览器访问 http://<jetson-ip>:8080

// 每帧检测后：
std::vector<debug::DetectionData> dets;
for (const auto & armor : armors) {
    debug::DetectionData d;
    d.pts    = armor.points;
    d.color  = static_cast<int>(armor.color);
    d.number = static_cast<int>(armor.name);
    d.conf   = armor.confidence;
    dets.push_back(d);
}
debugger.push(frame, dets, {}, latency_ms);
```

---

## 模型

- 权重文件：`0708.onnx`（RobotDetectionModel，YOLOv5+MobileNetV3）
- 输入：640×640，输出：25200×22（YOLO grid）
- 首次运行自动通过 `nvonnxparser` 转换为 TRT engine（`.trt` 缓存）
- FP16 自动检测：`builder->platformHasFastFp16()` 为 true 时启用（Xavier NX Tensor Core）
