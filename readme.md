# SPR_Vision_26

面向 RoboMaster 场景的视觉系统，当前仓库以 C++17 + CMake 为核心，覆盖自瞄主流程、调试可视化、标定工具和若干测试程序。  
项目默认针对 Jetson ARM64 环境构建，当前 `CMakeLists.txt` 中的默认优化目标为 Orin 系列（SM 8.7），同时保留了 Xavier NX 的 CUDA/TensorRT 兼容查找路径。

## 项目能力

- 自瞄主流程：相机取流、装甲板检测、PnP 解算、目标跟踪、瞄准与射击决策
- 多后端推理：TensorRT 为主，OpenVINO 为可选后端
- 调试能力：Web 可视化调试、OpenCV 窗口调试、日志记录
- 工具链：相机标定、手眼标定、离线/在线测试程序
- 可选扩展：ROS2 可视化、能量机关模块、全向感知模块

## 平台与依赖

### 基础依赖

- CMake >= 3.16.3
- OpenCV >= 4.5
- Eigen3
- yaml-cpp
- fmt
- spdlog
- nlohmann_json

Ubuntu / Jetson 上可直接安装的包：

```bash
sudo apt install \
  libeigen3-dev \
  libyaml-cpp-dev \
  libfmt-dev \
  libspdlog-dev \
  nlohmann-json3-dev
```

### 可选依赖

- CUDA + TensorRT：启用 TensorRT 推理分支
- OpenVINO Runtime：启用 OpenVINO 推理与 `tasks/omniperception`
- Ceres：与 OpenVINO 一起启用 `tasks/auto_buff`
- ROS2：构建 `standard_ros2` 及部分可视化能力

### 平台说明

根 `CMakeLists.txt` 当前默认策略：

- `aarch64/arm64` 下默认 `CMAKE_CUDA_ARCHITECTURES=87`
- ARM 优化参数为 `-O3 -march=armv8.2-a+fp16 -mtune=cortex-a78ae`
- CUDA 查找路径同时兼容 JetPack 6.x 常见 CUDA 12.6 路径和 Xavier NX 常见 CUDA 11.4 路径

如果你在 Xavier NX 上构建，建议显式传入合适的 CUDA 架构，例如：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=72
```

## 构建

```bash
git clone <your-repo-url>
cd SPR_Vision_26
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

启用 ROS2 时可追加：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS2=ON
cmake --build build -j$(nproc)
```

## 主要可执行文件

默认会构建以下目标：

- `standard`：主程序入口
- `auto_aim_debug_mpc`
- `auto_aim_orin_debug`
- `mt_auto_aim_debug`
- `capture`
- `calibrate_camera`
- `calibrate_handeye`
- `calibrate_robotworld_handeye`
- `auto_aim_test`
- `camera_detect_test`
- `camera_test`
- `camera_pnp_test`
- `detector_video_test`
- `planner_test`
- `planner_test_offline`
- `camera_planner_realtime`

当 ROS2 环境可用并启用 `-DENABLE_ROS2=ON` 时，还会构建：

- `standard_ros2`

## 快速运行

### 主程序

`src/standard.cpp` 的默认配置文件是 `configs/standard3.yaml`：

```bash
./build/standard
```

也可以显式指定配置文件：

```bash
./build/standard configs/standard3.yaml
./build/standard configs/standard3_tensorrt.yaml
```

### 自瞄测试程序

```bash
./build/auto_aim_test configs/standard3_tensorrt.yaml
```

`auto_aim_test` 支持摄像头输入和离线素材输入，并会读取配置中的 `enable_web_debug`、`web_debug_port`、`yolo_debug` 等选项。

## 配置文件

主要配置位于 `configs/`：

- `standard3.yaml`：基础配置，偏传统/常规运行
- `standard3_tensorrt.yaml`：TensorRT 运行示例
- `camera.yaml`、`calibration.yaml`、`vtune_test.yaml`：专项用途配置

`standard3_tensorrt.yaml` 中包含的典型配置项有：

- `inference_backend`：`tensorrt` 或 `openvino`
- `yolo_name`：当前启用的模型名
- `*_engine_path` / `*_onnx_path`：TensorRT engine 与 ONNX 路径
- `enable_web_debug`：是否启用网页调试
- `enable_imshow`：是否开启 OpenCV 显示窗口
- `camera_name`、`exposure_ms`、`gain`：相机参数
- `com_port`、`can_interface`：云台/板卡通信参数
- `camera_matrix`、`distort_coeffs`、`R_camera2gimbal`、`t_camera2gimbal`：标定结果

注意：

- 当前部分 TensorRT 路径使用了绝对路径，例如 `/home/spr/SPR_Vision_26/...`
- 更换机器或部署到新环境时，请先检查这些路径是否有效
- 无桌面或 SSH 环境下，建议将 `enable_imshow` 设为 `false`

## 推理后端说明

### TensorRT

- 当检测到 CUDA、TensorRT、`nvonnxparser` 后，`tasks/auto_aim` 会编译 TensorRT 版本 YOLO
- 支持的实现位于 `tasks/auto_aim/yolos/`，包含 `yolov5_trt.cpp`、`yolov8_trt.cpp`、`yolo11_trt.cpp`、`yolo26_trt.cpp`
- CUDA 预处理由 `cuda_preprocess.cu` 提供

### OpenVINO

- 检测到 OpenVINO Runtime 后，会启用 `tasks/auto_aim` 中的 OpenVINO 分支
- `tasks/omniperception` 仅在 OpenVINO 可用时构建
- `tasks/auto_buff` 需要 OpenVINO 和 Ceres 同时可用

## Web 调试

`debug/web_debugger.cpp` 提供一个轻量级 HTTP + WebSocket 调试服务，构建时会将 `debug/static/` 复制到 `build/debug/`。

当配置中启用：

```yaml
enable_web_debug: true
web_debug_port: 8080
```

可在浏览器访问：

```text
http://<device-ip>:8080
```

适合用于远程查看图像帧、检测结果和调试信息。

## 目录结构

```text
SPR_Vision_26/
├── src/                  # 主程序与调试程序入口
├── tasks/auto_aim/       # 自瞄主模块
├── tasks/auto_buff/      # 能量机关模块（依赖 OpenVINO + Ceres）
├── tasks/omniperception/ # 全向感知模块（依赖 OpenVINO）
├── io/                   # 相机、板卡、云台、串口等 IO
├── tools/                # 日志、数学、绘图、录像等工具
├── debug/                # Web 调试服务与前端静态资源
├── calibration/          # 相机与手眼标定工具
├── tests/                # 功能测试与离线测试程序
├── configs/              # YAML 配置文件
├── assets/               # 模型、引擎与测试素材
├── logs/                 # 运行日志
└── build/                # 构建输出目录
```

## 相机与通信

`io/` 目录当前包含：

- 海康相机驱动
- 迈德威视相机驱动
- USB 相机适配
- 串口通信
- 板卡与云台相关接口

其中串口目标会链接 `serial` 子模块，运行前建议固定设备名，例如将实际设备映射为 `/dev/gimbal`。

示例流程：

```bash
sudo usermod -a -G dialout $USER
udevadm info -a -n /dev/ttyACM0 | rg 'serial|idVendor|idProduct'
sudo vim /etc/udev/rules.d/99-usb-serial.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/gimbal
```

规则示例：

```text
SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="1234", ATTRS{serial}=="A1234567", SYMLINK+="gimbal"
```

## 标定与测试

标定相关目标：

- `capture`
- `calibrate_camera`
- `calibrate_handeye`
- `calibrate_robotworld_handeye`

测试与验证相关目标：

- `auto_aim_test`
- `camera_detect_test`
- `camera_test`
- `camera_pnp_test`
- `detector_video_test`
- `planner_test`
- `planner_test_offline`
- `camera_planner_realtime`

## 已知事项

- `autostart.sh` 仍保留了旧工程路径与旧可执行文件名，直接使用前需要按当前仓库路径和目标重新修改
- 部分配置文件中包含与本机相关的绝对路径，迁移环境时请优先检查
- 若在无显示环境运行，建议关闭 `enable_imshow`，仅保留 Web 调试或日志输出
