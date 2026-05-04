# Debug Diary

## 2026-05-04 Orin NX 自启动与 Web 调试问题复盘

### 背景

平台为 Jetson Orin NX，程序入口为：

```bash
./build/auto_aim_debug_mpc configs/standard3_tensorrt.yaml
```

硬件连接：

- Hikrobot USB 工业相机，`VID:PID = 2bdf:0001`
- CH340/CH341 USB 串口，`VID:PID = 1a86:7523`
- 串口通过 `/dev/gimbal` 访问

目标是使用 `systemd` 实现开机自启动，同时保留 Web 调试能力。

## 问题 1：systemd 自启动后 Hikrobot 相机枚举失败

### 现象

手动启动程序时相机可以正常工作，但通过 `systemd` 自启动后持续报错：

```text
MV_CC_EnumDevices failed: 0x80000006
MV_CC_StopGrabbing failed: 0x80000000
Reset usb successfully :)
Unable to open usb!
```

同时串口正常：

```text
[Gimbal] Serial /dev/gimbal @ 115200 baud
[Gimbal] First q received.
```

说明问题主要集中在 Hikrobot 相机/MVS SDK，而不是 USB 串口。

### 排查

确认相机在 USB 总线上可见：

```bash
lsusb | grep 2bdf
```

示例输出：

```text
Bus 002 Device 004: ID 2bdf:0001 Hikrobot MV-CS016-10UC
```

注意 `Device` 编号每次插拔或 reset 后可能变化，例如从 `003` 变成 `004`，所以不能在脚本中固定 `/dev/bus/usb/002/003` 这种路径。

确认当前 USB 节点权限：

```bash
ls -l /dev/bus/usb/002/004
```

当前权限已经是可访问的：

```text
crw-rw-rw-+ root plugdev ...
```

继续对比手动终端和 systemd 环境变量：

```bash
env | grep -E "MVS|MVCAM|GENICAM|LD_LIBRARY|PATH"
sudo systemctl show autoaim.service -p Environment
```

发现手动终端包含 MVS 环境：

```text
ALLUSERSPROFILE=/opt/MVS/MVFG
MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
MVCAM_SDK_PATH=/opt/MVS
MVCAM_COMMON_RUNENV=/opt/MVS/lib
LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:...
```

而 systemd 环境为空：

```text
Environment=
```

### 原因

`systemd` 不会继承用户交互 shell 的环境变量。Hikrobot MVS SDK 依赖 `MVCAM_*`、`ALLUSERSPROFILE` 和 `/opt/MVS/lib/aarch64` 等运行时环境。缺少这些变量时，手动运行正常，但 systemd 下 SDK 枚举不到相机。

### 解决方案

在 `autostart.sh` 中显式导出 MVS 环境变量：

```bash
#!/bin/bash
set -e

cd /home/spr/SPR_Vision_26

export ALLUSERSPROFILE=/opt/MVS/MVFG
export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
export MVCAM_SDK_PATH=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

sleep 15

echo "autoaim start $(date)" >> /home/spr/vision_autostart.log

exec ./build/auto_aim_debug_mpc configs/standard3_tensorrt.yaml
```

## 问题 2：Hikrobot USB 权限与设备号变化

### 现象

`lsusb` 中 Hikrobot 相机存在，但日志有时出现：

```text
Unable to open usb!
```

并且相机设备号会变化：

```text
Bus 002 Device 003
Bus 002 Device 004
```

### 原因

USB 设备 reset 或重新插拔后，Linux 分配的 `Device` 编号会变化。程序或脚本如果依赖固定 `/dev/bus/usb/002/003`，reset 后路径会失效。

另外，开机早期 udev 规则未生效时，也可能导致 `/dev/bus/usb/...` 权限不稳定。

### 解决方案

用 `VID:PID` 写 udev 规则，不固定 Bus/Device：

```text
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="2bdf", ATTR{idProduct}=="0001", MODE="0666", GROUP="plugdev", TAG+="uaccess"
```

保存为：

```bash
/etc/udev/rules.d/99-hikrobot.rules
```

生效：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

检查：

```bash
lsusb | grep 2bdf
ls -l /dev/bus/usb/<Bus>/<Device>
```

## 问题 3：`/tmp/vision_autostart.log` 权限导致服务反复失败

### 现象

服务启动后立刻失败：

```text
/home/spr/SPR_Vision_26/autostart.sh: 行 8: /tmp/vision_autostart.log: 权限不够
autoaim.service: Main process exited, code=exited, status=1/FAILURE
```

### 原因

`autostart.sh` 使用了：

```bash
set -e
echo "autostart $(date)" >> /tmp/vision_autostart.log
```

如果 `/tmp/vision_autostart.log` 之前由其他用户创建，当前服务用户无权限追加，脚本会因为 `set -e` 直接退出。

### 解决方案

不要写 `/tmp` 中可能残留权限异常的文件，改写到用户目录：

```bash
echo "autoaim start $(date)" >> /home/spr/vision_autostart.log
```

或者临时删除旧文件：

```bash
sudo rm -f /tmp/vision_autostart.log
```

## 问题 4：Web 调试页面一直加载

### 现象

程序已经正常运行，相机线程和 YOLO 推理也正常：

```text
HikRobot's capture thread started.
[YOLOV8_TRT] Frame 0 ...
```

但浏览器访问 Web 调试页面一直加载。检查端口发现：

```bash
ss -ltnp 'sport = :8080'
```

8080 端口被旧进程占用：

```text
0.0.0.0:8080 users:(("auto_aim_debug_",pid=21853,fd=48))
```

同时存在多个 `auto_aim_debug_mpc` 进程：

```text
18236 Tl ./build/auto_aim_debug_mpc ...
21853 Tl ./build/auto_aim_debug_mpc ...
21940 Tl ./build/auto_aim_debug_mpc ...
```

其中 `Tl` 表示进程被暂停。

### 原因

之前用 `Ctrl+Z` 停程序，实际上只是暂停进程，不是退出。暂停的旧进程仍然占着 8080 端口，新的程序无法绑定 Web 调试端口，浏览器访问到的是旧的卡死服务。

### 解决方案

清理旧进程：

```bash
sudo systemctl stop autoaim.service
kill -9 18236 21853 21940
```

或：

```bash
pkill -f auto_aim_debug_mpc
```

确认清理干净：

```bash
ps -eo pid,stat,cmd | awk '/auto_aim_debug_mpc/ && !/awk/ {print}'
ss -ltnp 'sport = :8080'
```

正常情况下不应再有自瞄进程，也不应有 8080 监听。

重新启动：

```bash
./build/auto_aim_debug_mpc configs/standard3_tensorrt.yaml
```

或：

```bash
sudo systemctl start autoaim.service
```

### 预防

停止程序时使用：

```text
Ctrl+C
```

不要使用：

```text
Ctrl+Z
```

`Ctrl+Z` 只会暂停进程，可能继续占用相机、串口和 8080 端口。

## 问题 5：WebSocket 坏连接可能拖慢 Web 调试

### 现象

旧 Web 页面或断开的浏览器连接可能残留，导致 `ss` 中出现大量：

```text
CLOSE-WAIT
FIN-WAIT-2
SYN-SENT
```

### 原因

原始 `WebDebugger` 在广播 WebSocket 帧时是同步发送，如果客户端断开或网络阻塞，可能影响 Web 服务响应。

### 解决方案

已修改 `debug/web_debugger.cpp`：

- HTTP socket 增加收发超时
- WebSocket 推帧使用 `MSG_DONTWAIT`
- 发送失败或发送不完整时清理客户端 fd
- `select()` 的 timeout 每轮重置
- `listen` backlog 从 `8` 提升到 `32`

修改后需要重新编译：

```bash
cmake --build build --target auto_aim_debug_mpc -j$(nproc)
sudo systemctl restart autoaim.service
```

## 当前推荐自启动配置

### `autoaim.service`

```ini
[Unit]
Description=SPR Vision Auto Aim
After=systemd-udev-settle.service multi-user.target
Wants=systemd-udev-settle.service

[Service]
Type=simple
User=spr
WorkingDirectory=/home/spr/SPR_Vision_26
ExecStart=/home/spr/SPR_Vision_26/autostart.sh
Restart=always
RestartSec=8
SupplementaryGroups=dialout plugdev video render

[Install]
WantedBy=multi-user.target
```

### 常用命令

```bash
sudo systemctl daemon-reload
sudo systemctl restart autoaim.service
journalctl -u autoaim.service -f -o cat
```

检查残留进程和 Web 端口：

```bash
ps -eo pid,stat,cmd | awk '/auto_aim_debug_mpc/ && !/awk/ {print}'
ss -ltnp 'sport = :8080'
```

## 结论

本次问题不是单一故障，而是多个因素叠加：

1. `systemd` 默认环境缺少 Hikrobot MVS SDK 环境变量，导致自启动时相机枚举失败。
2. USB reset 后设备号会变化，不能依赖固定 `/dev/bus/usb/...` 路径。
3. `/tmp` 日志文件权限异常会让 `set -e` 的启动脚本直接失败。
4. `Ctrl+Z` 暂停的旧进程占用 8080，导致 Web 调试一直加载。
5. WebSocket 坏连接需要及时清理，避免拖住 Web 调试服务。

最终稳定方案是：`systemd + 显式 MVS 环境变量 + udev 权限规则 + 避免 Ctrl+Z + 清理 WebSocket 坏连接`。
