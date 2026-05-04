#!/bin/bash
set -e

cd /home/spr/SPR_Vision_26

# Hikrobot MVS runtime environment
export ALLUSERSPROFILE=/opt/MVS/MVFG
export MVCAM_GENICAM_CLPROTOCOL=/opt/MVS/lib/CLProtocol
export MVCAM_SDK_PATH=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

# Runtime libraries
export LD_LIBRARY_PATH=/opt/MVS/lib/aarch64:/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH

# Minimal PATH for systemd
export PATH=/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Wait for USB camera and serial devices to settle
sleep 15

echo "autoaim start $(date)" >> /home/spr/vision_autostart.log

exec ./build/auto_aim_debug_mpc configs/standard3_tensorrt.yaml