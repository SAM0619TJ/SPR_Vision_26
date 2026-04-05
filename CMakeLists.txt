cmake_minimum_required(VERSION 3.16.3)

# 必须在 project() 之前设置 CUDA 架构策略
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# ── CUDA 编译器检测 ────────────────────────────────────────────────────────────
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(CMAKE_CUDA_COMPILER
    NAMES nvcc
    PATHS
      /usr/local/cuda-11.4/bin   # JetPack 5.x Xavier NX
      /usr/local/cuda-11.8/bin
      /usr/local/cuda-12/bin
      /usr/local/cuda/bin
    NO_DEFAULT_PATH
  )
endif()

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      # Xavier NX = SM 7.2 (Volta)
      # 去掉 Orin 的 87，只保留 Xavier NX 目标，加快编译速度
      set(CMAKE_CUDA_ARCHITECTURES "72")
    else()
      set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
    endif()
  endif()
  project(SPR_Vision_26 LANGUAGES CXX CUDA)
  message(STATUS "CUDA enabled, arch: ${CMAKE_CUDA_ARCHITECTURES}")
  set(CUDA_FOUND TRUE)
else()
  project(SPR_Vision_26 LANGUAGES CXX)
  message(STATUS "CUDA not found, building without CUDA support")
  set(CUDA_FOUND FALSE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Xavier NX ARM FP16 指令集优化
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8.2-a+fp16 -mtune=cortex-a57")
  message(STATUS "ARM64 optimizations enabled: -march=armv8.2-a+fp16")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()

# ── 依赖查找 ──────────────────────────────────────────────────────────────────
find_package(OpenCV 4.5 REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

# OpenVINO 可选（Xavier NX 上通常不用）
find_package(OpenVINO QUIET COMPONENTS Runtime)
if(OpenVINO_FOUND)
  message(STATUS "OpenVINO found")
else()
  message(STATUS "OpenVINO not found: OpenVINO backend disabled")
endif()

# ── 架构相关路径 ───────────────────────────────────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(ARCH_SUFFIX "x86_64-linux-gnu")
  set(CUDA_ARCH_DIR "x86_64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(ARCH_SUFFIX "aarch64-linux-gnu")
  set(CUDA_ARCH_DIR "aarch64-linux")
else()
  set(ARCH_SUFFIX "x86_64-linux-gnu")
  set(CUDA_ARCH_DIR "x86_64-linux")
endif()

# ── TensorRT 查找 ─────────────────────────────────────────────────────────────
# JetPack 5.x 的 TRT 8.5 头文件在系统路径，无需手动指定 TRT-10.0 路径
if(CUDA_FOUND)
  get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CUDA_COMPILER_BIN}" DIRECTORY)

  find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS
      /usr/include/${ARCH_SUFFIX}
      /usr/local/cuda/include
      /usr/include
    PATH_SUFFIXES tensorrt)

  find_library(TensorRT_LIBRARY nvinfer
    HINTS /usr/lib/${ARCH_SUFFIX} /usr/local/cuda/lib64 /usr/lib)
  find_library(TensorRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS /usr/lib/${ARCH_SUFFIX} /usr/local/cuda/lib64 /usr/lib)
  find_library(TensorRT_ONNX_LIBRARY nvonnxparser
    HINTS /usr/lib/${ARCH_SUFFIX} /usr/local/cuda/lib64 /usr/lib)
  find_library(CUDA_RUNTIME_LIBRARY cudart
    HINTS
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      /usr/local/cuda/lib64
      /usr/local/cuda-11.4/targets/${CUDA_ARCH_DIR}/lib
      /usr/local/cuda/targets/${CUDA_ARCH_DIR}/lib)

  if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY AND TensorRT_PLUGIN_LIBRARY
     AND TensorRT_ONNX_LIBRARY AND CUDA_RUNTIME_LIBRARY)
    set(TensorRT_FOUND TRUE)
    set(TensorRT_LIBRARIES
      ${TensorRT_LIBRARY} ${TensorRT_PLUGIN_LIBRARY}
      ${TensorRT_ONNX_LIBRARY} ${CUDA_RUNTIME_LIBRARY})
    message(STATUS "TensorRT found: ${TensorRT_INCLUDE_DIR}")
  else()
    set(TensorRT_FOUND FALSE)
    message(WARNING "TensorRT not found — TRT backend disabled")
  endif()
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
if(TensorRT_FOUND)
  include_directories(${TensorRT_INCLUDE_DIR})
endif()

# ── ROS2 可选 ─────────────────────────────────────────────────────────────────
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(visualization_msgs QUIET)
find_package(tf2_ros QUIET)
find_package(geometry_msgs QUIET)

# ── 子目录 ────────────────────────────────────────────────────────────────────
add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(debug)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)
add_subdirectory(librm)

# ── 可执行文件 ────────────────────────────────────────────────────────────────
add_executable(standard src/standard.cpp)
target_link_libraries(standard fmt::fmt yaml-cpp auto_aim debug tools io ${OpenCV_LIBS})

add_executable(daheng_camera_test src/daheng_camera_test.cpp)
target_link_libraries(daheng_camera_test fmt::fmt yaml-cpp auto_aim tools io ${OpenCV_LIBS})

add_executable(uav_debug_jetson src/uav_debug_jetsn.cpp)
target_link_libraries(uav_debug_jetson fmt::fmt yaml-cpp auto_aim tools io ${OpenCV_LIBS})

# ── 标定工具 ──────────────────────────────────────────────────────────────────
add_executable(capture calibration/capture.cpp)
target_link_libraries(capture fmt::fmt tools io ${OpenCV_LIBS})

add_executable(calibrate_camera calibration/calibrate_camera.cpp)
target_link_libraries(calibrate_camera fmt::fmt yaml-cpp tools ${OpenCV_LIBS})

add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
target_link_libraries(calibrate_handeye fmt::fmt yaml-cpp tools ${OpenCV_LIBS})

# ── 测试 ──────────────────────────────────────────────────────────────────────
add_executable(auto_aim_test tests/auto_aim_test.cpp)
target_link_libraries(auto_aim_test fmt::fmt yaml-cpp tools io auto_aim ${OpenCV_LIBS})

add_executable(camera_detect_test tests/camera_detect_test.cpp)
target_link_libraries(camera_detect_test fmt::fmt yaml-cpp tools auto_aim io ${OpenCV_LIBS})

add_executable(camera_pnp_test tests/camera_pnp_test.cpp)
target_link_libraries(camera_pnp_test fmt::fmt yaml-cpp tools auto_aim io ${OpenCV_LIBS})

add_executable(detector_video_test tests/detector_video_test.cpp)
target_link_libraries(detector_video_test fmt::fmt yaml-cpp tools auto_aim ${OpenCV_LIBS})

add_executable(planner_test tests/planner_test.cpp)
target_link_libraries(planner_test fmt::fmt yaml-cpp auto_aim tools io ${OpenCV_LIBS})

add_executable(planner_test_offline tests/planner_test_offline.cpp)
target_link_libraries(planner_test_offline fmt::fmt yaml-cpp auto_aim tools io ${OpenCV_LIBS})

add_executable(camera_planner_realtime tests/camera_planner_realtime.cpp)
target_link_libraries(camera_planner_realtime fmt::fmt yaml-cpp auto_aim tools io ${OpenCV_LIBS})

# ── ROS2 可选编译 ─────────────────────────────────────────────────────────────
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND
   AND visualization_msgs_FOUND AND tf2_ros_FOUND AND geometry_msgs_FOUND)
  message(STATUS "ROS2 found, compiling standard_ros2")
  add_executable(standard_ros2 src/standard.cpp)
  target_link_libraries(standard_ros2
    fmt::fmt yaml-cpp auto_aim tools io rclcpp::rclcpp ${OpenCV_LIBS})
  ament_target_dependencies(standard_ros2
    rclcpp std_msgs visualization_msgs tf2_ros geometry_msgs)
  target_compile_definitions(standard_ros2 PRIVATE AMENT_CMAKE_FOUND)
else()
  message(STATUS "ROS2 not found, skipping standard_ros2")
endif()
