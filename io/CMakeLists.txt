cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)
# 创建目标 io
add_library(io STATIC
    hikrobot/hikrobot.cpp
    mindvision/mindvision.cpp
    usbcamera/usbcamera.cpp
    camera.cpp
    cboard.cpp
    dm_imu/dm_imu.cpp
    gimbal/gimbal.cpp
)
# target_include_directories(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# hikrobot
target_include_directories(io PUBLIC hikrobot/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

# mindvision
target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_link_libraries(io  tools MvCameraControl MVSDK usb-1.0 yaml-cpp serial)

################## 检查 ROS2 Foxy 环境 ##################
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(tf2_ros QUIET)
find_package(geometry_msgs QUIET)

# TF2 支持（用于 CBoard 实时发布 TF）
if(ament_cmake_FOUND AND rclcpp_FOUND AND tf2_ros_FOUND AND geometry_msgs_FOUND)
    message(STATUS "ROS2 Foxy: TF2 support found, enabling real-time TF publishing in CBoard.")
    target_compile_definitions(io PRIVATE AMENT_CMAKE_FOUND)
    ament_target_dependencies(io rclcpp tf2_ros geometry_msgs)
endif()

# ROS2 扩展代码（需要 std_msgs）
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND)
    message(STATUS "ROS2 Foxy: environment found, compiling ROS2-related code.")
    target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp)
    ament_target_dependencies(io rclcpp std_msgs)
else()
    message(STATUS "ROS2 not found, skipping ROS2 specific code.")
endif()
