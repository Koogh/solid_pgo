cmake_minimum_required(VERSION 3.8)
project(solid_pgo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

# ROS2 dependencies
find_package(ament_cmake     REQUIRED)
find_package(rclcpp          REQUIRED)
find_package(sensor_msgs     REQUIRED)
find_package(nav_msgs        REQUIRED)
find_package(geometry_msgs   REQUIRED)
find_package(tf2             REQUIRED)
find_package(tf2_ros         REQUIRED)
find_package(pcl_conversions REQUIRED)

# Non-ROS dependencies
find_package(PCL     REQUIRED)
find_package(GTSAM   REQUIRED QUIET)
find_package(Eigen3  REQUIRED)
find_package(OpenMP  REQUIRED)

include_directories(
  # core/ headers (ROS-free)
  ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
  # SOLiD source
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/SOLiD/include
  # system
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(solid_pgo_node
  src/solid_pgo_node.cpp
  ../third_party/SOLiD/src/solid.cpp
)

target_compile_options(solid_pgo_node PRIVATE ${OpenMP_CXX_FLAGS})

ament_target_dependencies(solid_pgo_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  pcl_conversions
)

target_link_libraries(solid_pgo_node
  ${PCL_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam
)

# Install
install(TARGETS solid_pgo_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/launch
  ${CMAKE_CURRENT_SOURCE_DIR}/config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
