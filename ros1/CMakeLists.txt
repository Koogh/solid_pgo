cmake_minimum_required(VERSION 2.8.3)
project(solid_pgo)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
  pcl_conversions
)

find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED QUIET)
find_package(Eigen3 REQUIRED)

include_directories(
  # core/ headers (ROS-free: SOLiDManager, PoseGraphManager, common)
  ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
  # SOLiD source
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/SOLiD/include
  # catkin / system
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs sensor_msgs tf pcl_conversions
  DEPENDS PCL EIGEN3
)

add_executable(solid_pgo_node
  src/solid_pgo_node.cpp
  ../third_party/SOLiD/src/solid.cpp
)

target_compile_options(solid_pgo_node PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(solid_pgo_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
  gtsam
)
