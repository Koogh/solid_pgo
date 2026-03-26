cmake_minimum_required(VERSION 3.8)
project(solid_pgo)

# ---- C++ standard ----
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O2)
endif()

# ---- ROS 2 dependencies ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

# ---- Non-ROS dependencies ----
find_package(PCL REQUIRED COMPONENTS common io filters registration kdtree)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)

# ---- Include dirs ----
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

# ---- Executable ----
add_executable(solid_pgo_node src/solid_pgo_node.cpp)

ament_target_dependencies(solid_pgo_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  tf2
  pcl_ros
  pcl_conversions
)

target_link_libraries(solid_pgo_node
  ${PCL_LIBRARIES}
  gtsam
  Eigen3::Eigen
)

# ---- Install ----
install(TARGETS solid_pgo_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
