cmake_minimum_required(VERSION 3.8)
project(obstacle_avoidance)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)
# find_package(obstacle_avoidance REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(tf2 REQUIRED)

include_directories(${rclpy_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${GAZEBO_INCLUDE_DIRS})
include_directories(${GAZEBO_LIBRARY_DIRS})

install(
  DIRECTORY src launch rviz config urdf scripts world ${CMAKE_CURRENT_BINARY_DIR} 
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
	DESTINATION include/
)

set(dependencies
   "geometry_msgs"
   "nav_msgs"
   "rclpy"
   "sensor_msgs"
   "tf2"
)

set(srv_files
  "scripts/DQN/srv/Dqn.srv"
)

#add_executable(DQN scripts/test.py)
#ament_target_dependencies(
#	DQN
#	geometry_msgs
#	nav_msgs
#	sensor_msgs
#	rclpy
#)

#install(TARGETS
#	DQN
#	DESTINATION lib/${PROHECT_NAME}
#)

rosidl_generate_interfaces(${PROJECT_NAME}
	${srv_files}
	DEPENDENCIES std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(include)
ament_export_dependencies(gazebo_ros_pkgs)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(nav_msgs)
ament_export_dependencies(rclpy)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(tf2)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
