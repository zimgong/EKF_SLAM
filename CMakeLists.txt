cmake_minimum_required(VERSION 3.0.2)
project(mapping)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
    tf2_geometry_msgs
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/ekf_slam_node.py
  scripts/slam_main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 
