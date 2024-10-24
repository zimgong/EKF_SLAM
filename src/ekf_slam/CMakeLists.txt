cmake_minimum_required(VERSION 3.0.2)
project(ekf_slam)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roslib
  roscpp
  tf2
  tf2_ros
  tf2_geometry_msgs
  std_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  dynamic_reconfigure
  mrpt_msgs_bridge
  roslaunch
)

#find mrpt packages
find_package(mrpt-gui REQUIRED)
find_package(mrpt-slam REQUIRED)
find_package(mrpt-ros1bridge REQUIRED)

message(STATUS "MRPT_VERSION: ${mrpt-slam_VERSION}")

if(CMAKE_COMPILER_IS_GNUCXX)
	# High level of warnings.
	# The -Wno-long-long is required in 64bit systems when including sytem headers.
	# The -Wno-variadic-macros was needed for Eigen3, StdVector.h
	add_compile_options(-Wall -Wno-long-long -Wno-variadic-macros)
	# Workaround: Eigen <3.4 produces *tons* of warnings in GCC >=6. See http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1221
	if(NOT ${CMAKE_CXX_COMPILER_VERSION} LESS "6.0")
		add_compile_options(-Wno-ignored-attributes -Wno-int-in-bool-context)
	endif()
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
  add_compile_options(-O3)
endif()

catkin_package(
	CATKIN_DEPENDS
	nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  mrpt_msgs_bridge
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  include
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(ekf_slam
    src/ekf_slam.cpp
    src/ekf_slam_wrapper.cpp
    src/ekf_slam_app.cpp
)

TARGET_LINK_LIBRARIES(ekf_slam
  mrpt::slam
  mrpt::gui
  mrpt::ros1bridge
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS ekf_slam
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(DIRECTORY
  config
  launch
  rviz
  tutorial
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

# Add testing for *.launch files
roslaunch_add_file_check(launch)
