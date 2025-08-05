cmake_minimum_required(VERSION 3.8)
project(particle_filter_cpp)

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
# find_package(rclcpp_parameters REQUIRED)

# Optional: Find nav2_map_server for map loading
find_package(nav2_map_server QUIET)

# Find RangeLibc (external dependency)
find_path(RANGELIBC_INCLUDE_DIR 
    NAMES RangeLib.h
    PATHS 
        /usr/local/include/range_libc/includes
        /usr/include/range_libc/includes
        ${CMAKE_CURRENT_SOURCE_DIR}/../../range_libc/includes
        $ENV{HOME}/range_libc/includes
    DOC "RangeLibc include directory"
)

find_library(RANGELIBC_LIBRARY
    NAMES range_libc rangelibc
    PATHS
        /usr/local/lib
        /usr/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../../range_libc/build
        $ENV{HOME}/range_libc/build
    DOC "RangeLibc library"
)

if(RANGELIBC_INCLUDE_DIR AND RANGELIBC_LIBRARY)
    message(STATUS "Found RangeLibc: ${RANGELIBC_LIBRARY}")
    set(RANGELIBC_FOUND TRUE)
    add_definitions(-DUSE_RANGELIBC)
else()
    message(WARNING "RangeLibc not found. Ray casting will use slower fallback implementation.")
    set(RANGELIBC_FOUND FALSE)
endif()

# Include directories
include_directories(include)
if(RANGELIBC_FOUND)
    include_directories(${RANGELIBC_INCLUDE_DIR})
endif()

# Set dependencies list
set(dependencies
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  Eigen3
  pcl_ros
  pcl_conversions
  # rclcpp_parameters
)

# Create particle filter library with modular structure
add_library(${PROJECT_NAME}_lib SHARED
  src/particle_filter.cpp
  src/utils.cpp
  src/modules/particle.cpp
  src/modules/initialization.cpp
  src/modules/motion_model.cpp
  src/modules/sensor_model.cpp
  src/modules/resampling.cpp
  src/modules/pose_estimation.cpp
  src/modules/mcl_algorithm.cpp
)

ament_target_dependencies(${PROJECT_NAME}_lib ${dependencies})

# Create executable
add_executable(particle_filter_node 
  src/particle_filter_node.cpp
)

target_link_libraries(particle_filter_node ${PROJECT_NAME}_lib)
ament_target_dependencies(particle_filter_node ${dependencies})

# Link RangeLibc if found
if(RANGELIBC_FOUND)
    target_link_libraries(${PROJECT_NAME}_lib ${RANGELIBC_LIBRARY})
    target_link_libraries(particle_filter_node ${RANGELIBC_LIBRARY})
endif()

# Create component library for composition
rclcpp_components_register_nodes(${PROJECT_NAME}_lib "particle_filter_cpp::ParticleFilter")

# Install targets
install(TARGETS
  ${PROJECT_NAME}_lib
  particle_filter_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# Install headers
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
  FILES_MATCHING PATTERN "*.py" PATTERN "*.xml" PATTERN "*.yaml"
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config/
  FILES_MATCHING PATTERN "*.yaml" PATTERN "*.yml"
)

# Install maps
install(DIRECTORY maps/
  DESTINATION share/${PROJECT_NAME}/maps/
  FILES_MATCHING PATTERN "*.png" PATTERN "*.pgm" PATTERN "*.yaml"
)

# Install rviz config
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz/
  FILES_MATCHING PATTERN "*.rviz"
)

# Export dependencies - Fixed the export call
ament_export_dependencies(${dependencies})
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME}_lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  
  # Linting
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Unit tests
  ament_add_gtest(${PROJECT_NAME}_test
    test/test_particle_filter.cpp
    test/test_utils.cpp
  )
  
  if(TARGET ${PROJECT_NAME}_test)
    target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME}_lib)
    ament_target_dependencies(${PROJECT_NAME}_test ${dependencies})
  endif()
endif()

ament_package()
