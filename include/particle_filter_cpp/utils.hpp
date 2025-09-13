// ================================================================================================
// UTILITY FUNCTIONS HEADER - Helper functions for particle filter operations
// ================================================================================================
// Collection of geometric transformations, coordinate conversions, message utilities,
// and performance monitoring tools for Monte Carlo Localization
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__UTILS_HPP_
#define PARTICLE_FILTER_CPP__UTILS_HPP_

#include <Eigen/Dense>
#include <functional>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <vector>

namespace particle_filter_cpp
{
namespace utils
{

// --------------------------------- GEOMETRY NAMESPACE ---------------------------------
namespace geometry
{
  // Quaternion ↔ Euler angle conversions
  double quaternion_to_yaw(const geometry_msgs::msg::Quaternion &q);      // Extract Z-axis rotation
  geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw);           // Create pure Z rotation
  double normalize_angle(double angle);                                   // Wrap to [-π, π]
  
  // 2D rotation matrix
  Eigen::Matrix2d rotation_matrix(double angle);                          // Generate R(θ)
  
  // Vehicle frame transformations
  Eigen::Vector3d apply_vehicle_offset(const Eigen::Vector3d& pose, double forward_offset);  // Apply lidar->base_link offset
} // namespace geometry


// --------------------------------- VALIDATION NAMESPACE ---------------------------------
namespace validation
{
  // Pose validation functions
  bool is_pose_valid(const Eigen::Vector3d& pose, double max_range = 10000.0);
} // namespace validation

// --------------------------------- PERFORMANCE NAMESPACE ---------------------------------
namespace performance
{
  // Timing statistics structure
  struct TimingStats
  {
    double total_mcl_time = 0.0;
    double ray_casting_time = 0.0;
    double sensor_model_time = 0.0;
    double motion_model_time = 0.0;
    double resampling_time = 0.0;
    double query_prep_time = 0.0;
    int measurement_count = 0;
    
    void reset();
    void print_stats(const std::function<void(const std::string&)>& logger) const;
  };
  
} // namespace performance

// --------------------------------- MESSAGE CONVERSIONS ---------------------------------

// Eigen → ROS message conversions for visualization
geometry_msgs::msg::PoseArray particles_to_pose_array(const Eigen::MatrixXd &particles);


} // namespace utils
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__UTILS_HPP_
