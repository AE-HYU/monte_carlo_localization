// ================================================================================================
// UTILITY FUNCTIONS HEADER - Helper functions for particle filter operations
// ================================================================================================
// Collection of geometric transformations, coordinate conversions, message utilities,
// and performance monitoring tools for Monte Carlo Localization
// ================================================================================================

#ifndef MCL_PKG__UTILS_HPP_
#define MCL_PKG__UTILS_HPP_

#include <Eigen/Dense>
#include <functional>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <vector>

namespace mcl_pkg
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
  
} // namespace geometry


// --------------------------------- TRANSFORMS NAMESPACE ---------------------------------
namespace transforms
{
  // Coordinate frame transformations
  Eigen::Vector3d apply_laser_to_base_offset(const Eigen::Vector3d& pose_in_laser_frame,
                                              double laser_offset_x, double laser_offset_y);

  Eigen::Vector3d calculate_lidar_frame_motion(const Eigen::Vector3d& current_pose,
                                                const Eigen::Vector3d& previous_pose);
} // namespace transforms

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
    // Cumulative times (for averaging over 100 updates)
    double total_mcl_time = 0.0;
    double ray_casting_time = 0.0;
    double sensor_model_time = 0.0;
    double motion_model_time = 0.0;
    double resampling_time = 0.0;
    double query_prep_time = 0.0;
    double quality_metrics_time = 0.0;  // Time for covariance/max_weight/spread calculation
    int measurement_count = 0;

    // Current update times (latest single measurement)
    double current_query_prep_time = 0.0;
    double current_ray_casting_time = 0.0;
    double current_sensor_model_time = 0.0;

    // ESS statistics
    double ess_sum = 0.0;          // Cumulative ESS for averaging
    int ess_count = 0;             // Number of ESS measurements
    int resample_count = 0;        // Number of times resampled

    // TF lookup statistics
    int tf_exact_time_count = 0;   // Number of successful exact time TF lookups
    int tf_fallback_count = 0;     // Number of fallback to TimePointZero
    int tf_total_count = 0;        // Total TF lookup attempts

    void reset();
    void print_stats(const std::function<void(const std::string&)>& logger) const;
  };
  
} // namespace performance

// --------------------------------- MESSAGE CONVERSIONS ---------------------------------

// Eigen → ROS message conversions for visualization
geometry_msgs::msg::PoseArray particles_to_pose_array(const Eigen::MatrixXd &particles);


} // namespace utils
} // namespace mcl_pkg

#endif // MCL_PKG__UTILS_HPP_
