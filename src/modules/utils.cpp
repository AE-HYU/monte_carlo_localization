// ================================================================================================
// UTILITY FUNCTIONS - Helper functions for particle filter operations
// ================================================================================================
// Geometric transformations, coordinate conversions, and performance monitoring utilities
// for Monte Carlo Localization
// ================================================================================================

#include "mcl_pkg/utils.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace mcl_pkg
{
namespace utils
{

// --------------------------------- GEOMETRY NAMESPACE ---------------------------------
namespace geometry
{

// Convert quaternion to yaw angle (Z-axis rotation)
double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q)
{
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    return yaw;
}

// Convert yaw angle to quaternion (pure Z-axis rotation)
geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw)
{
    tf2::Quaternion tf_q;
    tf_q.setRPY(0.0, 0.0, yaw);  // Roll=0, Pitch=0, Yaw=angle
    return tf2::toMsg(tf_q);
}

// Normalize angle to [-π, π] range
double normalize_angle(double angle)
{
    // Fast normalization using fmod instead of while loop
    // This is O(1) instead of O(n) for large angles
    angle = std::fmod(angle + M_PI, 2.0 * M_PI);
    if (angle < 0) angle += 2.0 * M_PI;
    return angle - M_PI;
}

// Generate 2D rotation matrix R(θ)
Eigen::Matrix2d rotation_matrix(double angle)
{
    Eigen::Matrix2d rot;
    rot << std::cos(angle), -std::sin(angle),   // [cos(θ)  -sin(θ)]
           std::sin(angle),  std::cos(angle);   // [sin(θ)   cos(θ)]
    return rot;
}


} // namespace geometry


// --------------------------------- TRANSFORMS NAMESPACE ---------------------------------
namespace transforms
{

// Apply static laser-to-base_link offset using cached values
Eigen::Vector3d apply_laser_to_base_offset(const Eigen::Vector3d& pose_in_laser_frame,
                                            double laser_offset_x, double laser_offset_y)
{
    // Transform: base_link = laser - offset rotated by particle heading
    double cos_theta = std::cos(pose_in_laser_frame[2]);
    double sin_theta = std::sin(pose_in_laser_frame[2]);

    // Rotate offset by particle heading and subtract from laser position
    return Eigen::Vector3d(
        pose_in_laser_frame[0] - (laser_offset_x * cos_theta - laser_offset_y * sin_theta),
        pose_in_laser_frame[1] - (laser_offset_x * sin_theta + laser_offset_y * cos_theta),
        pose_in_laser_frame[2]  // Heading unchanged
    );
}

// Calculate motion between two poses (for lidar frame motion calculation)
Eigen::Vector3d calculate_lidar_frame_motion(const Eigen::Vector3d& current_pose,
                                              const Eigen::Vector3d& previous_pose)
{
    // Calculate global displacement
    Eigen::Vector2d delta_global = current_pose.head<2>() - previous_pose.head<2>();

    // Calculate angular difference and normalize to [-π, π] for shortest path
    double delta_theta = geometry::normalize_angle(current_pose[2] - previous_pose[2]);

    // Transform global displacement to robot-local coordinates using previous pose
    Eigen::Matrix2d rot = geometry::rotation_matrix(-previous_pose[2]);
    Eigen::Vector2d delta_local = rot * delta_global;

    // Return motion in robot frame: [forward, lateral, rotation]
    return Eigen::Vector3d(delta_local[0], delta_local[1], delta_theta);
}

} // namespace transforms


// --------------------------------- VALIDATION NAMESPACE ---------------------------------
namespace validation
{

// Check if pose contains valid finite values within reasonable range
bool is_pose_valid(const Eigen::Vector3d& pose, double max_range)
{
    return std::isfinite(pose[0]) && std::isfinite(pose[1]) && std::isfinite(pose[2]) &&
           std::abs(pose[0]) < max_range && std::abs(pose[1]) < max_range;
}

} // namespace validation

// --------------------------------- PERFORMANCE NAMESPACE ---------------------------------
namespace performance
{

// Reset all timing statistics
void TimingStats::reset()
{
    total_mcl_time = 0.0;
    ray_casting_time = 0.0;
    sensor_model_time = 0.0;
    motion_model_time = 0.0;
    resampling_time = 0.0;
    query_prep_time = 0.0;
    quality_metrics_time = 0.0;
    measurement_count = 0;

    // Reset ESS statistics
    ess_sum = 0.0;
    ess_count = 0;
    resample_count = 0;

    // Reset TF lookup statistics
    tf_exact_time_count = 0;
    tf_fallback_count = 0;
    tf_total_count = 0;
}

// Print performance statistics using provided logger function
void TimingStats::print_stats(const std::function<void(const std::string&)>& logger) const
{
    if (measurement_count == 0)
        return;
        
    double avg_total = total_mcl_time / measurement_count;
    double avg_raycast = ray_casting_time / measurement_count;
    double avg_sensor = sensor_model_time / measurement_count;
    double avg_motion = motion_model_time / measurement_count;
    double avg_resample = resampling_time / measurement_count;
    double avg_query = query_prep_time / measurement_count;
    
    logger("=== PERFORMANCE STATS (last " + std::to_string(measurement_count) + " iterations) ===");
    logger("Total MCL:        " + std::to_string(avg_total) + " ms/iter (" + std::to_string(1000.0/avg_total) + " Hz)");
    logger("Ray casting:      " + std::to_string(avg_raycast) + " ms/iter (" + std::to_string(100.0*avg_raycast/avg_total) + "%)");
    logger("Sensor eval:      " + std::to_string(avg_sensor) + " ms/iter (" + std::to_string(100.0*avg_sensor/avg_total) + "%) [lookup tables only]");
    logger("Query prep:       " + std::to_string(avg_query) + " ms/iter (" + std::to_string(100.0*avg_query/avg_total) + "%)");
    logger("Motion model:     " + std::to_string(avg_motion) + " ms/iter (" + std::to_string(100.0*avg_motion/avg_total) + "%)");
    logger("Resampling:       " + std::to_string(avg_resample) + " ms/iter (" + std::to_string(100.0*avg_resample/avg_total) + "%)");

    // TF lookup statistics
    if (tf_total_count > 0) {
        double exact_time_rate = 100.0 * tf_exact_time_count / tf_total_count;
        double fallback_rate = 100.0 * tf_fallback_count / tf_total_count;
        logger("TF Lookup:        Exact=" + std::to_string(tf_exact_time_count) + " (" + std::to_string(exact_time_rate) + "%), " +
               "Fallback=" + std::to_string(tf_fallback_count) + " (" + std::to_string(fallback_rate) + "%)");
    }

    logger("=====================================");
}

} // namespace performance

// --------------------------------- MESSAGE CONVERSIONS ---------------------------------

// Convert particle matrix to ROS PoseArray for visualization
geometry_msgs::msg::PoseArray particles_to_pose_array(const Eigen::MatrixXd& particles)
{
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.poses.reserve(particles.rows());
    
    // Convert each particle [x, y, θ] to Pose message
    for (int i = 0; i < particles.rows(); ++i) {
        geometry_msgs::msg::Pose pose;
        pose.position.x = particles(i, 0);  // x coordinate
        pose.position.y = particles(i, 1);  // y coordinate
        pose.position.z = 0.0;              // 2D navigation (z = 0)
        pose.orientation = geometry::yaw_to_quaternion(particles(i, 2));  // θ → quaternion
        pose_array.poses.push_back(pose);
    }
    
    return pose_array;
}

} // namespace utils
} // namespace mcl_pkg