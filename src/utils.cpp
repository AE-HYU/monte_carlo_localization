// ================================================================================================
// UTILITY FUNCTIONS - Helper functions for particle filter operations
// ================================================================================================
// Geometric transformations, coordinate conversions, and performance monitoring utilities
// for Monte Carlo Localization
// ================================================================================================

#include "particle_filter_cpp/utils.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace particle_filter_cpp
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
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// Generate 2D rotation matrix R(θ)
Eigen::Matrix2d rotation_matrix(double angle)
{
    Eigen::Matrix2d rot;
    rot << std::cos(angle), -std::sin(angle),   // [cos(θ)  -sin(θ)]
           std::sin(angle),  std::cos(angle);   // [sin(θ)   cos(θ)]
    return rot;
}

// Apply vehicle frame offset (lidar -> base_link transformation)
Eigen::Vector3d apply_vehicle_offset(const Eigen::Vector3d& pose, double forward_offset)
{
    double cos_theta = std::cos(pose[2]);
    double sin_theta = std::sin(pose[2]);
    
    Eigen::Vector3d offset_pose;
    offset_pose[0] = pose[0] - forward_offset * cos_theta;  // x with offset
    offset_pose[1] = pose[1] - forward_offset * sin_theta;  // y with offset  
    offset_pose[2] = pose[2];                               // theta unchanged
    return offset_pose;
}

} // namespace geometry


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
    measurement_count = 0;
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
} // namespace particle_filter_cpp