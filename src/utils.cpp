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

double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q)
{
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    return yaw;
}

geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw)
{
    tf2::Quaternion tf_q;
    tf_q.setRPY(0.0, 0.0, yaw);
    return tf2::toMsg(tf_q);
}

double normalize_angle(double angle)
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

Eigen::Vector3d map_to_world(const Eigen::Vector3d& map_coord, const nav_msgs::msg::MapMetaData& map_info)
{
    double scale = map_info.resolution;
    double angle = quaternion_to_yaw(map_info.origin.orientation);
    
    // Apply rotation
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    double x_rot = cos_a * map_coord.x() - sin_a * map_coord.y();
    double y_rot = sin_a * map_coord.x() + cos_a * map_coord.y();
    
    // Apply scale and translation
    double x_world = x_rot * scale + map_info.origin.position.x;
    double y_world = y_rot * scale + map_info.origin.position.y;
    double theta_world = map_coord.z() + angle;
    
    return Eigen::Vector3d(x_world, y_world, normalize_angle(theta_world));
}

Eigen::Vector3d world_to_map(const Eigen::Vector3d& world_coord, const nav_msgs::msg::MapMetaData& map_info)
{
    double scale = map_info.resolution;
    double angle = -quaternion_to_yaw(map_info.origin.orientation);
    
    // Apply translation
    double x_trans = world_coord.x() - map_info.origin.position.x;
    double y_trans = world_coord.y() - map_info.origin.position.y;
    
    // Apply scale
    x_trans /= scale;
    y_trans /= scale;
    
    // Apply rotation
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    double x_map = cos_a * x_trans - sin_a * y_trans;
    double y_map = sin_a * x_trans + cos_a * y_trans;
    double theta_map = world_coord.z() + angle;
    
    return Eigen::Vector3d(x_map, y_map, normalize_angle(theta_map));
}

geometry_msgs::msg::PoseArray particles_to_pose_array(const Eigen::MatrixXd& particles)
{
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.poses.reserve(particles.rows());
    
    for (int i = 0; i < particles.rows(); ++i) {
        geometry_msgs::msg::Pose pose;
        pose.position.x = particles(i, 0);
        pose.position.y = particles(i, 1);
        pose.position.z = 0.0;
        pose.orientation = yaw_to_quaternion(particles(i, 2));
        pose_array.poses.push_back(pose);
    }
    
    return pose_array;
}

geometry_msgs::msg::Pose eigen_to_pose(const Eigen::Vector3d& pose_vec)
{
    geometry_msgs::msg::Pose pose;
    pose.position.x = pose_vec[0];
    pose.position.y = pose_vec[1];
    pose.position.z = 0.0;
    pose.orientation = yaw_to_quaternion(pose_vec[2]);
    return pose;
}

Eigen::Matrix2d rotation_matrix(double angle)
{
    Eigen::Matrix2d rot;
    rot << std::cos(angle), -std::sin(angle),
           std::sin(angle),  std::cos(angle);
    return rot;
}

void map_to_world_inplace(Eigen::MatrixXd& poses, const nav_msgs::msg::MapMetaData& map_info)
{
    for (int i = 0; i < poses.rows(); ++i) {
        auto world_pose = map_to_world(poses.row(i).transpose(), map_info);
        poses.row(i) = world_pose.transpose();
    }
}

Timer::Timer(int smoothing_size) 
    : durations_(smoothing_size), smoothing_size_(smoothing_size), current_index_(0), count_(0)
{
    last_time_ = std::chrono::steady_clock::now();
}

void Timer::tick()
{
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time_).count() / 1000.0;
    
    durations_[current_index_] = duration;
    current_index_ = (current_index_ + 1) % smoothing_size_;
    if (count_ < smoothing_size_) count_++;
    
    last_time_ = now;
}

double Timer::fps() const
{
    if (count_ == 0) return 0.0;
    double mean_dur = mean_duration();
    return mean_dur > 0 ? 1000.0 / mean_dur : 0.0;
}

double Timer::mean_duration() const
{
    if (count_ == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < count_; ++i) {
        sum += durations_[i];
    }
    return sum / count_;
}

} // namespace utils
} // namespace particle_filter_cpp