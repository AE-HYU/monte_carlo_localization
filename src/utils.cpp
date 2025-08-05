#include "particle_filter_cpp/utils.hpp"
#include "particle_filter_cpp/modules/particle.hpp"
#include "particle_filter_cpp/particle_filter.hpp"
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
    tf2::Quaternion tf_q;
    tf2::fromMsg(q, tf_q);
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

bool is_valid_point(const Eigen::Vector2d& point, const nav_msgs::msg::OccupancyGrid& map)
{
    int x = static_cast<int>(point.x());
    int y = static_cast<int>(point.y());
    
    if (x < 0 || x >= static_cast<int>(map.info.width) || 
        y < 0 || y >= static_cast<int>(map.info.height)) {
        return false;
    }
    
    int index = y * map.info.width + x;
    return map.data[index] == 0; // 0 = free space
}

double get_map_value(const Eigen::Vector2d& point, const nav_msgs::msg::OccupancyGrid& map)
{
    int x = static_cast<int>(point.x());
    int y = static_cast<int>(point.y());
    
    if (x < 0 || x >= static_cast<int>(map.info.width) || 
        y < 0 || y >= static_cast<int>(map.info.height)) {
        return -1.0; // Unknown
    }
    
    int index = y * map.info.width + x;
    return static_cast<double>(map.data[index]);
}

std::vector<Eigen::Vector2d> get_free_space_points(const nav_msgs::msg::OccupancyGrid& map)
{
    std::vector<Eigen::Vector2d> free_points;
    
    for (unsigned int y = 0; y < map.info.height; ++y) {
        for (unsigned int x = 0; x < map.info.width; ++x) {
            int index = y * map.info.width + x;
            if (map.data[index] == 0) { // Free space
                // Convert to world coordinates
                Eigen::Vector3d map_coord(x, y, 0.0);
                Eigen::Vector3d world_coord = map_to_world(map_coord, map.info);
                free_points.emplace_back(world_coord.x(), world_coord.y());
            }
        }
    }
    
    return free_points;
}

geometry_msgs::msg::PoseArray particles_to_pose_array(const std::vector<modules::Particle>& particles)
{
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.poses.reserve(particles.size());
    
    for (const auto& particle : particles) {
        pose_array.poses.push_back(particle_to_pose(particle));
    }
    
    return pose_array;
}

geometry_msgs::msg::Pose particle_to_pose(const modules::Particle& particle)
{
    geometry_msgs::msg::Pose pose;
    pose.position.x = particle.x;
    pose.position.y = particle.y;
    pose.position.z = 0.0;
    pose.orientation = yaw_to_quaternion(particle.theta);
    return pose;
}

std::vector<modules::Particle> pose_array_to_particles(const geometry_msgs::msg::PoseArray& pose_array)
{
    std::vector<modules::Particle> particles;
    particles.reserve(pose_array.poses.size());
    
    for (const auto& pose : pose_array.poses) {
        double x = pose.position.x;
        double y = pose.position.y;
        double theta = quaternion_to_yaw(pose.orientation);
        particles.emplace_back(x, y, theta);
    }
    
    return particles;
}

std::vector<double> normalize_weights(const std::vector<double>& weights)
{
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    std::vector<double> normalized_weights;
    normalized_weights.reserve(weights.size());
    
    if (sum > 0.0) {
        for (double weight : weights) {
            normalized_weights.push_back(weight / sum);
        }
    } else {
        double uniform_weight = 1.0 / weights.size();
        normalized_weights.assign(weights.size(), uniform_weight);
    }
    
    return normalized_weights;
}

double compute_effective_sample_size(const std::vector<double>& weights)
{
    double sum_squared = 0.0;
    for (double weight : weights) {
        sum_squared += weight * weight;
    }
    return 1.0 / sum_squared;
}

std::vector<int> systematic_resampling(const std::vector<double>& weights, int num_samples)
{
    std::vector<int> indices;
    indices.reserve(num_samples);
    
    // Create cumulative sum
    std::vector<double> cumsum(weights.size());
    std::partial_sum(weights.begin(), weights.end(), cumsum.begin());
    
    // Systematic resampling
    double step = 1.0 / num_samples;
    double start = (static_cast<double>(rand()) / RAND_MAX) * step;
    
    int current_index = 0;
    for (int i = 0; i < num_samples; ++i) {
        double target = start + i * step;
        
        while (current_index < static_cast<int>(cumsum.size()) - 1 && 
               cumsum[current_index] < target) {
            current_index++;
        }
        
        indices.push_back(current_index);
    }
    
    return indices;
}

std::vector<int> low_variance_resampling(const std::vector<double>& weights, int num_samples)
{
    // This is the same as systematic resampling in this implementation
    return systematic_resampling(weights, num_samples);
}

std::vector<double> generate_gaussian_noise(int count, double mean, double std_dev, std::mt19937& gen)
{
    std::normal_distribution<double> dist(mean, std_dev);
    std::vector<double> noise;
    noise.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        noise.push_back(dist(gen));
    }
    
    return noise;
}

std::vector<double> generate_uniform_noise(int count, double min, double max, std::mt19937& gen)
{
    std::uniform_real_distribution<double> dist(min, max);
    std::vector<double> noise;
    noise.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        noise.push_back(dist(gen));
    }
    
    return noise;
}

double cast_ray(const Eigen::Vector2d& start, double angle, double max_range, 
                const nav_msgs::msg::OccupancyGrid& map)
{
    // Simple ray casting implementation using DDA or Bresenham-like algorithm
    double step_size = map.info.resolution / 2.0; // Half pixel resolution
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    
    for (double range = 0.0; range < max_range; range += step_size) {
        Eigen::Vector2d point = start + Eigen::Vector2d(range * cos_a, range * sin_a);
        
        // Convert to map coordinates
        Eigen::Vector3d world_coord(point.x(), point.y(), 0.0);
        Eigen::Vector3d map_coord = world_to_map(world_coord, map.info);
        
        if (!is_valid_point(Eigen::Vector2d(map_coord.x(), map_coord.y()), map)) {
            return range;
        }
        
        double map_value = get_map_value(Eigen::Vector2d(map_coord.x(), map_coord.y()), map);
        if (map_value > 50) { // Occupied threshold
            return range;
        }
    }
    
    return max_range;
}

std::vector<double> cast_rays(const Eigen::Vector3d& pose, const std::vector<double>& angles,
                              double max_range, const nav_msgs::msg::OccupancyGrid& map)
{
    std::vector<double> ranges;
    ranges.reserve(angles.size());
    
    Eigen::Vector2d start(pose.x(), pose.y());
    
    for (double angle_offset : angles) {
        double absolute_angle = pose.z() + angle_offset;
        double range = cast_ray(start, absolute_angle, max_range, map);
        ranges.push_back(range);
    }
    
    return ranges;
}

Timer::Timer(int smoothing_size) 
    : smoothing_size_(smoothing_size), current_index_(0), count_(0)
{
    durations_.resize(smoothing_size_);
    last_time_ = std::chrono::steady_clock::now();
}

void Timer::tick()
{
    auto current_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - last_time_).count();
    
    durations_[current_index_] = duration / 1000000.0; // Convert to seconds
    current_index_ = (current_index_ + 1) % smoothing_size_;
    
    if (count_ < smoothing_size_) {
        count_++;
    }
    
    last_time_ = current_time;
}

double Timer::fps() const
{
    if (count_ == 0) return 0.0;
    
    double total_duration = 0.0;
    for (int i = 0; i < count_; ++i) {
        total_duration += durations_[i];
    }
    
    return count_ / total_duration;
}

double Timer::mean_duration() const
{
    if (count_ == 0) return 0.0;
    
    double total_duration = 0.0;
    for (int i = 0; i < count_; ++i) {
        total_duration += durations_[i];
    }
    
    return total_duration / count_;
}

} // namespace utils
} // namespace particle_filter_cpp