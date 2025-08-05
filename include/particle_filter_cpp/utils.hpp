#ifndef PARTICLE_FILTER_CPP__UTILS_HPP_
#define PARTICLE_FILTER_CPP__UTILS_HPP_

#include <Eigen/Dense>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <vector>
#include <random>
#include <chrono>

namespace particle_filter_cpp
{
namespace utils
{

// Quaternion/angle conversion utilities  
double quaternion_to_yaw(const geometry_msgs::msg::Quaternion& q);
geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw);
double normalize_angle(double angle);

// Coordinate transformations
Eigen::Vector3d map_to_world(const Eigen::Vector3d& map_coord, const nav_msgs::msg::MapMetaData& map_info);
Eigen::Vector3d world_to_map(const Eigen::Vector3d& world_coord, const nav_msgs::msg::MapMetaData& map_info);

// ROS message utilities for particles
geometry_msgs::msg::PoseArray particles_to_pose_array(const Eigen::MatrixXd& particles);
geometry_msgs::msg::Pose eigen_to_pose(const Eigen::Vector3d& pose_vec);

// Matrix utilities
Eigen::Matrix2d rotation_matrix(double angle);
void map_to_world_inplace(Eigen::MatrixXd& poses, const nav_msgs::msg::MapMetaData& map_info);

// Timer utility class
class Timer
{
public:
    explicit Timer(int smoothing_size = 10);
    void tick();
    double fps() const;
    double mean_duration() const;

private:
    std::vector<double> durations_;
    int smoothing_size_;
    int current_index_;
    int count_;
    std::chrono::steady_clock::time_point last_time_;
};

// Circular array utility class
template<typename T>
class CircularArray
{
public:
    explicit CircularArray(size_t size) : data_(size), size_(size), index_(0), count_(0) {}
    
    void push(const T& value) {
        data_[index_] = value;
        index_ = (index_ + 1) % size_;
        if (count_ < size_) count_++;
    }
    
    T mean() const {
        if (count_ == 0) return T{};
        T sum = T{};
        for (size_t i = 0; i < count_; ++i) {
            sum += data_[i];
        }
        return sum / static_cast<T>(count_);
    }
    
    T median() const {
        if (count_ == 0) return T{};
        std::vector<T> sorted_data(data_.begin(), data_.begin() + count_);
        std::sort(sorted_data.begin(), sorted_data.end());
        if (count_ % 2 == 0) {
            return (sorted_data[count_/2 - 1] + sorted_data[count_/2]) / T{2};
        } else {
            return sorted_data[count_/2];
        }
    }
    
    size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }
    void clear() { count_ = 0; index_ = 0; }

private:
    std::vector<T> data_;
    size_t size_;
    size_t index_;
    size_t count_;
};

} // namespace utils
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__UTILS_HPP_