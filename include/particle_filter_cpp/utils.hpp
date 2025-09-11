// ================================================================================================
// UTILITY FUNCTIONS HEADER - Helper functions for particle filter operations
// ================================================================================================
// Collection of geometric transformations, coordinate conversions, message utilities,
// and performance monitoring tools for Monte Carlo Localization
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__UTILS_HPP_
#define PARTICLE_FILTER_CPP__UTILS_HPP_

#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <random>
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
} // namespace geometry

// --------------------------------- COORDINATES NAMESPACE ---------------------------------
namespace coordinates
{
  // Map ↔ world coordinate system transformations
  Eigen::Vector3d map_to_world(const Eigen::Vector3d &map_coord, const nav_msgs::msg::MapMetaData &map_info);
  Eigen::Vector3d world_to_map(const Eigen::Vector3d &world_coord, const nav_msgs::msg::MapMetaData &map_info);
  void map_to_world_inplace(Eigen::MatrixXd &poses, const nav_msgs::msg::MapMetaData &map_info);
  
  // Vehicle coordinate frame transformations
  Eigen::Vector3d odom_to_rear_axle(const Eigen::Vector3d& odom_pose, double wheelbase);
  Eigen::Vector3d rear_axle_to_odom(const Eigen::Vector3d& rear_axle_pose, double wheelbase);
} // namespace coordinates

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
  
  // Smoothed timing measurements for performance analysis
  class Timer
  {
    public:
      explicit Timer(int smoothing_size = 10);  // Sliding window size
      void tick();                              // Record elapsed time
      double fps() const;                       // Calculate smoothed FPS
      double mean_duration() const;             // Mean duration [ms]

    private:
      std::vector<double> durations_;           // Circular buffer [ms]
      int smoothing_size_;                      // Window size
      int current_index_;                       // Current buffer position
      int count_;                              // Number of valid samples
      std::chrono::steady_clock::time_point last_time_;  // Last measurement time
  };
} // namespace performance

// --------------------------------- MESSAGE CONVERSIONS ---------------------------------

// Eigen → ROS message conversions for visualization
geometry_msgs::msg::PoseArray particles_to_pose_array(const Eigen::MatrixXd &particles);
geometry_msgs::msg::Pose eigen_to_pose(const Eigen::Vector3d &pose_vec);

// --------------------------------- DATA STRUCTURES ---------------------------------

// Generic circular buffer with statistical functions
template <typename T> class CircularArray
{
  public:
    explicit CircularArray(size_t size) : data_(size), size_(size), index_(0), count_(0)
    {
    }

    void push(const T &value)                 // Add new value (overwrite oldest)
    {
        data_[index_] = value;
        index_ = (index_ + 1) % size_;
        if (count_ < size_)
            count_++;
    }

    T mean() const                            // Arithmetic mean of stored values
    {
        if (count_ == 0)
            return T{};
        T sum = T{};
        for (size_t i = 0; i < count_; ++i)
        {
            sum += data_[i];
        }
        return sum / static_cast<T>(count_);
    }

    T median() const                          // Median of stored values
    {
        if (count_ == 0)
            return T{};
        std::vector<T> sorted_data(data_.begin(), data_.begin() + count_);
        std::sort(sorted_data.begin(), sorted_data.end());
        if (count_ % 2 == 0)
        {
            return (sorted_data[count_ / 2 - 1] + sorted_data[count_ / 2]) / T{2};
        }
        else
        {
            return sorted_data[count_ / 2];
        }
    }

    // Container interface
    size_t size() const { return count_; }    // Number of valid elements
    bool empty() const { return count_ == 0; }
    void clear() { count_ = 0; index_ = 0; }  // Reset buffer

  private:
    std::vector<T> data_;                     // Fixed-size storage
    size_t size_;                            // Maximum capacity
    size_t index_;                           // Next write position
    size_t count_;                           // Current valid elements
};

// --------------------------------- TEMPLATE IMPLEMENTATIONS ---------------------------------

} // namespace utils
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__UTILS_HPP_
