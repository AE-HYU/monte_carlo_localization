#ifndef PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/srv/get_map.hpp>

#include <eigen3/Eigen/Dense>
#include <vector>
#include <random>
#include <memory>
#include <mutex>

// RangeLibc integration (conditional compilation)
#ifdef USE_RANGELIBC
#include "RangeLib.h"
#include "RangeUtils.h"
#endif

namespace particle_filter_cpp
{

struct Particle
{
    double x;
    double y;
    double theta;
    double weight;
    
    Particle() : x(0.0), y(0.0), theta(0.0), weight(0.0) {}
    Particle(double x_, double y_, double theta_, double weight_ = 1.0) 
        : x(x_), y(y_), theta(theta_), weight(weight_) {}
};

class ParticleFilter : public rclcpp::Node
{
public:
    explicit ParticleFilter(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~ParticleFilter() = default;

private:
    // Callback functions
    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    
    // Map service callback
    void get_map();
    
    // Core MCL functions
    void motion_model(const nav_msgs::msg::Odometry& odom);
    void sensor_model(const sensor_msgs::msg::LaserScan& scan);
    void resampling();
    void update_filter();
    
    // Particle management
    void initialize_particles_uniform();
    void initialize_particles_pose(const geometry_msgs::msg::Pose& pose);
    Eigen::Vector3d compute_expected_pose();
    
    // Visualization
    void publish_particles();
    void publish_pose();
    void publish_transform();
    
    // Utility functions
    double compute_likelihood(const std::vector<float>& ranges, const Particle& particle);
    void normalize_weights();
    
    // RangeLibc specific functions
#ifdef USE_RANGELIBC
    void initialize_rangelib();
    void precompute_sensor_model();
#endif
    
    // Publishers and subscribers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
    
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    
    // TF
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Map service client
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr map_client_;
    
    // Parameters
    int num_particles_;
    int max_viz_particles_;
    double max_range_;
    int max_range_px_;
    double motion_dispersion_x_;
    double motion_dispersion_y_;
    double motion_dispersion_theta_;
    double z_hit_;
    double z_short_;
    double z_max_;
    double z_rand_;
    double sigma_hit_;
    std::string scan_topic_;
    std::string odom_topic_;
    std::string range_method_;
    int theta_discretization_;
    bool publish_odom_;
    bool viz_;
    
    // State variables
    std::vector<Particle> particles_;
    std::vector<double> weights_;
    Eigen::Vector3d current_pose_;
    nav_msgs::msg::Odometry last_odom_;
    sensor_msgs::msg::LaserScan last_scan_;
    nav_msgs::msg::OccupancyGrid map_;
    
    // RangeLibc variables
#ifdef USE_RANGELIBC
    std::unique_ptr<RangeLib::RaycastBase> range_method_ptr_;
    std::vector<float> laser_angles_;
    std::vector<float> downsampled_angles_;
    std::vector<float> queries_;
    std::vector<float> ranges_;
    bool rangelib_initialized_;
#endif
    
    // Flags
    bool map_initialized_;
    bool odom_initialized_;
    bool scan_initialized_;
    bool particles_initialized_;
    
    // Thread safety
    std::mutex state_mutex_;
    
    // Random number generation
    std::mt19937 gen_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
};

} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_