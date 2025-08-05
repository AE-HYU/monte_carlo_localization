#ifndef PARTICLE_FILTER_CPP__PARTICLE_FILTER_SIMPLE_HPP_
#define PARTICLE_FILTER_CPP__PARTICLE_FILTER_SIMPLE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <angles/angles.h>

#include <vector>
#include <random>
#include <memory>
#include <chrono>

namespace particle_filter_cpp {

struct Particle {
    double x, y, theta;
};

class ParticleFilter : public rclcpp::Node {
public:
    ParticleFilter();

private:
    // ROS2 interface
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particles_pub_;
    
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Parameters
    void declare_parameters();
    void get_parameters();
    
    std::string scan_topic_;
    std::string odom_topic_;
    int angle_step_;
    int max_particles_;
    double squash_factor_;
    double max_range_;
    double z_hit_, z_short_, z_max_, z_rand_;
    double sigma_hit_;
    double motion_dispersion_x_, motion_dispersion_y_, motion_dispersion_theta_;
    
    // Callbacks
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void scan_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    
    // Core algorithm
    void initialize_particles_global();
    void motion_model(double dx, double dy, double dtheta);
    void sensor_model(const std::vector<float>& ranges, double angle_min, double angle_increment);
    double cast_ray(double x, double y, double angle);
    void resample();
    
    // Publishing
    void publish_pose();
    void publish_particles();
    
    // State
    std::vector<Particle> particles_;
    std::vector<double> weights_;
    nav_msgs::msg::OccupancyGrid map_;
    nav_msgs::msg::Odometry last_odom_;
    
    bool initialized_;
    bool map_initialized_;
    bool last_odom_initialized_;
    bool first_scan_;
    
    // Map processing
    std::vector<std::pair<int, int>> permissible_region_;
    
    // Random number generation
    std::mt19937 rng_;
};

} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARTICLE_FILTER_SIMPLE_HPP_