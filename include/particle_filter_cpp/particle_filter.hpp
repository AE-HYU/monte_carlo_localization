#ifndef PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/srv/get_map.hpp>

#include "particle_filter_cpp/modules/mcl_algorithm.hpp"
#include <memory>

namespace particle_filter_cpp
{

class ParticleFilter : public rclcpp::Node
{
public:
    explicit ParticleFilter(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    // ROS callbacks
    void laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    
    // Map service
    void get_map();
    
    // ROS message conversions
    modules::LaserScanData convert_laser_scan(const sensor_msgs::msg::LaserScan& msg);
    modules::OdometryData convert_odom(const nav_msgs::msg::Odometry& msg);
    modules::MapInfo convert_map(const nav_msgs::msg::OccupancyGrid& map);
    
    // Publishing
    void publish_results();
    void publish_particles();
    void publish_pose();
    void publish_transform();
    
    // ROS publishers and subscribers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_sub_;
    
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr map_client_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Core MCL algorithm (ROS-independent)
    std::unique_ptr<modules::MCLAlgorithm> mcl_;
    
    // ROS parameters
    bool publish_odom_;
    bool viz_;
    
    // State tracking
    bool map_received_;
    modules::OdometryData last_odom_;
    bool first_odom_;
};

} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_