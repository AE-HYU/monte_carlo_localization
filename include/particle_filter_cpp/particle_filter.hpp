#ifndef PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/srv/get_map.hpp>

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <memory>
#include <mutex>

namespace particle_filter_cpp
{

// Direct port of Python ParticleFilter with C++ optimizations
class ParticleFilter : public rclcpp::Node
{
public:
    explicit ParticleFilter(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    // Core MCL algorithm - direct port from Python
    void MCL(const Eigen::Vector3d& action, const std::vector<float>& observation);
    void motion_model(Eigen::MatrixXd& proposal_dist, const Eigen::Vector3d& action);
    void sensor_model(const Eigen::MatrixXd& proposal_dist, const std::vector<float>& obs, std::vector<double>& weights);
    Eigen::Vector3d expected_pose();
    
    // Initialization
    void initialize_global();
    void initialize_particles_pose(const Eigen::Vector3d& pose);
    void precompute_sensor_model();
    
    // ROS callbacks
    void lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg);
    void clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr msg);
    
    // Map handling
    void get_omap();
    
    // Publishing
    void publish_tf(const Eigen::Vector3d& pose, const rclcpp::Time& stamp);
    void visualize();
    void publish_particles(const Eigen::MatrixXd& particles_to_pub);
    void publish_scan(const std::vector<float>& angles, const std::vector<float>& ranges);
    
    // Utility functions
    double quaternion_to_angle(const geometry_msgs::msg::Quaternion& q);
    geometry_msgs::msg::Quaternion angle_to_quaternion(double angle);
    void map_to_world(Eigen::MatrixXd& poses);
    Eigen::Matrix2d rotation_matrix(double angle);
    
    // Simple ray casting implementation
    std::vector<float> calc_range_many(const Eigen::MatrixXd& queries);
    float cast_ray(double x, double y, double angle);
    
    // Parameters (matching Python implementation exactly)
    int ANGLE_STEP;
    int MAX_PARTICLES;
    int MAX_VIZ_PARTICLES;
    double INV_SQUASH_FACTOR;
    double MAX_RANGE_METERS;
    int THETA_DISCRETIZATION;
    std::string WHICH_RM;
    int RANGELIB_VAR;
    bool SHOW_FINE_TIMING;
    bool PUBLISH_ODOM;
    bool DO_VIZ;
    
    // Sensor model constants
    double Z_SHORT, Z_MAX, Z_RAND, Z_HIT, SIGMA_HIT;
    
    // Motion model constants
    double MOTION_DISPERSION_X, MOTION_DISPERSION_Y, MOTION_DISPERSION_THETA;
    
    // State variables
    Eigen::MatrixXd particles_;  // [MAX_PARTICLES x 3] matrix
    std::vector<double> weights_;
    Eigen::Vector3d inferred_pose_;
    Eigen::Vector3d odometry_data_;
    Eigen::Vector3d last_pose_;
    
    // Laser data
    std::vector<float> laser_angles_;
    std::vector<float> downsampled_angles_;
    std::vector<float> downsampled_ranges_;
    
    // Map data
    nav_msgs::msg::OccupancyGrid::SharedPtr map_msg_;
    Eigen::MatrixXi permissible_region_;
    bool map_initialized_;
    bool lidar_initialized_;
    bool odom_initialized_;
    bool first_sensor_update_;
    
    // Sensor model lookup table
    Eigen::MatrixXd sensor_model_table_;
    int MAX_RANGE_PX;
    double map_resolution_;
    Eigen::Vector3d map_origin_;
    
    // Cached arrays for performance
    Eigen::MatrixXd local_deltas_;
    Eigen::MatrixXd queries_;
    std::vector<float> ranges_;
    std::vector<float> tiled_angles_;
    
    // Visualization queries
    Eigen::MatrixXd viz_queries_;
    std::vector<float> viz_ranges_;
    
    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_sub_;
    
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_fake_scan_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr rect_pub_;
    
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr map_client_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> pub_tf_;
    
    // Threading
    std::mutex state_lock_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
    
    // Timing and statistics
    rclcpp::Time last_stamp_;
    int iters_;
    double current_speed_;
    
    // Particle indices for resampling
    std::vector<int> particle_indices_;
    
    // Update control
    void update();
};

} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_