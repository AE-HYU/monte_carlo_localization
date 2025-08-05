#include "particle_filter_cpp/particle_filter.hpp"
#include "particle_filter_cpp/utils.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>

namespace particle_filter_cpp
{

ParticleFilter::ParticleFilter(const rclcpp::NodeOptions& options)
    : Node("particle_filter", options), map_received_(false), first_odom_(true)
{
    // Declare parameters
    this->declare_parameter("num_particles", 4000);
    this->declare_parameter("max_range", 10.0);
    this->declare_parameter("motion_dispersion_x", 0.05);
    this->declare_parameter("motion_dispersion_y", 0.025);
    this->declare_parameter("motion_dispersion_theta", 0.25);
    this->declare_parameter("z_hit", 0.75);
    this->declare_parameter("z_short", 0.01);
    this->declare_parameter("z_max", 0.07);
    this->declare_parameter("z_rand", 0.12);
    this->declare_parameter("sigma_hit", 8.0);
    this->declare_parameter("scan_topic", "/scan");
    this->declare_parameter("odom_topic", "/odom");
    this->declare_parameter("range_method", "cddt");
    this->declare_parameter("publish_odom", true);
    this->declare_parameter("viz", true);

    // Get parameters and setup MCL
    modules::MCLParams mcl_params;
    mcl_params.num_particles = this->get_parameter("num_particles").as_int();
    mcl_params.motion_params.dispersion_x = this->get_parameter("motion_dispersion_x").as_double();
    mcl_params.motion_params.dispersion_y = this->get_parameter("motion_dispersion_y").as_double();
    mcl_params.motion_params.dispersion_theta = this->get_parameter("motion_dispersion_theta").as_double();
    mcl_params.sensor_params.z_hit = this->get_parameter("z_hit").as_double();
    mcl_params.sensor_params.z_short = this->get_parameter("z_short").as_double();
    mcl_params.sensor_params.z_max = this->get_parameter("z_max").as_double();
    mcl_params.sensor_params.z_rand = this->get_parameter("z_rand").as_double();
    mcl_params.sensor_params.sigma_hit = this->get_parameter("sigma_hit").as_double();
    mcl_params.sensor_params.max_range = this->get_parameter("max_range").as_double();
    
    // Resampling parameters
    mcl_params.resampling_params.method = modules::ResamplingMethod::SYSTEMATIC;
    mcl_params.resampling_params.ess_threshold_ratio = 0.5;
    mcl_params.resampling_params.adaptive = true;
    
    // Pose estimation parameters
    mcl_params.pose_params.method = modules::PoseEstimationMethod::WEIGHTED_AVERAGE;
    
    // Algorithm parameters
    mcl_params.update_min_distance = 0.1;
    mcl_params.update_min_angle = 0.1;
    mcl_params.enable_timing = true;
    
    publish_odom_ = this->get_parameter("publish_odom").as_bool();
    viz_ = this->get_parameter("viz").as_bool();

    // Initialize MCL algorithm
    mcl_ = std::make_unique<modules::MCLAlgorithm>(mcl_params);
    
    // Initialize TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    
    // Initialize publishers
    if (viz_) {
        particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 1);
    }
    
    if (publish_odom_) {
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", 1);
    }
    
    // Initialize subscribers
    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        this->get_parameter("scan_topic").as_string(), 1,
        std::bind(&ParticleFilter::laser_callback, this, std::placeholders::_1));
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        this->get_parameter("odom_topic").as_string(), 1,
        std::bind(&ParticleFilter::odom_callback, this, std::placeholders::_1));
    
    initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1,
        std::bind(&ParticleFilter::initial_pose_callback, this, std::placeholders::_1));
    
    // Initialize map service client
    map_client_ = this->create_client<nav_msgs::srv::GetMap>("/map_server/map");
    
    // Get the map
    get_map();
    
    RCLCPP_INFO(this->get_logger(), "Particle Filter ROS node initialized");
}

void ParticleFilter::get_map()
{
    RCLCPP_INFO(this->get_logger(), "Requesting map from map server...");
    
    while (!map_client_->wait_for_service(std::chrono::seconds(1))) {
        if (!rclcpp::ok()) return;
        RCLCPP_INFO(this->get_logger(), "Map service not available, waiting...");
    }

    auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();
    auto future = map_client_->async_send_request(request);

    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS) {
        
        auto map_info = convert_map(future.get()->map);
        mcl_->initialize_global(map_info);
        map_received_ = true;
        
        RCLCPP_INFO(this->get_logger(), "Map received and MCL initialized");
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to get map from map server");
    }
}

void ParticleFilter::laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    if (!map_received_) return;
    
    auto scan_data = convert_laser_scan(*msg);
    
    // Update MCL with latest data
    if (mcl_->update(last_odom_, scan_data)) {
        publish_results();
    }
}

void ParticleFilter::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    last_odom_ = convert_odom(*msg);
    
    if (first_odom_) {
        first_odom_ = false;
        RCLCPP_INFO(this->get_logger(), "Received first odometry message");
    }
}

void ParticleFilter::initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    if (!map_received_) return;
    
    RCLCPP_INFO(this->get_logger(), "Received initial pose estimate");
    
    Eigen::Vector3d pose(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        utils::quaternion_to_yaw(msg->pose.pose.orientation)
    );
    
    Eigen::Vector3d std_dev(0.5, 0.5, 0.4); // Standard deviations
    mcl_->initialize_pose(pose, std_dev);
}

modules::LaserScanData ParticleFilter::convert_laser_scan(const sensor_msgs::msg::LaserScan& msg)
{
    modules::LaserScanData scan_data;
    scan_data.ranges = msg.ranges;
    scan_data.max_range = msg.range_max;
    scan_data.angle_min = msg.angle_min;
    scan_data.angle_max = msg.angle_max;
    scan_data.angle_increment = msg.angle_increment;
    scan_data.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
    
    // Generate angles
    scan_data.angles.resize(msg.ranges.size());
    for (size_t i = 0; i < msg.ranges.size(); ++i) {
        scan_data.angles[i] = msg.angle_min + i * msg.angle_increment;
    }
    
    return scan_data;
}

modules::OdometryData ParticleFilter::convert_odom(const nav_msgs::msg::Odometry& msg)
{
    modules::OdometryData odom_data;
    odom_data.pose = Eigen::Vector3d(
        msg.pose.pose.position.x,
        msg.pose.pose.position.y,
        utils::quaternion_to_yaw(msg.pose.pose.orientation)
    );
    odom_data.velocity = Eigen::Vector3d(
        msg.twist.twist.linear.x,
        msg.twist.twist.linear.y,
        msg.twist.twist.angular.z
    );
    odom_data.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
    
    return odom_data;
}

modules::MapInfo ParticleFilter::convert_map(const nav_msgs::msg::OccupancyGrid& map)
{
    modules::MapInfo map_info;
    map_info.width = map.info.width;
    map_info.height = map.info.height;
    map_info.resolution = map.info.resolution;
    map_info.origin = Eigen::Vector3d(
        map.info.origin.position.x,
        map.info.origin.position.y,
        utils::quaternion_to_yaw(map.info.origin.orientation)
    );
    map_info.data = map.data;
    
    return map_info;
}

void ParticleFilter::publish_results()
{
    publish_transform();
    
    if (viz_) {
        publish_particles();
        publish_pose();
    }
}

void ParticleFilter::publish_particles()
{
    if (!particle_pub_ || particle_pub_->get_subscription_count() == 0) return;
    
    auto particles = mcl_->get_particles();
    auto pose_array = utils::particles_to_pose_array(particles);
    pose_array.header.stamp = this->get_clock()->now();
    pose_array.header.frame_id = "map";
    
    particle_pub_->publish(pose_array);
}

void ParticleFilter::publish_pose()
{
    if (!pose_pub_ || pose_pub_->get_subscription_count() == 0) return;
    
    auto pose = mcl_->get_estimated_pose();
    
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->get_clock()->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = pose[0];
    pose_msg.pose.position.y = pose[1];
    pose_msg.pose.orientation = utils::yaw_to_quaternion(pose[2]);
    
    pose_pub_->publish(pose_msg);
}

void ParticleFilter::publish_transform()
{
    auto pose = mcl_->get_estimated_pose();
    
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = this->get_clock()->now();
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    transform.transform.translation.x = pose[0];
    transform.transform.translation.y = pose[1];
    transform.transform.translation.z = 0.0;
    transform.transform.rotation = utils::yaw_to_quaternion(pose[2]);
    
    tf_broadcaster_->sendTransform(transform);
    
    // Publish odometry if requested
    if (publish_odom_ && odom_pub_) {
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header = transform.header;
        odom_msg.child_frame_id = "base_link";
        odom_msg.pose.pose.position.x = pose[0];
        odom_msg.pose.pose.position.y = pose[1];
        odom_msg.pose.pose.orientation = utils::yaw_to_quaternion(pose[2]);
        
        // Add covariance from MCL statistics
        auto stats = mcl_->get_statistics();
        auto cov = stats.covariance;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                odom_msg.pose.covariance[i*6 + j] = cov(i, j);
            }
        }
        
        odom_pub_->publish(odom_msg);
    }
}

} // namespace particle_filter_cpp

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(particle_filter_cpp::ParticleFilter)