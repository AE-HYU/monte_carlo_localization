// ================================================================================================
// MONTE CARLO LOCALIZATION (MCL) - Main Class Definition
// ================================================================================================
// Features: Multinomial resampling, velocity motion model, beam sensor model, ray casting
// ================================================================================================

#ifndef MCL_PKG__MCL_HPP_
#define MCL_PKG__MCL_HPP_

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

#include "mcl_pkg/utils.hpp"

namespace mcl_pkg
{

class MCL : public rclcpp::Node
{
  public:
    explicit MCL(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    ~MCL();

    // ================================================================================================
    // PUBLIC MEMBERS - Accessible by external modules (parameter_manager, map_manager, etc.)
    // ================================================================================================

    // --------------------------------- CORE MCL ALGORITHM ---------------------------------
    void run_mcl(const Eigen::Vector3d &action, const std::vector<float> &observation);
    void motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action);
    void sensor_model(const Eigen::MatrixXd &proposal_dist, const std::vector<float> &obs,
                      std::vector<double> &weights);

    Eigen::Vector3d expected_pose();

    // --------------------------------- RESAMPLING ---------------------------------
    double calculate_ess();  // Calculate Effective Sample Size
    void resample();         // Execute resampling (multinomial)

    // --------------------------------- INITIALIZATION ---------------------------------
    void initParameters();  // Parameter validation with semantic checks

    // --------------------------------- ROS2 CALLBACKS ---------------------------------
    void lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg);
    void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg);
    void clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr msg);

    // Dynamic parameter reconfiguration
    rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
        const std::vector<rclcpp::Parameter> &parameters);

    // --------------------------------- TF UTILITIES ---------------------------------
    Eigen::Vector3d calculate_lidar_frame_motion(const rclcpp::Time& current_lidar_stamp);

    // --------------------------------- RAY CASTING ---------------------------------
    std::vector<float> calc_range_many(const Eigen::MatrixXd &queries);
    float cast_ray(double x, double y, double angle,
                   const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map);

    // --------------------------------- ALGORITHM PARAMETERS ---------------------------------
    int ANGLE_STEP;
    int MAX_PARTICLES;
    int MIN_PARTICLES;  // Minimum particle count for adaptive resampling
    int MAX_VIZ_PARTICLES;
    double INV_SQUASH_FACTOR;
    double MAX_RANGE_METERS;
    bool PUBLISH_ODOM;
    bool PUBLISH_MAP_ODOM_TF;
    bool DO_VIZ;
    double TIMER_FREQUENCY;
    bool USE_PARALLEL_RAYCASTING;
    int NUM_THREADS;
    double MAX_POSE_RANGE;

    // --------------------------------- SENSOR MODEL PARAMETERS ---------------------------------
    std::string SENSOR_MODEL_TYPE;  // "beam" or "likelihood_field"
    double Z_SHORT, Z_MAX, Z_RAND, Z_HIT, SIGMA_HIT;
    double LIKELIHOOD_SIGMA;  // Sigma for likelihood field model (meters)

    // --------------------------------- MOTION MODEL PARAMETERS ---------------------------------
    std::string MOTION_MODEL_TYPE;  // "simple" or "odometry" (RTR)

    // Simple motion model (fixed Gaussian noise)
    double MOTION_DISPERSION_X, MOTION_DISPERSION_Y, MOTION_DISPERSION_THETA;

    // Odometry motion model (RTR with alpha parameters)
    double ALPHA1, ALPHA2, ALPHA3, ALPHA4;
    double SMALL_TRANS_THRESHOLD, SMALL_ROT_THRESHOLD;  // Small motion thresholds

    // --------------------------------- ROBOT GEOMETRY PARAMETERS ---------------------------------
    double WHEELBASE;

    // --------------------------------- RESAMPLING PARAMETERS ---------------------------------
    bool USE_ADAPTIVE_RESAMPLING;  // Enable/disable adaptive resampling
    double ESS_THRESHOLD;           // ESS threshold for resampling (0.0-1.0)
    std::string RESAMPLING_TYPE;    // Resampling method: "multinomial" or "low_variance"

    // --------------------------------- TF FRAME NAMES ---------------------------------
    std::string MAP_FRAME;
    std::string ODOM_FRAME;
    std::string BASE_FRAME;
    std::string LASER_FRAME;

    // --------------------------------- PARTICLE FILTER STATE ---------------------------------
    Eigen::MatrixXd particles_;
    std::vector<double> weights_;

    // --------------------------------- SENSOR DATA ---------------------------------
    std::vector<float> downsampled_angles_;
    std::vector<float> downsampled_ranges_;
    rclcpp::Time last_lidar_time_;
    bool has_new_lidar_data_;
    double mcl_processing_time_;  // Store actual MCL processing time for timestamp compensation

    // --------------------------------- MAP DATA ---------------------------------
    nav_msgs::msg::OccupancyGrid::SharedPtr map_msg_;
    bool map_initialized_;
    bool lidar_initialized_;
    bool odom_initialized_;
    bool first_sensor_update_;

    // --------------------------------- SENSOR MODEL OPTIMIZATION ---------------------------------
    Eigen::MatrixXd sensor_model_table_;
    int MAX_RANGE_PX_;

    // Distance field for likelihood field sensor model
    std::vector<float> distance_field_;  // Precomputed distance to nearest obstacle (meters)
    int distance_field_width_;           // Width of distance field (pixels)
    int distance_field_height_;          // Height of distance field (pixels)
    double distance_field_resolution_;   // Resolution of distance field (meters/pixel)
    bool distance_field_initialized_;    // Whether distance field is ready

    // Gaussian likelihood lookup table for likelihood field model
    std::vector<double> likelihood_lookup_table_;  // Precomputed Gaussian values
    double likelihood_table_resolution_;           // Distance resolution for lookup (meters)
    int likelihood_table_size_;                    // Size of lookup table

    // Precomputed cos/sin for likelihood field endpoint calculation
    std::vector<double> cos_table_;  // cos(downsampled_angles_[i])
    std::vector<double> sin_table_;  // sin(downsampled_angles_[i])

    // --------------------------------- PERFORMANCE CACHES ---------------------------------
    Eigen::MatrixXd local_deltas_;
    Eigen::MatrixXd queries_;
    Eigen::MatrixXd proposal_distribution_;  // Pre-allocated for MCL resampling
    std::vector<float> ranges_;
    std::vector<float> tiled_angles_;
    std::vector<float> obs_px_;              // Pre-allocated for sensor model
    std::vector<float> ranges_px_;           // Pre-allocated for sensor model

    // --------------------------------- ROS2 INTERFACES ---------------------------------
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_sub_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

    // Services and TF
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr map_client_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> pub_tf_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Timers
    rclcpp::TimerBase::SharedPtr update_timer_;
    rclcpp::TimerBase::SharedPtr map_timer_;
    rclcpp::TimerBase::SharedPtr map_loader_timer_;  // Async map loading timer

    // --------------------------------- THREADING ---------------------------------
    std::mutex state_lock_;
    std::mutex lidar_lock_;
    std::mutex odom_lock_;
    std::mutex map_lock_;
    std::mutex rng_lock_;

    // --------------------------------- RANDOM NUMBER GENERATION ---------------------------------
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;

    // --------------------------------- VELOCITY TRACKING ---------------------------------
    double current_velocity_;          // Current linear velocity (m/s)
    double current_angular_vel_;       // Current angular velocity (rad/s)

    // --------------------------------- LASER-BASELINK OFFSET ---------------------------------
    double laser_offset_x_;            // Laser offset from base_link (forward, m)
    double laser_offset_y_;            // Laser offset from base_link (lateral, m)

    // --------------------------------- PERFORMANCE PROFILING ---------------------------------
    utils::performance::TimingStats timing_stats_;

    // --------------------------------- UPDATE CONTROL ---------------------------------
    void timer_update(); // Main MCL update function - includes odometry publishing
    void publish_map_periodically();

    // --------------------------------- CALLBACK GROUPS FOR THREADING ---------------------------------
    rclcpp::CallbackGroup::SharedPtr sensor_group_;   // LiDAR, Odom 등 센서 콜백
    rclcpp::CallbackGroup::SharedPtr compute_group_;  // timer_update 등 무거운 콜백

    // --------------------------------- DYNAMIC RECONFIGURATION ---------------------------------
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

};

} // namespace mcl_pkg

#endif // MCL_PKG__MCL_HPP_
