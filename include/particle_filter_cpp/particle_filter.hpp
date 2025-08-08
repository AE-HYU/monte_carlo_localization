// ================================================================================================
// PARTICLE FILTER HEADER - Monte Carlo Localization (MCL) Class Definition
// ================================================================================================
// Core MCL implementation with:
// - Multinomial resampling for particle selection
// - Velocity motion model with Gaussian noise
// - 4-component beam sensor model (Z_HIT + Z_SHORT + Z_MAX + Z_RAND)
// - Lookup table optimization for O(1) sensor probability queries
// - Real-time ray casting for range simulation
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_
#define PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <random>
#include <vector>

namespace particle_filter_cpp
{

// Monte Carlo Localization node with performance optimizations
class ParticleFilter : public rclcpp::Node
{
  public:
    explicit ParticleFilter(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  private:
    // --------------------------------- CORE MCL ALGORITHM ---------------------------------
    void MCL(const Eigen::Vector3d &action, const std::vector<float> &observation);
    void motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action);
    void sensor_model(const Eigen::MatrixXd &proposal_dist, const std::vector<float> &obs,
                      std::vector<double> &weights);
    Eigen::Vector3d expected_pose();  // Weighted mean pose estimation

    // --------------------------------- INITIALIZATION ---------------------------------
    void initialize_global();               // Global localization (uniform over free space)
    void initialize_particles_pose(const Eigen::Vector3d &pose);  // Local initialization
    void precompute_sensor_model();         // Build lookup table for sensor model

    // --------------------------------- ROS2 CALLBACKS ---------------------------------
    void lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg);    // LiDAR measurements
    void odomCB(const nav_msgs::msg::Odometry::SharedPtr msg);         // Odometry + trigger MCL
    void clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);  // RViz pose
    void clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr msg);             // RViz global init

    // --------------------------------- MAP MANAGEMENT ---------------------------------
    void get_omap();  // Load occupancy grid and extract free space

    // --------------------------------- OUTPUT & VISUALIZATION ---------------------------------
    void publish_tf(const Eigen::Vector3d &pose, const rclcpp::Time &stamp);  // TF + odom
    void visualize();                                                        // RViz displays
    void publish_particles(const Eigen::MatrixXd &particles_to_pub);         // Particle cloud
    void publish_scan(const std::vector<float> &angles, const std::vector<float> &ranges);  // Debug scan

    // --------------------------------- UTILITY FUNCTIONS ---------------------------------
    double quaternion_to_angle(const geometry_msgs::msg::Quaternion &q);    // Quaternion → yaw
    geometry_msgs::msg::Quaternion angle_to_quaternion(double angle);       // Yaw → quaternion
    void map_to_world(Eigen::MatrixXd &poses);                             // Coordinate transform
    Eigen::Matrix2d rotation_matrix(double angle);                         // 2D rotation matrix

    // --------------------------------- RAY CASTING ---------------------------------
    std::vector<float> calc_range_many(const Eigen::MatrixXd &queries);  // Batch processing
    float cast_ray(double x, double y, double angle);                    // Individual ray

    // --------------------------------- ALGORITHM PARAMETERS ---------------------------------
    int ANGLE_STEP;              // LiDAR downsampling factor
    int MAX_PARTICLES;           // Number of particles (N)
    int MAX_VIZ_PARTICLES;       // Visualization particle limit
    double INV_SQUASH_FACTOR;    // 1/squash_factor for weight variance reduction
    double MAX_RANGE_METERS;     // Sensor maximum range [m]
    int THETA_DISCRETIZATION;    // Angular discretization
    std::string WHICH_RM;        // Range method identifier
    int RANGELIB_VAR;           // Range library variant
    bool SHOW_FINE_TIMING;      // Performance timing flag
    bool PUBLISH_ODOM;          // Navigation odometry output
    bool DO_VIZ;                // Visualization enable

    // --------------------------------- SENSOR MODEL PARAMETERS ---------------------------------
    double Z_SHORT, Z_MAX, Z_RAND, Z_HIT, SIGMA_HIT;  // 4-component mixture weights + Gaussian σ

    // --------------------------------- MOTION MODEL PARAMETERS ---------------------------------
    double MOTION_DISPERSION_X, MOTION_DISPERSION_Y, MOTION_DISPERSION_THETA;  // Process noise σ

    // --------------------------------- PARTICLE FILTER STATE ---------------------------------
    Eigen::MatrixXd particles_;    // [N×3] particle states [x, y, θ]
    std::vector<double> weights_;  // [N×1] normalized particle weights
    Eigen::Vector3d inferred_pose_; // Final pose estimate
    Eigen::Vector3d odometry_data_; // Current motion command [Δx, Δy, Δθ]
    Eigen::Vector3d last_pose_;     // Previous pose for motion computation

    // --------------------------------- SENSOR DATA ---------------------------------
    std::vector<float> laser_angles_;        // Full LiDAR angle array
    std::vector<float> downsampled_angles_;  // Downsampled angles for efficiency
    std::vector<float> downsampled_ranges_;  // Current downsampled measurements

    // --------------------------------- MAP DATA ---------------------------------
    nav_msgs::msg::OccupancyGrid::SharedPtr map_msg_;  // Occupancy grid map
    Eigen::MatrixXi permissible_region_;               // Binary free space mask
    bool map_initialized_;     // Map loading status
    bool lidar_initialized_;   // LiDAR initialization status
    bool odom_initialized_;    // Odometry initialization status
    bool first_sensor_update_; // First sensor model call flag

    // --------------------------------- SENSOR MODEL OPTIMIZATION ---------------------------------
    Eigen::MatrixXd sensor_model_table_;  // [range_px × range_px] probability lookup table
    int MAX_RANGE_PX;          // Maximum range in pixels
    double map_resolution_;    // Map resolution [m/pixel]
    Eigen::Vector3d map_origin_; // Map origin [x, y, θ]

    // --------------------------------- PERFORMANCE CACHES ---------------------------------
    Eigen::MatrixXd local_deltas_;    // [N×3] motion transformation cache
    Eigen::MatrixXd queries_;         // [N×K×3] ray casting query matrix
    std::vector<float> ranges_;       // [N×K×1] ray casting results
    std::vector<float> tiled_angles_; // [N×K×1] angle array for all particles

    // --------------------------------- VISUALIZATION CACHE ---------------------------------
    Eigen::MatrixXd viz_queries_;  // Visualization ray queries
    std::vector<float> viz_ranges_; // Visualization ray results

    // --------------------------------- ROS2 INTERFACES ---------------------------------
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr laser_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr click_sub_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_pub_;    // Particle cloud
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;      // Pose estimate
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;             // Navigation odom
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_fake_scan_;     // Debug scan
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr rect_pub_;   // Debug shapes

    // Services and TF
    rclcpp::Client<nav_msgs::srv::GetMap>::SharedPtr map_client_;        // Map service client
    std::unique_ptr<tf2_ros::TransformBroadcaster> pub_tf_;             // TF broadcaster

    // --------------------------------- THREADING ---------------------------------
    std::mutex state_lock_;  // MCL update synchronization

    // --------------------------------- RANDOM NUMBER GENERATION ---------------------------------
    std::mt19937 rng_;                                    // Mersenne Twister RNG
    std::uniform_real_distribution<double> uniform_dist_; // U(0,1) for sampling
    std::normal_distribution<double> normal_dist_;        // N(0,1) for noise

    // --------------------------------- TIMING & STATISTICS ---------------------------------
    rclcpp::Time last_stamp_;  // Last sensor timestamp
    int iters_;                // MCL iteration counter
    double current_speed_;     // Current robot speed [m/s]

    // --------------------------------- ALGORITHM INTERNALS ---------------------------------
    std::vector<int> particle_indices_;  // Index array for resampling

    // --------------------------------- UPDATE CONTROL ---------------------------------
    void update();  // Main MCL update loop
};

} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARTICLE_FILTER_HPP_
