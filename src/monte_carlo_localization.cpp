// ================================================================================================
// MONTE CARLO LOCALIZATION - Core MCL Algorithm Implementation
// ================================================================================================
// Modular implementation with external modules for parameter, map, initialization, visualization
// ================================================================================================

#include "mcl_pkg/mcl.hpp"
#include "mcl_pkg/parameter_manager.hpp"
#include "mcl_pkg/map_manager.hpp"
#include "mcl_pkg/initialization.hpp"
#include "mcl_pkg/visualization.hpp"
#include "mcl_pkg/utils.hpp"
#include "mcl_pkg/sensor_model/beam_sensor_model.hpp"
#include "mcl_pkg/sensor_model/likelihood_field_sensor_model.hpp"
#include "mcl_pkg/motion_model/simple_motion_model.hpp"
#include "mcl_pkg/motion_model/odometry_motion_model.hpp"
#include "mcl_pkg/resampling_model/multinomial_resampling.hpp"
#include "mcl_pkg/resampling_model/low_variance_resampling.hpp"

#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <omp.h>

namespace mcl_pkg
{

// ================================================================================================
// CONSTRUCTOR & DESTRUCTOR
// ================================================================================================

MCL::MCL(const rclcpp::NodeOptions &options)
    : Node("particle_filter", options), rng_(std::random_device{}()), normal_dist_(0.0, 1.0)
{
    // Initialize and validate parameters using parameter_manager module
    parameter_manager::initParameters(this);

    // Initialize state
    map_initialized_ = false;
    lidar_initialized_ = false;
    odom_initialized_ = false;
    first_sensor_update_ = true;
    has_new_lidar_data_ = false;
    mcl_processing_time_ = 0.0;
    current_velocity_ = 0.0;
    current_angular_vel_ = 0.0;

    // Initialize MCL-based velocity estimation
    estimated_vx_ = 0.0;
    estimated_vy_ = 0.0;
    estimated_vyaw_ = 0.0;
    last_estimated_pose_ = Eigen::Vector3d::Zero();
    velocity_initialized_ = false;
    velocity_filter_alpha_ = 0.3;  // Will be overridden by parameter_manager (0 = max filtering, 1 = no filtering)

    // Initialize distance field (likelihood field model)
    distance_field_initialized_ = false;
    distance_field_width_ = 0;
    distance_field_height_ = 0;
    distance_field_resolution_ = 0.0;

    // Initialize likelihood lookup table
    likelihood_table_resolution_ = 0.01;  // 1cm resolution
    likelihood_table_size_ = 0;

    // Initialize laser offset with default (will be loaded from TF)
    laser_offset_x_ = 0.28;
    laser_offset_y_ = 0.0;

    // Initialize particles and weights
    particles_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);
    weights_.resize(MAX_PARTICLES, 1.0 / MAX_PARTICLES);
    proposal_distribution_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);
    local_deltas_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);

    // Setup OpenMP threading
    if (USE_PARALLEL_RAYCASTING) {
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (NUM_THREADS == 0) {
            // Reserve 1 thread for executor and 1 for system
            // MCL worker is detached, so doesn't conflict with executor
            NUM_THREADS = std::max(2, hw - 2);
        } else if (NUM_THREADS > 0) {
            NUM_THREADS = std::min(NUM_THREADS, hw - 1);
        }
        omp_set_num_threads(NUM_THREADS);
    }

    // Create callback groups for threading
    sensor_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    compute_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    auto sensor_sub_options = rclcpp::SubscriptionOptions();
    sensor_sub_options.callback_group = sensor_group_;

    // ROS2 Publishers
    particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 10);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 10);
    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", 10);
    map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(10).transient_local());
    quality_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/pf/viz/quality", 10);

    // ROS2 Subscribers
    std::string scan_topic = this->get_parameter("scan_topic").as_string();
    std::string odom_topic = this->get_parameter("odom_topic").as_string();

    // LiDAR: Use SensorDataQoS (best_effort + volatile) with buffer size 1
    rclcpp::SensorDataQoS lidar_qos;
    lidar_qos.keep_last(1);
    rclcpp::SubscriptionOptions lidar_opts;
    lidar_opts.callback_group = sensor_group_;

    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic, lidar_qos,
        std::bind(&MCL::lidarCB, this, std::placeholders::_1),
        lidar_opts);

    // Odom: Regular QoS with some buffering
    rclcpp::QoS odom_qos(10);
    rclcpp::SubscriptionOptions odom_opts;
    odom_opts.callback_group = sensor_group_;

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, odom_qos,
        std::bind(&MCL::odomCB, this, std::placeholders::_1),
        odom_opts);

    // Initialization callbacks: Use sensor_group (Reentrant) for immediate response
    // The callbacks use state_lock internally, so they're safe to run concurrently with MCL
    rclcpp::SubscriptionOptions init_opts;
    init_opts.callback_group = sensor_group_;

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", rclcpp::QoS(1),
        std::bind(&MCL::clicked_pose, this, std::placeholders::_1),
        init_opts);

    click_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/clicked_point", rclcpp::QoS(1),
        std::bind(&MCL::clicked_point, this, std::placeholders::_1),
        init_opts);

    // Map client and TF
    map_client_ = this->create_client<nav_msgs::srv::GetMap>("/map_server/map");
    pub_tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Try to get laser-baselink offset from static TF
    try {
        rclcpp::sleep_for(std::chrono::milliseconds(100));

        auto static_tf = tf_buffer_->lookupTransform(
            BASE_FRAME, LASER_FRAME, tf2::TimePointZero, tf2::durationFromSec(1.0));

        laser_offset_x_ = static_tf.transform.translation.x;
        laser_offset_y_ = static_tf.transform.translation.y;

        RCLCPP_INFO(this->get_logger(),
            "Loaded laser offset from static TF: [%.3f, %.3f]m",
            laser_offset_x_, laser_offset_y_);
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(),
            "Could not get static TF %s->%s: %s. Using default offset [%.3f, %.3f]m",
            BASE_FRAME.c_str(), LASER_FRAME.c_str(), ex.what(),
            laser_offset_x_, laser_offset_y_);
    }

    // Timers (timer_update removed - now lidar-driven)
    map_loader_timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        [this]() { map_manager::try_load_map(this); });

    map_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(200),
        std::bind(&MCL::publish_map_periodically, this),
        compute_group_);

    // Register dynamic parameter callback
    param_callback_handle_ = this->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter> &parameters) {
            return parameter_manager::dynamicParametersCallback(this, parameters);
        });

    RCLCPP_INFO(this->get_logger(), "Particle filter initialized - LiDAR-driven (async), %s threading (%d threads)",
        USE_PARALLEL_RAYCASTING ? "parallel" : "sequential",
        USE_PARALLEL_RAYCASTING ? NUM_THREADS : 1);
    RCLCPP_INFO(this->get_logger(), "Motion model: %s | Sensor model: %s | Resampling: %s%s",
        MOTION_MODEL_TYPE.c_str(), SENSOR_MODEL_TYPE.c_str(), RESAMPLING_TYPE.c_str(),
        USE_ADAPTIVE_RESAMPLING ? " (adaptive)" : "");
    RCLCPP_INFO(this->get_logger(), "MCL worker: Max consecutive runs = %d (prevents infinite loop)",
        MAX_CONSECUTIVE_MCL_RUNS);
    RCLCPP_INFO(this->get_logger(), "Async map loading started - node ready, waiting for map server...");
    RCLCPP_INFO(this->get_logger(), "Dynamic parameter reconfiguration enabled");
}

MCL::~MCL()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down particle filter");

    // Stop MCL worker if running
    mcl_running_ = false;

    // Cancel all timers explicitly for clean shutdown
    if (map_loader_timer_) map_loader_timer_->cancel();
    if (map_timer_) map_timer_->cancel();

    // Other resources (smart pointers, Eigen matrices) are cleaned up automatically via RAII
}

// ================================================================================================
// ROS2 CALLBACKS
// ================================================================================================

void MCL::lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    // Downsampling and pending data update (single lock for atomicity)
    {
        std::lock_guard<std::mutex> lock(lidar_lock_);

        if (!lidar_initialized_) {
            int num_ranges = msg->ranges.size();
            int downsampled_size = num_ranges / ANGLE_STEP;
            downsampled_angles_.resize(downsampled_size);
            downsampled_ranges_.resize(downsampled_size);

            for (int i = 0; i < downsampled_size; ++i) {
                downsampled_angles_[i] = msg->angle_min + (i * ANGLE_STEP) * msg->angle_increment;
            }

            // Precompute cos/sin for likelihood field model
            if (SENSOR_MODEL_TYPE == "likelihood_field") {
                cos_table_.resize(downsampled_size);
                sin_table_.resize(downsampled_size);
                for (int i = 0; i < downsampled_size; ++i) {
                    cos_table_[i] = std::cos(downsampled_angles_[i]);
                    sin_table_[i] = std::sin(downsampled_angles_[i]);
                }
                RCLCPP_INFO(this->get_logger(), "LiDAR initialized - %zu angles, cos/sin tables precomputed",
                            downsampled_angles_.size());
            } else {
                RCLCPP_INFO(this->get_logger(), "LiDAR initialized - %zu angles", downsampled_angles_.size());
            }

            lidar_initialized_ = true;
        }

        // Downsample ranges
        int downsampled_size = downsampled_angles_.size();
        for (int i = 0; i < downsampled_size; ++i) {
            int idx = i * ANGLE_STEP;
            downsampled_ranges_[i] = (idx < static_cast<int>(msg->ranges.size()))
                                         ? msg->ranges[idx]
                                         : msg->range_max;
        }

        last_lidar_time_ = msg->header.stamp;

        // Update pending MCL task (atomically with downsampling)
        pending_mcl_data_ = MCLTaskData{
            .observation = downsampled_ranges_,
            .timestamp = msg->header.stamp
        };
    }

    // Check if MCL worker is already running
    bool was_running = mcl_running_.exchange(true);
    if (was_running) {
        // MCL already running, pending data updated above
        RCLCPP_DEBUG(this->get_logger(),
            "[LiDAR] MCL busy, updated pending data");
        return;
    }

    // Start MCL worker thread
    RCLCPP_DEBUG(this->get_logger(),
        "[LiDAR] Triggering MCL worker");

    std::thread([this]() {
        execute_mcl_worker();
    }).detach();
}

void MCL::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Store latest odom data for MCL worker and visualization (needs lock)
    {
        std::lock_guard<std::mutex> lock(odom_lock_);
        current_velocity_ = msg->twist.twist.linear.x;
        current_angular_vel_ = msg->twist.twist.angular.z * M_PI / 180.0;  // Convert deg/s to rad/s
        latest_odom_timestamp_ = msg->header.stamp;
        latest_odom_pose_[0] = msg->pose.pose.position.x;
        latest_odom_pose_[1] = msg->pose.pose.position.y;
        latest_odom_pose_[2] = utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation);

        if (!odom_initialized_) {
            odom_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Odometry initialized");
        }
    }

    // Publish pose immediately with odom data (no extrapolation needed)
    publish_pose(msg);
}

void MCL::clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    if (!map_initialized_)
        return;

    // Extract pose from RViz message
    Eigen::Vector3d base_pose(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation));

    // Convert base_link to laser frame by adding offset rotated by heading
    double cos_theta = std::cos(base_pose[2]);
    double sin_theta = std::sin(base_pose[2]);
    Eigen::Vector3d laser_pose(
        base_pose[0] + (laser_offset_x_ * cos_theta - laser_offset_y_ * sin_theta),
        base_pose[1] + (laser_offset_x_ * sin_theta + laser_offset_y_ * cos_theta),
        base_pose[2]);

    RCLCPP_INFO(this->get_logger(), "Pose initialized from RViz at base_link [%.3f, %.3f, %.3f] -> laser [%.3f, %.3f, %.3f]",
        base_pose[0], base_pose[1], base_pose[2],
        laser_pose[0], laser_pose[1], laser_pose[2]);

    // Initialize particles around laser pose
    initialization::initialize_particles_pose(this, laser_pose);
}

void MCL::clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr msg)
{
    if (!map_initialized_)
        return;

    Eigen::Vector3d pose(msg->point.x, msg->point.y, 0.0);
    initialization::initialize_particles_pose(this, pose);
}

// ================================================================================================
// MAIN MCL WORKER (Lidar-driven, async)
// ================================================================================================

void MCL::execute_mcl_worker()
{
    int consecutive_runs = 0;

    while (consecutive_runs < MAX_CONSECUTIVE_MCL_RUNS) {
        auto worker_start = std::chrono::high_resolution_clock::now();

        // 1. Check initialization
        if (!lidar_initialized_ || !odom_initialized_ || !map_initialized_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "[MCL Worker] Not ready - LiDAR: %s, Odom: %s, Map: %s",
                lidar_initialized_ ? "OK" : "NO",
                odom_initialized_ ? "OK" : "NO",
                map_initialized_ ? "OK" : "NO");
            mcl_running_ = false;
            return;
        }

        // 2. Get pending MCL task (guaranteed to exist when worker is triggered)
        MCLTaskData task;
        {
            std::lock_guard<std::mutex> lock(lidar_lock_);
            task = *pending_mcl_data_;
            pending_mcl_data_.reset();  // Clear after taking
        }

        rclcpp::Time lidar_timestamp = task.timestamp;
        std::vector<float> observation = task.observation;  // Copy, not reference

        // 3. Get odom->base at lidar timestamp (exact time or latest available)
        auto motion_start = std::chrono::high_resolution_clock::now();

        Eigen::Vector3d odom_to_base_vec;
        timing_stats_.tf_total_count++;

        try {
            // Try exact time first
            auto odom_to_base_tf = tf_buffer_->lookupTransform(
                ODOM_FRAME, BASE_FRAME, lidar_timestamp, rclcpp::Duration(0, 0));

            odom_to_base_vec = Eigen::Vector3d(
                odom_to_base_tf.transform.translation.x,
                odom_to_base_tf.transform.translation.y,
                utils::geometry::quaternion_to_yaw(odom_to_base_tf.transform.rotation)
            );

            timing_stats_.tf_exact_time_count++;
            RCLCPP_DEBUG(this->get_logger(),
                "[MCL Worker] Got exact time odom->base for lidar timestamp");
        } catch (tf2::TransformException &) {
            // Exact time failed - use latest available transform
            try {
                auto odom_to_base_tf = tf_buffer_->lookupTransform(
                    ODOM_FRAME, BASE_FRAME, tf2::TimePointZero);

                odom_to_base_vec = Eigen::Vector3d(
                    odom_to_base_tf.transform.translation.x,
                    odom_to_base_tf.transform.translation.y,
                    utils::geometry::quaternion_to_yaw(odom_to_base_tf.transform.rotation)
                );

                rclcpp::Time tf_time(odom_to_base_tf.header.stamp);
                double time_diff = (lidar_timestamp.nanoseconds() - tf_time.nanoseconds()) / 1e9;

                timing_stats_.tf_fallback_count++;
                RCLCPP_DEBUG(this->get_logger(),
                    "[MCL Worker] Using latest odom->base (time_diff=%.4fs)", time_diff);

            } catch (tf2::TransformException &ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "[MCL Worker] Could not get odom->base: %s - skipping MCL update", ex.what());
                mcl_running_ = false;
                return;
            }
        }

        // 4. Calculate motion using the odom->base we just obtained
        Eigen::Vector3d action = calculate_lidar_frame_motion(lidar_timestamp, odom_to_base_vec);
        auto motion_end = std::chrono::high_resolution_clock::now();
        double motion_calc_time_ms = std::chrono::duration<double, std::milli>(motion_end - motion_start).count();

        // 5. Execute MCL (heavy computation) and get quality metrics
        auto mcl_start = std::chrono::high_resolution_clock::now();
        MCLQualityMetrics metrics;
        {
            std::lock_guard<std::mutex> lock(state_lock_);
            metrics = run_mcl(action, observation);
        }
        auto mcl_end = std::chrono::high_resolution_clock::now();
        mcl_processing_time_ = std::chrono::duration<double, std::milli>(mcl_end - mcl_start).count();

        // 6. Extract quality metrics (computed before resampling, so they're meaningful)
        Eigen::Vector3d current_pose_laser = metrics.mean_pose;
        double max_weight = metrics.max_weight;
        Eigen::Matrix3d covariance = metrics.covariance;
        double particle_spread = metrics.particle_spread;

        // 7. Convert laser frame to base_link
        Eigen::Vector3d current_pose_base = utils::transforms::apply_laser_to_base_offset(
            current_pose_laser, laser_offset_x_, laser_offset_y_);

        // 7.5. Calculate MCL-based velocity estimation (actual ground velocity with slip considered)
        if (velocity_initialized_) {
            // Calculate time difference
            double dt = (lidar_timestamp - last_velocity_update_time_).seconds();

            if (dt > 0.001 && dt < 0.5) {  // Sanity check: 1ms to 500ms
                // Calculate pose change in map frame
                double dx_map = current_pose_base[0] - last_estimated_pose_[0];
                double dy_map = current_pose_base[1] - last_estimated_pose_[1];
                double dtheta = utils::geometry::normalize_angle(current_pose_base[2] - last_estimated_pose_[2]);

                // Transform velocity from map frame to robot frame
                // Robot frame velocity = R^T * map_frame_velocity
                double cos_theta = std::cos(last_estimated_pose_[2]);
                double sin_theta = std::sin(last_estimated_pose_[2]);

                double vx_raw = (dx_map * cos_theta + dy_map * sin_theta) / dt;
                double vy_raw = (-dx_map * sin_theta + dy_map * cos_theta) / dt;
                double vyaw_raw = dtheta / dt;

                // Apply low-pass filter to smooth velocity (reduces MCL noise)
                // filtered = alpha * new + (1 - alpha) * old
                estimated_vx_ = velocity_filter_alpha_ * vx_raw + (1.0 - velocity_filter_alpha_) * estimated_vx_;
                estimated_vy_ = velocity_filter_alpha_ * vy_raw + (1.0 - velocity_filter_alpha_) * estimated_vy_;
                estimated_vyaw_ = velocity_filter_alpha_ * vyaw_raw + (1.0 - velocity_filter_alpha_) * estimated_vyaw_;

                RCLCPP_DEBUG(this->get_logger(),
                    "[Velocity Est] vx=%.3f m/s (odom: %.3f), vy=%.3f m/s, vyaw=%.3f rad/s (dt=%.3fs)",
                    estimated_vx_, current_velocity_, estimated_vy_, estimated_vyaw_, dt);
            }
        } else {
            // First iteration - initialize
            velocity_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "MCL-based velocity estimation initialized");
        }

        // Update last pose and time for next iteration
        last_estimated_pose_ = current_pose_base;
        last_velocity_update_time_ = lidar_timestamp;

        // 8. Compute map->odom using the SAME odom->base from step 3 (for consistency)
        // Reuse the odom_to_base_vec that was used for motion calculation

        // Create transforms
        tf2::Transform map_to_base_tf;
        map_to_base_tf.setOrigin(tf2::Vector3(current_pose_base[0], current_pose_base[1], 0.0));
        tf2::Quaternion map_to_base_quat;
        map_to_base_quat.setRPY(0, 0, current_pose_base[2]);
        map_to_base_tf.setRotation(map_to_base_quat);

        tf2::Transform odom_to_base_tf;
        odom_to_base_tf.setOrigin(tf2::Vector3(
            odom_to_base_vec[0], odom_to_base_vec[1], 0.0));
        tf2::Quaternion odom_to_base_quat;
        odom_to_base_quat.setRPY(0, 0, odom_to_base_vec[2]);
        odom_to_base_tf.setRotation(odom_to_base_quat);

        // Compute map->odom = map->base × (odom->base)^(-1)
        tf2::Transform map_to_odom_tf = map_to_base_tf * odom_to_base_tf.inverse();

        Eigen::Vector3d map_to_odom_result(
            map_to_odom_tf.getOrigin().x(),
            map_to_odom_tf.getOrigin().y(),
            tf2::getYaw(map_to_odom_tf.getRotation())
        );

        RCLCPP_DEBUG(this->get_logger(),
            "[MCL Worker] Computed map->odom using same odom->base as motion (consistent)");

        // 9. Store latest map->odom and quality metrics for publishing
        // Compute human-readable ray-match percentage here and store it into latest_max_weight_
        // to avoid adding an extra member and to keep a single lock for storing metrics.
        int num_rays = 1080 / ANGLE_STEP;  // Downsampled ray count
        double effective_rays = num_rays / (1.0 / INV_SQUASH_FACTOR);  // Account for squash factor
        double ray_match_percentage = 0.0;
        if (max_weight > 0.0) {
            double avg_ray_prob = std::pow(max_weight, 1.0 / effective_rays);
            ray_match_percentage = avg_ray_prob * 100.0;
        }

        {
            std::lock_guard<std::mutex> lock(mcl_result_lock_);
            latest_mcl_timestamp_ = lidar_timestamp;
            latest_map_to_odom_ = map_to_odom_result;

            // Store percentage into latest_max_weight_ (reusing this field as percentage)
            latest_max_weight_ = ray_match_percentage;
            latest_covariance_ = covariance;
            latest_particle_spread_ = particle_spread;

            RCLCPP_DEBUG(this->get_logger(),
                "[MCL Worker] Stored map->odom [%.3f, %.3f, %.3f] for publishing",
                map_to_odom_result[0], map_to_odom_result[1], map_to_odom_result[2]);
        }

        // 10. Publish particles (visualization)
        auto particle_start = std::chrono::high_resolution_clock::now();
        visualization::publish_particles_viz(this, current_pose_base, lidar_timestamp);
        auto particle_end = std::chrono::high_resolution_clock::now();
        double particle_time_ms = std::chrono::duration<double, std::milli>(particle_end - particle_start).count();

        // 11. Performance logging (periodic)
        static int mcl_update_count = 0;
        mcl_update_count++;

        if (mcl_update_count % 20 == 0) {
            auto worker_end = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration<double, std::milli>(worker_end - worker_start).count();

            // Build detailed performance message
            double avg_quality_time = timing_stats_.measurement_count > 0
                ? timing_stats_.quality_metrics_time / timing_stats_.measurement_count : 0.0;

            std::string msg = "MCL #" + std::to_string(mcl_update_count) + " [" + SENSOR_MODEL_TYPE + "] - ";
                            // + "Total: " + std::to_string(total_time) + "ms, " 
                            // + "TF_Motion: " + std::to_string(motion_calc_time_ms) + "ms, " 
                            // + "MCL: " + std::to_string(mcl_processing_time_) + "ms " 
                            // + "(Quality: " + std::to_string(avg_quality_time) + "ms), " 
                            // + "Particles: " + std::to_string(particle_time_ms) + "ms";

        // Add sensor model specific breakdown (current update)
        if (SENSOR_MODEL_TYPE == "beam") {
            // msg += " | Current: (Query: " + std::to_string(timing_stats_.current_query_prep_time) + "ms, " +
            //        "Raycast: " + std::to_string(timing_stats_.current_ray_casting_time) + "ms, " +
            //        "Sensor: " + std::to_string(timing_stats_.current_sensor_model_time) + "ms)";
            // Also add average for comparison
            if (timing_stats_.measurement_count > 0) {
                double avg_query = timing_stats_.query_prep_time / timing_stats_.measurement_count;
                double avg_raycast = timing_stats_.ray_casting_time / timing_stats_.measurement_count;
                double avg_sensor = timing_stats_.sensor_model_time / timing_stats_.measurement_count;
                // msg += " | Avg: (Query: " + std::to_string(avg_query) + "ms, " +
                //        "Raycast: " + std::to_string(avg_raycast) + "ms, " +
                //        "Sensor: " + std::to_string(avg_sensor) + "ms)";
            }
        } else if (timing_stats_.measurement_count > 0) {
            double avg_sensor = timing_stats_.sensor_model_time / timing_stats_.measurement_count;
            // msg += " (LikelihoodField(avg): " + std::to_string(avg_sensor) + "ms)";
        }

        // Add ESS statistics if adaptive resampling is enabled
        if (USE_ADAPTIVE_RESAMPLING && timing_stats_.ess_count > 0) {
            double avg_ess = timing_stats_.ess_sum / timing_stats_.ess_count;
            double resample_rate = (static_cast<double>(timing_stats_.resample_count) / timing_stats_.ess_count) * 100.0;
            msg += ", ESS: " + std::to_string(static_cast<int>(avg_ess)) + "/" + std::to_string(MAX_PARTICLES) +
                   " (" + std::to_string(static_cast<int>(avg_ess * 100.0 / MAX_PARTICLES)) + "%), " +
                   "Resample: " + std::to_string(static_cast<int>(resample_rate)) + "%";
        }

        // msg += ", Pose: [" + std::to_string(current_pose_base[0]) + ", " +
        //        std::to_string(current_pose_base[1]) + ", " +
        //        std::to_string(current_pose_base[2]) + "]";

        RCLCPP_INFO(this->get_logger(), "%s", msg.c_str());

        // Localization quality metrics (already computed above with state_lock_)
        double pos_uncertainty = std::sqrt(covariance(0,0) + covariance(1,1));
        double angle_uncertainty = std::sqrt(covariance(2,2)) * 180.0 / M_PI;

        RCLCPP_INFO(this->get_logger(),
            "Quality | Ray Match: %.1f%% (raw: %.4f) | Pos Unc: %.3fm | Angle Unc: %.1f° | Spread: %.3fm",
            ray_match_percentage, max_weight, pos_uncertainty, angle_uncertainty, particle_spread
        );

        // Also print detailed timing stats collected over the last interval
        // timing_stats_.print_stats([this](const std::string &s) {
        //     RCLCPP_INFO(this->get_logger(), "%s", s.c_str());
        // });

            // Reset timing stats for next 100 iterations
            timing_stats_.reset();
        }

        // 12. Increment run counter
        consecutive_runs++;

        // 13. Check if new pending data arrived during MCL processing
        {
            std::lock_guard<std::mutex> lock(lidar_lock_);
            if (!pending_mcl_data_) {
                // No new data, worker done
                mcl_running_ = false;
                RCLCPP_DEBUG(this->get_logger(),
                    "[MCL Worker] Completed (consecutive_runs=%d)", consecutive_runs);
                return;
            }
            // New data available, continue while loop
            RCLCPP_DEBUG(this->get_logger(),
                "[MCL Worker] Continuing with new pending data (run %d/%d)",
                consecutive_runs + 1, MAX_CONSECUTIVE_MCL_RUNS);
        }
    }

    // Exited while loop - reached maximum consecutive runs
    mcl_running_ = false;
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
        "[MCL Worker] Reached max consecutive runs (%d) - forcing break. "
        "MCL is too slow for current LiDAR rate!",
        MAX_CONSECUTIVE_MCL_RUNS);
}

Eigen::Vector3d MCL::calculate_lidar_frame_motion(const rclcpp::Time& current_lidar_stamp,
                                                   const Eigen::Vector3d& current_odom_to_base)
{
    static Eigen::Vector3d last_pose = Eigen::Vector3d::Zero();
    static bool first_call = true;

    if (first_call) {
        first_call = false;
        // Initialize with provided odom->base
        last_pose = current_odom_to_base;
        RCLCPP_DEBUG(this->get_logger(),
            "[Motion] Initialized with odom->base [%.3f, %.3f, %.3f]",
            current_odom_to_base[0], current_odom_to_base[1], current_odom_to_base[2]);
        return Eigen::Vector3d::Zero();
    }

    // Use the provided odom->base (already computed with exact time or extrapolation)
    Eigen::Vector3d current_pose = current_odom_to_base;

    // Calculate motion from last stored pose to current
    Eigen::Vector3d motion = utils::transforms::calculate_lidar_frame_motion(current_pose, last_pose);

    // Store current pose for next iteration
    last_pose = current_pose;

    RCLCPP_DEBUG(this->get_logger(),
        "[Motion] Computed motion [%.3f, %.3f, %.3f] from odom delta",
        motion[0], motion[1], motion[2]);

    return motion;
}

/**
 * @brief Applies motion model to particles with Gaussian noise
 */
void MCL::motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action)
{
    // Select motion model based on parameter
    if (MOTION_MODEL_TYPE == "odometry") {
        // RTR-based odometry motion model (Probabilistic Robotics Ch 5.4)
        motion_model::odometry_motion_update(this, proposal_dist, action);
    } else {
        // Simple motion model (fixed Gaussian noise)
        motion_model::simple_motion_update(this, proposal_dist, action);
    }
}

/**
 * @brief Evaluates sensor model likelihood - dispatches to beam or likelihood_field model
 */
void MCL::sensor_model(const Eigen::MatrixXd &proposal_dist, const std::vector<float> &obs,
                                  std::vector<double> &weights)
{
    // Select sensor model based on parameter
    if (SENSOR_MODEL_TYPE == "beam") {
        // Beam model (with ray casting)
        sensor_model::beam_sensor_update(this, proposal_dist, obs, weights);
    } else {
        // Likelihood field model (no ray casting)
        sensor_model::likelihood_field_sensor_update(this, proposal_dist, obs, weights);
    }
}


/**
 * @brief Calculate Effective Sample Size (ESS)
 * @return ESS value (1 to MAX_PARTICLES)
 *
 * ESS measures the diversity of particle weights.
 * ESS = 1 / Σ(w_i²)
 *
 * - ESS = MAX_PARTICLES: all weights equal (ideal)
 * - ESS = 1: one particle has all weight (degenerate)
 */
double MCL::calculate_ess()
{
    double sum_sq = 0.0;
    for (const auto& w : weights_)
    {
        sum_sq += w * w;
    }

    // Avoid division by zero
    if (sum_sq < 1e-10)
    {
        return 0.0;
    }

    return 1.0 / sum_sq;
}

/**
 * @brief Execute resampling using selected method
 *
 * Dispatches to either multinomial or low variance resampling based on
 * RESAMPLING_TYPE parameter. Resets weights to uniform after resampling.
 */
void MCL::resample()
{
    // Select resampling method and resample directly into particles_
    if (RESAMPLING_TYPE == "low_variance")
    {
        resampling_model::low_variance_resample(this, proposal_distribution_, weights_, particles_);
    }
    else
    {
        resampling_model::multinomial_resample(this, proposal_distribution_, weights_, particles_);
    }

    // Reset weights to uniform after resampling
    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);
}

/**
 * @brief Executes complete MCL cycle: predict, update, normalize, resample
 * @return MCLQualityMetrics containing mean pose, covariance, max_weight, and particle_spread
 *         These metrics are computed BEFORE resampling when weights are still meaningful
 */
MCLQualityMetrics MCL::run_mcl(const Eigen::Vector3d &action, const std::vector<float> &observation)
{
    auto mcl_start = std::chrono::high_resolution_clock::now();
    
    // 1. Motion prediction - apply motion model to current particles
    auto motion_start = std::chrono::high_resolution_clock::now();
    // Copy particles to proposal distribution for motion prediction
    proposal_distribution_ = particles_;
    motion_model(proposal_distribution_, action);
    auto motion_end = std::chrono::high_resolution_clock::now();
    timing_stats_.motion_model_time += std::chrono::duration<double, std::milli>(motion_end - motion_start).count();

    // 2. Sensor likelihood evaluation
    sensor_model(proposal_distribution_, observation, weights_);

    // 3. Weight normalization
    double sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (sum_weights > 0)
    {
        for (double &w : weights_)
        {
            w /= sum_weights;
        }
    }

    // 3.5. Calculate max_weight BEFORE resampling (when weights are meaningful)
    // Max weight measures sensor model confidence - must be calculated before resampling
    // because after resampling, all weights become uniform (1/MAX_PARTICLES)
    auto quality_start = std::chrono::high_resolution_clock::now();

    double max_weight = weights_[0];
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        if (weights_[i] > max_weight) {
            max_weight = weights_[i];
        }
    }

    auto quality_end = std::chrono::high_resolution_clock::now();
    double quality_time_ms = std::chrono::duration<double, std::milli>(quality_end - quality_start).count();
    timing_stats_.quality_metrics_time += quality_time_ms;

    // 4. Adaptive Resampling (with ESS check if enabled)
    auto resample_start = std::chrono::high_resolution_clock::now();

    if (USE_ADAPTIVE_RESAMPLING)
    {
        // Calculate Effective Sample Size
        double ess = calculate_ess();
        double ess_threshold_particles = MAX_PARTICLES * ESS_THRESHOLD;

        // Update ESS statistics
        timing_stats_.ess_sum += ess;
        timing_stats_.ess_count++;

        if (ess < ess_threshold_particles)
        {
            // ESS too low → resample
            resample();
            timing_stats_.resample_count++;
            RCLCPP_DEBUG(this->get_logger(), "Resampled (ESS: %.1f < %.1f)", ess, ess_threshold_particles);
        }
        else
        {
            // ESS good → just update particles without resampling
            particles_ = proposal_distribution_;
            RCLCPP_DEBUG(this->get_logger(), "No resample (ESS: %.1f >= %.1f)", ess, ess_threshold_particles);
        }
    }
    else
    {
        // Always resample (original behavior)
        resample();
        timing_stats_.resample_count++;
    }

    auto resample_end = std::chrono::high_resolution_clock::now();
    timing_stats_.resampling_time += std::chrono::duration<double, std::milli>(resample_end - resample_start).count();

    // 5. Calculate covariance and mean_pose AFTER resampling
    // After resampling, particles_ represents the true belief distribution
    // Weights are now uniform (1/MAX_PARTICLES), so weighted mean = unweighted mean
    auto quality_start2 = std::chrono::high_resolution_clock::now();

    // Pass 1: Calculate mean pose from final particles
    Eigen::Vector3d mean_pose = Eigen::Vector3d::Zero();
    double sum_sin = 0.0, sum_cos = 0.0;
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        mean_pose[0] += particles_(i, 0);
        mean_pose[1] += particles_(i, 1);
        sum_sin += std::sin(particles_(i, 2));
        sum_cos += std::cos(particles_(i, 2));
    }
    mean_pose[0] /= MAX_PARTICLES;
    mean_pose[1] /= MAX_PARTICLES;
    mean_pose[2] = std::atan2(sum_sin, sum_cos);

    // Pass 2: Calculate covariance and particle spread from final particles
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double particle_spread = 0.0;

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        Eigen::Vector3d diff;
        diff[0] = particles_(i, 0) - mean_pose[0];
        diff[1] = particles_(i, 1) - mean_pose[1];
        diff[2] = utils::geometry::normalize_angle(particles_(i, 2) - mean_pose[2]);

        // Unweighted covariance (weights are uniform after resampling)
        covariance += (diff * diff.transpose()) / MAX_PARTICLES;

        // Particle spread (unweighted average distance)
        particle_spread += std::sqrt(diff[0]*diff[0] + diff[1]*diff[1]);
    }
    particle_spread /= MAX_PARTICLES;

    auto quality_end2 = std::chrono::high_resolution_clock::now();
    double quality_time_ms2 = std::chrono::duration<double, std::milli>(quality_end2 - quality_start2).count();
    timing_stats_.quality_metrics_time += quality_time_ms2;

    auto mcl_end = std::chrono::high_resolution_clock::now();
    timing_stats_.total_mcl_time += std::chrono::duration<double, std::milli>(mcl_end - mcl_start).count();
    timing_stats_.measurement_count++;
    // Collect timing stats here; printing is done periodically from timer_update()

    // Return quality metrics (max_weight from before resampling, covariance from after)
    return MCLQualityMetrics{
        .mean_pose = mean_pose,
        .covariance = covariance,
        .max_weight = max_weight,
        .particle_spread = particle_spread
    };
}

/**
 * @brief Computes weighted mean pose from particles
 */
Eigen::Vector3d MCL::expected_pose()
{
    Eigen::Vector3d pose = Eigen::Vector3d::Zero();
    double sum_sin = 0.0, sum_cos = 0.0;

    // Weighted mean for x, y
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        pose[0] += weights_[i] * particles_(i, 0);  // x
        pose[1] += weights_[i] * particles_(i, 1);  // y

        // Circular mean for angles
        sum_sin += weights_[i] * std::sin(particles_(i, 2));
        sum_cos += weights_[i] * std::cos(particles_(i, 2));
    }

    // Final angle calculation
    pose[2] = std::atan2(sum_sin, sum_cos);

    return pose;
}

// Note: Old quality metric functions (get_max_weight, calculate_covariance, calculate_particle_spread)
// have been removed. These metrics are now computed directly in run_mcl() and returned via
// MCLQualityMetrics struct for better performance (single-pass calculation).

// ================================================================================================
// POSE EXTRAPOLATION UTILITIES
// ================================================================================================

Eigen::Vector3d MCL::extrapolatePose(const Eigen::Vector3d& base_pose,
                                     double linear_vel, double angular_vel,
                                     double time_diff)
{
    // Calculate position change using current heading
    double dx = linear_vel * time_diff * std::cos(base_pose[2]);
    double dy = linear_vel * time_diff * std::sin(base_pose[2]);
    double dtheta = angular_vel * time_diff;

    // Apply changes to base pose
    Eigen::Vector3d extrapolated_pose;
    extrapolated_pose[0] = base_pose[0] + dx;
    extrapolated_pose[1] = base_pose[1] + dy;
    // Normalize angle to [-π, π] to prevent accumulation
    extrapolated_pose[2] = utils::geometry::normalize_angle(base_pose[2] + dtheta);

    return extrapolated_pose;
}

// ================================================================================================
// MAP PUBLISHING
// ================================================================================================

void MCL::publish_map_periodically()
{
    if (!map_initialized_ || !map_pub_)
        return;

    std::lock_guard<std::mutex> lock(map_lock_);
    if (map_msg_) {
        map_pub_->publish(*map_msg_);
    }
}

// ================================================================================================
// POSE PUBLISHING (triggered by odom callback)
// ================================================================================================
void MCL::publish_pose(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
{
    // Get latest map->odom transform (from MCL computation)
    Eigen::Vector3d map_to_odom_vec;
    {
        std::lock_guard<std::mutex> lock(mcl_result_lock_);
        map_to_odom_vec = latest_map_to_odom_;
    }

    // Extract odom->base from incoming message (no extrapolation needed)
    Eigen::Vector3d odom_to_base_vec(
        odom_msg->pose.pose.position.x,
        odom_msg->pose.pose.position.y,
        utils::geometry::quaternion_to_yaw(odom_msg->pose.pose.orientation)
    );

    rclcpp::Time timestamp = odom_msg->header.stamp;

    // Publish map->odom TF
    if (PUBLISH_MAP_ODOM_TF) {
        geometry_msgs::msg::TransformStamped map_to_odom_msg;
        map_to_odom_msg.header.stamp = timestamp;
        map_to_odom_msg.header.frame_id = MAP_FRAME;
        map_to_odom_msg.child_frame_id = ODOM_FRAME;

        map_to_odom_msg.transform.translation.x = map_to_odom_vec[0];
        map_to_odom_msg.transform.translation.y = map_to_odom_vec[1];
        map_to_odom_msg.transform.translation.z = 0.0;

        tf2::Quaternion map_to_odom_quat;
        map_to_odom_quat.setRPY(0, 0, map_to_odom_vec[2]);
        map_to_odom_msg.transform.rotation = tf2::toMsg(map_to_odom_quat);

        pub_tf_->sendTransform(map_to_odom_msg);
    }

    // Compute map->base transform (needed for odometry message and visualization)
    // Create map->odom transform (needed for TF multiplication)
    tf2::Transform map_to_odom_tf;
    map_to_odom_tf.setOrigin(tf2::Vector3(map_to_odom_vec[0], map_to_odom_vec[1], 0.0));
    tf2::Quaternion map_to_odom_quat;
    map_to_odom_quat.setRPY(0, 0, map_to_odom_vec[2]);
    map_to_odom_tf.setRotation(map_to_odom_quat);

    // Create odom->base transform (needed for TF multiplication)
    tf2::Transform odom_to_base_tf;
    odom_to_base_tf.setOrigin(tf2::Vector3(
        odom_to_base_vec[0], odom_to_base_vec[1], 0.0));
    tf2::Quaternion odom_to_base_quat;
    odom_to_base_quat.setRPY(0, 0, odom_to_base_vec[2]);
    odom_to_base_tf.setRotation(odom_to_base_quat);

    // Compute map->base = map->odom × odom->base
    tf2::Transform map_to_base_tf = map_to_odom_tf * odom_to_base_tf;

    // Publish map->base as odometry topic with covariance (optional, only if subscribers exist)
    if (PUBLISH_ODOM && odom_pub_ && odom_pub_->get_subscription_count() > 0) {
        nav_msgs::msg::Odometry odom_out_msg;
        odom_out_msg.header.stamp = timestamp;
        odom_out_msg.header.frame_id = MAP_FRAME;
        odom_out_msg.child_frame_id = BASE_FRAME;

        odom_out_msg.pose.pose.position.x = map_to_base_tf.getOrigin().x();
        odom_out_msg.pose.pose.position.y = map_to_base_tf.getOrigin().y();
        odom_out_msg.pose.pose.position.z = 0.0;

        odom_out_msg.pose.pose.orientation = tf2::toMsg(map_to_base_tf.getRotation());

        // Add MCL-estimated velocity (actual ground velocity considering slip)
        // These velocities are in the robot's base_link frame
        if (velocity_initialized_) {
            odom_out_msg.twist.twist.linear.x = estimated_vx_;   // Forward velocity (m/s)
            odom_out_msg.twist.twist.linear.y = estimated_vy_;   // Lateral velocity (m/s) - slip detection
            odom_out_msg.twist.twist.linear.z = 0.0;
            odom_out_msg.twist.twist.angular.x = 0.0;
            odom_out_msg.twist.twist.angular.y = 0.0;
            odom_out_msg.twist.twist.angular.z = estimated_vyaw_;  // Yaw rate (rad/s)
        } else {
            // Fallback to odometry velocity if MCL velocity not initialized yet
            odom_out_msg.twist.twist.linear.x = current_velocity_;
            odom_out_msg.twist.twist.linear.y = 0.0;
            odom_out_msg.twist.twist.linear.z = 0.0;
            odom_out_msg.twist.twist.angular.x = 0.0;
            odom_out_msg.twist.twist.angular.y = 0.0;
            odom_out_msg.twist.twist.angular.z = current_angular_vel_;
        }

        // Add covariance from particle filter for RViz visualization
        // RViz will display this as ellipse (x,y) and arc (yaw)
        // Read pre-computed covariance (thread-safe copy)
        Eigen::Matrix3d cov;
        {
            std::lock_guard<std::mutex> lock(mcl_result_lock_);
            cov = latest_covariance_;  // Atomic copy of 9 doubles
        }

        // Use actual covariance values without scaling
        // Covariance represents true localization uncertainty and should be shown accurately
        // to gauge the real magnitude of position/orientation uncertainty

        // Fill 6x6 covariance matrix (x, y, z, roll, pitch, yaw)
        // ROS uses row-major order: index = row * 6 + col
        std::fill(odom_out_msg.pose.covariance.begin(), odom_out_msg.pose.covariance.end(), 0.0);

        odom_out_msg.pose.covariance[0] = cov(0, 0);   // cov[0][0] = x-x
        odom_out_msg.pose.covariance[1] = cov(0, 1);   // cov[0][1] = x-y
        odom_out_msg.pose.covariance[6] = cov(0, 1);   // cov[1][0] = y-x (symmetric)
        odom_out_msg.pose.covariance[7] = cov(1, 1);   // cov[1][1] = y-y
        odom_out_msg.pose.covariance[35] = cov(2, 2);  // cov[5][5] = yaw-yaw

        odom_pub_->publish(odom_out_msg);
    }

    // Publish pose visualization for RViz (PoseStamped message)
    Eigen::Vector3d map_to_base_vec(
        map_to_base_tf.getOrigin().x(),
        map_to_base_tf.getOrigin().y(),
        tf2::getYaw(map_to_base_tf.getRotation())
    );
    visualization::publish_localization(this, map_to_base_vec, timestamp);

    // Publish max weight visualization (color-coded quality indicator)
    if (quality_marker_pub_ && quality_marker_pub_->get_subscription_count() > 0) {
        // Read ray-match percentage (stored into latest_max_weight_ by the worker)
        double ray_match_pct = 0.0;
        {
            std::lock_guard<std::mutex> lock(mcl_result_lock_);
            ray_match_pct = latest_max_weight_;  // percentage in [0..100]
        }

        visualization_msgs::msg::Marker marker;
        marker.header.stamp = timestamp;
        marker.header.frame_id = MAP_FRAME;
        marker.ns = "localization_quality";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Position above the robot
        marker.pose.position.x = map_to_base_vec[0];
        marker.pose.position.y = map_to_base_vec[1];
        marker.pose.position.z = 0.6;  // 60cm above (more visible)
        marker.pose.orientation.w = 1.0;

        // Size (increased for better visibility)
        marker.scale.x = 0.30;  // 30cm diameter
        marker.scale.y = 0.30;
        marker.scale.z = 0.30;

        // Color based on ray match percentage (higher = better)
        if (ray_match_pct >= 85.0) { // EXCELLENT >=85%
            // EXCELLENT: Green
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        } else if (ray_match_pct >= 80.0) { // GOOD >=80%
            // GOOD: Yellow-Green
            marker.color.r = 0.5;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        } else if (ray_match_pct >= 70.0) { // FAIR >=70%
            // FAIR: Yellow
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        } else if (ray_match_pct >= 60.0) { // POOR >=60%
            // POOR: Orange
            marker.color.r = 1.0;
            marker.color.g = 0.5;
            marker.color.b = 0.0;
        } else {
            // CRITICAL: Red
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
        }
        marker.color.a = 0.8;  // Slightly transparent

        marker.lifetime = rclcpp::Duration::from_seconds(0.2);  // 200ms lifetime

        quality_marker_pub_->publish(marker);
    }
}

} // namespace mcl_pkg

// ================================================================================================
// PROGRAM ENTRY POINT
// ================================================================================================
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<mcl_pkg::MCL>();
    
    // Use MultiThreadedExecutor for parallel callback processing
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    
    RCLCPP_INFO(node->get_logger(), "Monte Carlo Localization node started - spinning...");
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}
