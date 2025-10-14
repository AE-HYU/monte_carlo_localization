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

#include <algorithm>
#include <chrono>
#include <cmath>
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
            // Reserve 2 threads for executor workers and 1 for system
            NUM_THREADS = std::max(1, hw - 3);
        } else if (NUM_THREADS > 0) {
            NUM_THREADS = std::min(NUM_THREADS, hw - 3);
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

    // Timers
    update_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / TIMER_FREQUENCY)),
        std::bind(&MCL::timer_update, this),
        compute_group_);

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

    RCLCPP_INFO(this->get_logger(), "Particle filter initialized - %.1fHz, %s threading (%d threads)",
        TIMER_FREQUENCY, USE_PARALLEL_RAYCASTING ? "parallel" : "sequential",
        USE_PARALLEL_RAYCASTING ? NUM_THREADS : 1);
    RCLCPP_INFO(this->get_logger(), "Motion model: %s | Sensor model: %s | Resampling: %s%s",
        MOTION_MODEL_TYPE.c_str(), SENSOR_MODEL_TYPE.c_str(), RESAMPLING_TYPE.c_str(),
        USE_ADAPTIVE_RESAMPLING ? " (adaptive)" : "");
    RCLCPP_INFO(this->get_logger(), "Async map loading started - node ready, waiting for map server...");
    RCLCPP_INFO(this->get_logger(), "Dynamic parameter reconfiguration enabled");
}

MCL::~MCL()
{
    RCLCPP_INFO(this->get_logger(), "Shutting down particle filter");

    // Cancel all timers explicitly for clean shutdown
    if (update_timer_) update_timer_->cancel();
    if (map_loader_timer_) map_loader_timer_->cancel();
    if (map_timer_) map_timer_->cancel();

    // Other resources (smart pointers, Eigen matrices) are cleaned up automatically via RAII
}

// ================================================================================================
// ROS2 CALLBACKS
// ================================================================================================

void MCL::lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg)
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
    has_new_lidar_data_ = true;
}

void MCL::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(odom_lock_);

    current_velocity_ = msg->twist.twist.linear.x;
    current_angular_vel_ = msg->twist.twist.angular.z;

    if (!odom_initialized_) {
        odom_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "Odometry initialized");
    }
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
// MAIN UPDATE LOOP
// ================================================================================================

void MCL::timer_update()
{
    auto timer_start = std::chrono::high_resolution_clock::now();

    if (!lidar_initialized_ || !odom_initialized_ || !map_initialized_)
    {
        static int wait_count = 0;
        if (++wait_count % 50 == 0) {
            RCLCPP_WARN(this->get_logger(),
                "Still waiting for initialization (%d attempts) - LiDAR: %s, Odom: %s, Map: %s",
                wait_count,
                lidar_initialized_ ? "OK" : "NO",
                odom_initialized_ ? "OK" : "NO",
                map_initialized_ ? "OK" : "NO");
        }
        return;
    }

    // Check for new LiDAR data and copy if available
    bool has_new_data = false;
    std::vector<float> observation;
    Eigen::Vector3d action;
    rclcpp::Time lidar_timestamp;
    static rclcpp::Time latest_sensor_time = rclcpp::Time(0);

    {
        std::lock_guard<std::mutex> lock(lidar_lock_);
        has_new_data = has_new_lidar_data_;
        if (has_new_data) {
            has_new_lidar_data_ = false;
            observation = downsampled_ranges_;
            lidar_timestamp = last_lidar_time_;
            latest_sensor_time = lidar_timestamp;  // Update latest timestamp
        }
    }

    // Early return if no new LiDAR data (save CPU, prevent stale TF republishing)
    if (!has_new_data) {
        // rclcpp::Time now = this->get_clock()->now();
        // RCLCPP_WARN(this->get_logger(), "No new LiDAR data %f - skipping MCL update", now.seconds());
        return;
    }

    // Calculate odometry motion
    auto motion_start = std::chrono::high_resolution_clock::now();
    action = calculate_lidar_frame_motion(lidar_timestamp);
    auto motion_end = std::chrono::high_resolution_clock::now();
    double motion_calc_time_ms = std::chrono::duration<double, std::milli>(motion_end - motion_start).count();

    // Execute MCL with state lock
    if (!state_lock_.try_lock()) {
        RCLCPP_WARN(this->get_logger(), "MCL update skipped - previous update still running");
        return;  // Skip this update entirely if previous one is still running
    }

    auto mcl_start = std::chrono::high_resolution_clock::now();
    run_mcl(action, observation);
    auto mcl_end = std::chrono::high_resolution_clock::now();
    mcl_processing_time_ = std::chrono::duration<double, std::milli>(mcl_end - mcl_start).count();

    // Expected pose calculation (state_lock already held)
    Eigen::Vector3d current_pose_laser = expected_pose();

    state_lock_.unlock();

    // Convert laser frame to base_link
    Eigen::Vector3d current_pose_base = utils::transforms::apply_laser_to_base_offset(
        current_pose_laser, laser_offset_x_, laser_offset_y_);

    // Publish localization output synchronized with LiDAR data
    auto pub_start = std::chrono::high_resolution_clock::now();
    visualization::publish_localization(this, current_pose_base, lidar_timestamp);
    auto pub_end = std::chrono::high_resolution_clock::now();
    double pub_time_ms = std::chrono::duration<double, std::milli>(pub_end - pub_start).count();

    // Publish particles
    auto particle_start = std::chrono::high_resolution_clock::now();
    visualization::publish_particles_viz(this, current_pose_base, lidar_timestamp);
    auto particle_end = std::chrono::high_resolution_clock::now();
    double particle_time_ms = std::chrono::duration<double, std::milli>(particle_end - particle_start).count();
 

    // Performance logging (periodic)
    static int mcl_update_count = 0;
    mcl_update_count++;

    if (mcl_update_count % 100 == 0) {
        auto timer_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(timer_end - timer_start).count();

        // Build detailed performance message (all values from current update)
        std::string msg = "MCL #" + std::to_string(mcl_update_count) + " [" + SENSOR_MODEL_TYPE + "] - " +
                         "Total: " + std::to_string(total_time) + "ms, " +
                         "TF_Motion: " + std::to_string(motion_calc_time_ms) + "ms, " +
                         "MCL: " + std::to_string(mcl_processing_time_) + "ms, " +
                         "Pub: " + std::to_string(pub_time_ms) + "ms, " +
                         "Particles: " + std::to_string(particle_time_ms) + "ms";

        // Add sensor model specific breakdown (current update)
        if (SENSOR_MODEL_TYPE == "beam") {
            msg += " | Current: (Query: " + std::to_string(timing_stats_.current_query_prep_time) + "ms, " +
                   "Raycast: " + std::to_string(timing_stats_.current_ray_casting_time) + "ms, " +
                   "Sensor: " + std::to_string(timing_stats_.current_sensor_model_time) + "ms)";
            // Also add average for comparison
            if (timing_stats_.measurement_count > 0) {
                double avg_query = timing_stats_.query_prep_time / timing_stats_.measurement_count;
                double avg_raycast = timing_stats_.ray_casting_time / timing_stats_.measurement_count;
                double avg_sensor = timing_stats_.sensor_model_time / timing_stats_.measurement_count;
                msg += " | Avg: (Query: " + std::to_string(avg_query) + "ms, " +
                       "Raycast: " + std::to_string(avg_raycast) + "ms, " +
                       "Sensor: " + std::to_string(avg_sensor) + "ms)";
            }
        } else if (timing_stats_.measurement_count > 0) {
            double avg_sensor = timing_stats_.sensor_model_time / timing_stats_.measurement_count;
            msg += " (LikelihoodField(avg): " + std::to_string(avg_sensor) + "ms)";
        }

        // Add ESS statistics if adaptive resampling is enabled
        if (USE_ADAPTIVE_RESAMPLING && timing_stats_.ess_count > 0) {
            double avg_ess = timing_stats_.ess_sum / timing_stats_.ess_count;
            double resample_rate = (static_cast<double>(timing_stats_.resample_count) / timing_stats_.ess_count) * 100.0;
            msg += ", ESS: " + std::to_string(static_cast<int>(avg_ess)) + "/" + std::to_string(MAX_PARTICLES) +
                   " (" + std::to_string(static_cast<int>(avg_ess * 100.0 / MAX_PARTICLES)) + "%), " +
                   "Resample: " + std::to_string(static_cast<int>(resample_rate)) + "%";
        }

        msg += ", Pose: [" + std::to_string(current_pose_base[0]) + ", " +
               std::to_string(current_pose_base[1]) + ", " +
               std::to_string(current_pose_base[2]) + "]";

        RCLCPP_INFO(this->get_logger(), "%s", msg.c_str());

        // Also print detailed timing stats collected over the last interval
        timing_stats_.print_stats([this](const std::string &s) {
            RCLCPP_INFO(this->get_logger(), "%s", s.c_str());
        });

        // Reset timing stats for next 100 iterations
        timing_stats_.reset();
    }
}

Eigen::Vector3d MCL::calculate_lidar_frame_motion(const rclcpp::Time& current_lidar_stamp)
{
    (void)current_lidar_stamp;  // Unused - we now use TimePointZero for latest TF

    static Eigen::Vector3d last_pose = Eigen::Vector3d::Zero();
    static bool first_call = true;

    if (first_call) {
        first_call = false;
        try {
            // Get initial pose using latest TF
            auto transform = tf_buffer_->lookupTransform(
                ODOM_FRAME, BASE_FRAME, tf2::TimePointZero);

            last_pose = Eigen::Vector3d(
                transform.transform.translation.x,
                transform.transform.translation.y,
                utils::geometry::quaternion_to_yaw(transform.transform.rotation)
            );
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "Could not get initial TF: %s", ex.what());
        }
        return Eigen::Vector3d::Zero();
    }

    try {
        // Use latest available TF instead of waiting for specific timestamp
        // This eliminates 10-20ms blocking delays in simulation
        auto current_transform = tf_buffer_->lookupTransform(
            ODOM_FRAME, BASE_FRAME, tf2::TimePointZero);

        Eigen::Vector3d current_pose(
            current_transform.transform.translation.x,
            current_transform.transform.translation.y,
            utils::geometry::quaternion_to_yaw(current_transform.transform.rotation)
        );

        // Calculate motion from last stored pose to current
        Eigen::Vector3d motion = utils::transforms::calculate_lidar_frame_motion(current_pose, last_pose);

        // Store current pose for next iteration
        last_pose = current_pose;
        return motion;

    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(),
            "Could not get transform for lidar frame motion calculation: %s", ex.what());
        return Eigen::Vector3d::Zero();
    }
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
 */
void MCL::run_mcl(const Eigen::Vector3d &action, const std::vector<float> &observation)
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
    
    auto mcl_end = std::chrono::high_resolution_clock::now();
    timing_stats_.total_mcl_time += std::chrono::duration<double, std::milli>(mcl_end - mcl_start).count();
    timing_stats_.measurement_count++;
    // Collect timing stats here; printing is done periodically from timer_update()
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
