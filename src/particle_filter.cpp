// ================================================================================================
// PARTICLE FILTER IMPLEMENTATION - Monte Carlo Localization (MCL)
// ================================================================================================
// Features: Multinomial resampling, velocity motion model, beam sensor model, ray casting
// ================================================================================================

#include "particle_filter_cpp/particle_filter.hpp"
#include "particle_filter_cpp/utils.hpp"
#include <algorithm>
#include <angles/angles.h>
#include <chrono>
#include <cmath>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <numeric>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <omp.h>

namespace particle_filter_cpp
{

// --------------------------------- CONSTRUCTOR & INITIALIZATION ---------------------------------
ParticleFilter::ParticleFilter(const rclcpp::NodeOptions &options)
    : Node("particle_filter", options), rng_(std::random_device{}()), uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0)
{
    // ROS2 parameter declarations
    this->declare_parameter("angle_step", 18);
    this->declare_parameter("max_particles", 4000);
    this->declare_parameter("max_viz_particles", 60);
    this->declare_parameter("squash_factor", 2.2);
    this->declare_parameter("max_range", 12.0);
    this->declare_parameter("theta_discretization", 112);
    this->declare_parameter("range_method", "rmgpu");
    this->declare_parameter("rangelib_variant", 1);
    this->declare_parameter("fine_timing", 0);
    this->declare_parameter("publish_odom", true);
    this->declare_parameter("viz", true);
    this->declare_parameter("z_short", 0.01);
    this->declare_parameter("z_max", 0.07);
    this->declare_parameter("z_rand", 0.12);
    this->declare_parameter("z_hit", 0.80);
    this->declare_parameter("sigma_hit", 8.0);
    this->declare_parameter("motion_dispersion_x", 0.05);
    this->declare_parameter("motion_dispersion_y", 0.025);
    this->declare_parameter("motion_dispersion_theta", 0.25);
    this->declare_parameter("lidar_offset_x", 0.0);
    this->declare_parameter("lidar_offset_y", 0.0);
    this->declare_parameter("wheelbase", 0.325);
    this->declare_parameter("scan_topic", "/scan");
    this->declare_parameter("odom_topic", "/odom");
    this->declare_parameter("timer_frequency", 100.0);
    this->declare_parameter("use_parallel_raycasting", true);
    this->declare_parameter("num_threads", 0); // 0 = auto-detect
    this->declare_parameter("max_pose_range", 10000.0); // Maximum valid pose coordinate range
    this->declare_parameter("sim_mode", false); // Simulation mode flag

    // Retrieve parameter values
    ANGLE_STEP = this->get_parameter("angle_step").as_int();
    MAX_PARTICLES = this->get_parameter("max_particles").as_int();
    MAX_VIZ_PARTICLES = this->get_parameter("max_viz_particles").as_int();
    INV_SQUASH_FACTOR = 1.0 / this->get_parameter("squash_factor").as_double();
    MAX_RANGE_METERS = this->get_parameter("max_range").as_double();
    THETA_DISCRETIZATION = this->get_parameter("theta_discretization").as_int();
    WHICH_RM = this->get_parameter("range_method").as_string();
    RANGELIB_VAR = this->get_parameter("rangelib_variant").as_int();
    SHOW_FINE_TIMING = this->get_parameter("fine_timing").as_int() > 0;
    PUBLISH_ODOM = this->get_parameter("publish_odom").as_bool();
    DO_VIZ = this->get_parameter("viz").as_bool();
    TIMER_FREQUENCY = this->get_parameter("timer_frequency").as_double();
    USE_PARALLEL_RAYCASTING = this->get_parameter("use_parallel_raycasting").as_bool();
    NUM_THREADS = this->get_parameter("num_threads").as_int();
    MAX_POSE_RANGE = this->get_parameter("max_pose_range").as_double();
    SIM_MODE = this->get_parameter("sim_mode").as_bool();

    // 4-component sensor model parameters
    Z_SHORT = this->get_parameter("z_short").as_double();
    Z_MAX = this->get_parameter("z_max").as_double();
    Z_RAND = this->get_parameter("z_rand").as_double();
    Z_HIT = this->get_parameter("z_hit").as_double();
    SIGMA_HIT = this->get_parameter("sigma_hit").as_double();

    // Motion model noise parameters
    MOTION_DISPERSION_X = this->get_parameter("motion_dispersion_x").as_double();
    MOTION_DISPERSION_Y = this->get_parameter("motion_dispersion_y").as_double();
    MOTION_DISPERSION_THETA = this->get_parameter("motion_dispersion_theta").as_double();

    // Robot geometry parameters
    LIDAR_OFFSET_X = this->get_parameter("lidar_offset_x").as_double();
    LIDAR_OFFSET_Y = this->get_parameter("lidar_offset_y").as_double();
    WHEELBASE = this->get_parameter("wheelbase").as_double();

    // System state initialization
    MAX_RANGE_PX = 0;
    odometry_data_ = Eigen::Vector3d::Zero();
    iters_ = 0;
    map_initialized_ = false;
    lidar_initialized_ = false;
    odom_initialized_ = false;
    first_sensor_update_ = true;
    current_speed_ = 0.0;
    current_angular_velocity_ = 0.0;
    
    // Velocity-based MCL initialization
    current_velocity_ = 0.0;
    current_angular_vel_ = 0.0;
    new_lidar_available_ = false;
    last_mcl_update_time_ = rclcpp::Time(0);  // Initialize to 0 explicitly
    
    // Odometry-based tracking initialization
    odom_pose_ = Eigen::Vector3d::Zero();
    odom_reference_pose_ = Eigen::Vector3d::Zero();
    odom_reference_odom_ = Eigen::Vector3d::Zero();
    pose_initialized_from_rviz_ = false;
    odom_tracking_active_ = false;
    
    // --------------------------------- THREADING SETUP ---------------------------------
    // Setup OpenMP for parallel ray casting
    if (USE_PARALLEL_RAYCASTING) {
        if (NUM_THREADS == 0) {
            NUM_THREADS = omp_get_max_threads();
        }
        omp_set_num_threads(NUM_THREADS);
    }

    // Initialize particles with uniform weights
    particles_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);
    weights_.resize(MAX_PARTICLES, 1.0 / MAX_PARTICLES);
    particle_indices_.resize(MAX_PARTICLES);
    std::iota(particle_indices_.begin(), particle_indices_.end(), 0);

    // Motion model cache
    local_deltas_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);

    // ROS2 publishers for visualization and navigation
    if (DO_VIZ)
    {
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 1);
        particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
    }

    if (PUBLISH_ODOM)
    {
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", 1);
    }

    // Map publisher for persistent map display in RViz
    map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(1).transient_local());

    // Initialize TF broadcaster
    pub_tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Setup subscribers
    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        this->get_parameter("scan_topic").as_string(), 1,
        std::bind(&ParticleFilter::lidarCB, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        this->get_parameter("odom_topic").as_string(), 1,
        std::bind(&ParticleFilter::odomCB, this, std::placeholders::_1));

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1, std::bind(&ParticleFilter::clicked_pose, this, std::placeholders::_1));

    click_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/clicked_point", 1, std::bind(&ParticleFilter::clicked_point, this, std::placeholders::_1));

    // Initialize map service client
    map_client_ = this->create_client<nav_msgs::srv::GetMap>("/map_server/map");

    // Get the map
    get_omap();
    initialize_global();

    // Setup configurable frequency update timer for motion interpolation
    int timer_interval_ms = static_cast<int>(1000.0 / TIMER_FREQUENCY);
    update_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(timer_interval_ms),
        std::bind(&ParticleFilter::timer_update, this)
    );

    // Setup periodic map publisher timer (5 Hz for persistent display)
    map_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(200),
        std::bind(&ParticleFilter::publish_map_periodically, this)
    );

    RCLCPP_INFO(this->get_logger(), "Particle filter initialized with %.1fHz odometry publishing", TIMER_FREQUENCY);
    RCLCPP_INFO(this->get_logger(), "Ray casting method: TRADITIONAL");
    RCLCPP_INFO(this->get_logger(), "Parallel ray casting: %s (%d threads)", 
        USE_PARALLEL_RAYCASTING ? "ENABLED" : "DISABLED", USE_PARALLEL_RAYCASTING ? NUM_THREADS : 1);
}

// --------------------------------- MAP LOADING & PREPROCESSING ---------------------------------
void ParticleFilter::get_omap()
{
    RCLCPP_INFO(this->get_logger(), "Requesting map from map server...");

    while (!map_client_->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok())
            return;
        RCLCPP_INFO(this->get_logger(), "Get map service not available, waiting...");
    }

    auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();
    auto future = map_client_->async_send_request(request);

    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS)
    {
        map_msg_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(future.get()->map);
        map_resolution_ = map_msg_->info.resolution;
        map_origin_ = Eigen::Vector3d(map_msg_->info.origin.position.x, map_msg_->info.origin.position.y,
                                      quaternion_to_angle(map_msg_->info.origin.orientation));

        MAX_RANGE_PX = static_cast<int>(MAX_RANGE_METERS / map_resolution_);

        RCLCPP_INFO(this->get_logger(), "Initializing range method: %s", WHICH_RM.c_str());

        // Extract free space (occupancy = 0) for particle initialization
        int height = map_msg_->info.height;
        int width = map_msg_->info.width;
        permissible_region_ = Eigen::MatrixXi::Zero(height, width);

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int idx = i * width + j;
                if (idx < static_cast<int>(map_msg_->data.size()) && map_msg_->data[idx] == 0)
                {
                    permissible_region_(i, j) = 1; // permissible
                }
            }
        }

        map_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "Done loading map");

        // Publish map immediately after loading
        if (map_pub_) {
            map_pub_->publish(*map_msg_);
            RCLCPP_INFO(this->get_logger(), "Map published to /map topic");
        }

        // Generate lookup table for fast sensor model evaluation
        precompute_sensor_model();
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get map from map server");
    }
}

// --------------------------------- SENSOR MODEL PRECOMPUTATION ---------------------------------
void ParticleFilter::precompute_sensor_model()
{
    RCLCPP_INFO(this->get_logger(), "Precomputing sensor model");

    if (map_resolution_ <= 0.0)
    {
        RCLCPP_ERROR(this->get_logger(), "Invalid map resolution: %.6f", map_resolution_);
        return;
    }

    int table_width = MAX_RANGE_PX + 1;
    sensor_model_table_ = Eigen::MatrixXd::Zero(table_width, table_width);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build lookup table
    for (int d = 0; d < table_width; ++d)  // d = expected range
    {
        double norm = 0.0;

        for (int r = 0; r < table_width; ++r)  // r = observed range
        {
            double prob = 0.0;
            double z = static_cast<double>(r - d);

            // Z_HIT: Gaussian around expected range
            prob += Z_HIT * std::exp(-(z * z) / (2.0 * SIGMA_HIT * SIGMA_HIT)) / (SIGMA_HIT * std::sqrt(2.0 * M_PI));

            // Z_SHORT: Exponential for early obstacles
            if (r < d)
            {
                prob += 2.0 * Z_SHORT * (d - r) / static_cast<double>(d);
            }

            // Z_MAX: Delta function at maximum range
            if (r == MAX_RANGE_PX)
            {
                prob += Z_MAX;
            }

            // Z_RAND: Uniform distribution
            if (r < MAX_RANGE_PX)
            {
                prob += Z_RAND * 1.0 / static_cast<double>(MAX_RANGE_PX);
            }

            norm += prob;
            sensor_model_table_(r, d) = prob;
        }

        // Normalize
        if (norm > 0)
        {
            sensor_model_table_.col(d) /= norm;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_INFO(this->get_logger(), "Sensor model precomputed in %ld ms", duration.count());
}

// --------------------------------- SENSOR CALLBACKS ---------------------------------
void ParticleFilter::lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    if (laser_angles_.empty())
    {
        RCLCPP_INFO(this->get_logger(), "...Received first LiDAR message");

        // Extract scan parameters and downsample
        laser_angles_.resize(msg->ranges.size());
        for (size_t i = 0; i < msg->ranges.size(); ++i)
        {
            laser_angles_[i] = msg->angle_min + i * msg->angle_increment;
        }

        // Create downsampled angles
        for (size_t i = 0; i < laser_angles_.size(); i += ANGLE_STEP)
        {
            downsampled_angles_.push_back(laser_angles_[i]);
        }

        RCLCPP_INFO(this->get_logger(), "Downsampled to %zu angles", downsampled_angles_.size());
    }

    // Extract every ANGLE_STEP-th measurement
    downsampled_ranges_.clear();
    for (size_t i = 0; i < msg->ranges.size(); i += ANGLE_STEP)
    {
        downsampled_ranges_.push_back(msg->ranges[i]);
    }

    lidar_initialized_ = true;
    new_lidar_available_ = true;  // Set flag for new lidar data
}

void ParticleFilter::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    static bool first_odom_received = false;
    if (!first_odom_received) {
        RCLCPP_INFO(this->get_logger(), "...Received first Odometry message");
        first_odom_received = true;
    }
    
    // Store velocity information for bicycle model
    current_velocity_ = msg->twist.twist.linear.x;  // Linear velocity (m/s)
    current_angular_vel_ = msg->twist.twist.angular.z;  // Angular velocity (rad/s)
    
    // Keep legacy variables for backward compatibility with other parts
    current_speed_ = msg->twist.twist.linear.x;
    current_angular_velocity_ = msg->twist.twist.angular.z;
    last_odom_time_ = msg->header.stamp;

    // Update odometry-based pose tracking (high frequency)
    // Enable when: 1) pose initialized from RViz, OR 2) global initialization done with valid MCL estimate
    bool can_use_odom_tracking = pose_initialized_from_rviz_ || 
                               (map_initialized_ && iters_ > 0 && is_pose_valid(inferred_pose_));
    
    if (can_use_odom_tracking && odom_tracking_active_) {
        update_odom_pose(msg);
    }

    // For backward compatibility, store position data
    Eigen::Vector3d position(msg->pose.pose.position.x, msg->pose.pose.position.y,
                             quaternion_to_angle(msg->pose.pose.orientation));

    if (last_pose_.norm() <= 0)
    {
        RCLCPP_INFO(this->get_logger(), "...Received first Odometry message");
        last_pose_ = position;
    }
    
    last_pose_ = position;
    last_stamp_ = msg->header.stamp;
    odom_initialized_ = true;
    
    // No longer trigger MCL update from odom callback - will be handled by timer
}

// --------------------------------- INTERACTIVE INITIALIZATION ---------------------------------
void ParticleFilter::clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    Eigen::Vector3d pose(msg->pose.pose.position.x, msg->pose.pose.position.y,
                         quaternion_to_angle(msg->pose.pose.orientation));
    
    // Initialize particle filter around clicked pose
    initialize_particles_pose(pose);
    
    // Initialize odometry-based tracking from this pose
    initialize_odom_tracking(pose);
    
    // Set inferred pose immediately for visualization
    inferred_pose_ = pose;
    
    RCLCPP_INFO(this->get_logger(), "Pose initialized from RViz at [%.3f, %.3f, %.3f]", 
                pose[0], pose[1], pose[2]);
    
    // Trigger immediate visualization update
    visualize();
}

void ParticleFilter::clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr /*msg*/)
{
    initialize_global();
}

// --------------------------------- PARTICLE INITIALIZATION ---------------------------------
void ParticleFilter::initialize_particles_pose(const Eigen::Vector3d &pose)
{
    RCLCPP_INFO(this->get_logger(), "SETTING POSE");
    RCLCPP_INFO(this->get_logger(), "Position: [%.3f, %.3f]", pose[0], pose[1]);

    std::lock_guard<std::mutex> lock(state_lock_);

    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

    // Use clicked pose directly - simple approach
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        particles_(i, 0) = pose[0] + normal_dist_(rng_) * 0.5;  // σ_x = 0.5m
        particles_(i, 1) = pose[1] + normal_dist_(rng_) * 0.5;  // σ_y = 0.5m
        particles_(i, 2) = pose[2] + normal_dist_(rng_) * 0.4;  // σ_θ = 0.4rad
        
        // Normalize angle to [-π, π] range
        while (particles_(i, 2) > M_PI) particles_(i, 2) -= 2.0 * M_PI;
        while (particles_(i, 2) < -M_PI) particles_(i, 2) += 2.0 * M_PI;
    }
}

void ParticleFilter::initialize_global()
{
    if (!map_initialized_)
        return;

    RCLCPP_INFO(this->get_logger(), "GLOBAL INITIALIZATION");

    std::lock_guard<std::mutex> lock(state_lock_);

    // Extract all free space cells
    std::vector<std::pair<int, int>> permissible_positions;
    for (int i = 0; i < permissible_region_.rows(); ++i)
    {
        for (int j = 0; j < permissible_region_.cols(); ++j)
        {
            if (permissible_region_(i, j) == 1)
            {
                permissible_positions.emplace_back(i, j);
            }
        }
    }

    if (permissible_positions.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No permissible positions found in map!");
        return;
    }

    // Uniform sampling over free space
    std::uniform_int_distribution<int> pos_dist(0, permissible_positions.size() - 1);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        int idx = pos_dist(rng_);
        auto pos = permissible_positions[idx];

        // Grid to world coordinate transformation
        particles_(i, 0) = pos.second * map_resolution_ + map_origin_[0];
        particles_(i, 1) = pos.first * map_resolution_ + map_origin_[1];
        particles_(i, 2) = angle_dist(rng_);
    }

    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

    RCLCPP_INFO(this->get_logger(), "Initialized %d particles from %zu permissible positions", MAX_PARTICLES,
                permissible_positions.size());
}

// --------------------------------- MCL ALGORITHM CORE ---------------------------------
void ParticleFilter::motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action)
{
    // Apply motion transformation: local → global coordinates
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        double cos_theta = std::cos(proposal_dist(i, 2));
        double sin_theta = std::sin(proposal_dist(i, 2));

        local_deltas_(i, 0) = cos_theta * action[0] - sin_theta * action[1];
        local_deltas_(i, 1) = sin_theta * action[0] + cos_theta * action[1];
        local_deltas_(i, 2) = action[2];
    }

    proposal_dist += local_deltas_;

    // Add Gaussian process noise
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        proposal_dist(i, 0) += normal_dist_(rng_) * MOTION_DISPERSION_X;
        proposal_dist(i, 1) += normal_dist_(rng_) * MOTION_DISPERSION_Y;
        proposal_dist(i, 2) += normal_dist_(rng_) * MOTION_DISPERSION_THETA;
        
        // Normalize angle to [-π, π] range
        while (proposal_dist(i, 2) > M_PI) proposal_dist(i, 2) -= 2.0 * M_PI;
        while (proposal_dist(i, 2) < -M_PI) proposal_dist(i, 2) += 2.0 * M_PI;
    }
}

void ParticleFilter::bicycle_motion_model(Eigen::MatrixXd &proposal_dist, double velocity, double angular_velocity, double dt)
{
    // Bicycle model kinematics for car-like vehicles
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        double x = proposal_dist(i, 0);
        double y = proposal_dist(i, 1);
        double theta = proposal_dist(i, 2);
        
        // Apply bicycle model kinematics
        if (std::abs(angular_velocity) < 1e-6) {
            // Straight line motion
            proposal_dist(i, 0) = x + velocity * dt * std::cos(theta);
            proposal_dist(i, 1) = y + velocity * dt * std::sin(theta);
            proposal_dist(i, 2) = theta; // No change in orientation
        } else {
            // Curved motion with instantaneous center of rotation
            double radius = velocity / angular_velocity;
            double delta_theta = angular_velocity * dt;
            
            // Calculate new position using arc motion
            proposal_dist(i, 0) = x + radius * (std::sin(theta + delta_theta) - std::sin(theta));
            proposal_dist(i, 1) = y - radius * (std::cos(theta + delta_theta) - std::cos(theta));
            proposal_dist(i, 2) = theta + delta_theta;
        }
        
        // Add motion noise
        proposal_dist(i, 0) += normal_dist_(rng_) * MOTION_DISPERSION_X;
        proposal_dist(i, 1) += normal_dist_(rng_) * MOTION_DISPERSION_Y;
        proposal_dist(i, 2) += normal_dist_(rng_) * MOTION_DISPERSION_THETA;
        
        // Normalize angle to [-π, π] range
        while (proposal_dist(i, 2) > M_PI) proposal_dist(i, 2) -= 2.0 * M_PI;
        while (proposal_dist(i, 2) < -M_PI) proposal_dist(i, 2) += 2.0 * M_PI;
    }
}

void ParticleFilter::sensor_model(const Eigen::MatrixXd &proposal_dist, const std::vector<float> &obs,
                                  std::vector<double> &weights)
{
    int num_rays = downsampled_angles_.size();

    // First-time array allocation for ray casting
    if (first_sensor_update_)
    {
        queries_ = Eigen::MatrixXd::Zero(num_rays * MAX_PARTICLES, 3);
        ranges_.resize(num_rays * MAX_PARTICLES);
        tiled_angles_.clear();
        for (int i = 0; i < MAX_PARTICLES; ++i)
        {
            tiled_angles_.insert(tiled_angles_.end(), downsampled_angles_.begin(), downsampled_angles_.end());
        }
        first_sensor_update_ = false;
    }

    // Generate ray queries
    auto query_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        for (int j = 0; j < num_rays; ++j)
        {
            int idx = i * num_rays + j;
            queries_(idx, 0) = proposal_dist(i, 0);
            queries_(idx, 1) = proposal_dist(i, 1);
            queries_(idx, 2) = proposal_dist(i, 2) + downsampled_angles_[j];
        }
    }
    auto query_end = std::chrono::high_resolution_clock::now();
    timing_stats_.query_prep_time += std::chrono::duration<double, std::milli>(query_end - query_start).count();

    // Batch ray casting (timing handled separately in calc_range_many)
    ranges_ = calc_range_many(queries_);

    // Start timing for sensor model evaluation (lookup table part only)
    auto sensor_eval_start = std::chrono::high_resolution_clock::now();

    // Convert to pixel units and compute weights
    std::vector<float> obs_px(obs.size());
    std::vector<float> ranges_px(ranges_.size());

    for (size_t i = 0; i < obs.size(); ++i)
    {
        obs_px[i] = obs[i] / map_resolution_;
        if (obs_px[i] > MAX_RANGE_PX)
            obs_px[i] = MAX_RANGE_PX;
    }

    for (size_t i = 0; i < ranges_.size(); ++i)
    {
        ranges_px[i] = ranges_[i] / map_resolution_;
        if (ranges_px[i] > MAX_RANGE_PX)
            ranges_px[i] = MAX_RANGE_PX;
    }

    // Likelihood calculation using lookup table
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        double weight = 1.0;
        
        for (int j = 0; j < num_rays; ++j)
        {
            int obs_idx = static_cast<int>(std::round(obs_px[j]));
            int range_idx = static_cast<int>(std::round(ranges_px[i * num_rays + j]));

            obs_idx = std::max(0, std::min(obs_idx, MAX_RANGE_PX));
            range_idx = std::max(0, std::min(range_idx, MAX_RANGE_PX));

            weight *= sensor_model_table_(obs_idx, range_idx);
        }
        weights[i] = std::pow(weight, INV_SQUASH_FACTOR);
    }

    auto sensor_eval_end = std::chrono::high_resolution_clock::now();
    timing_stats_.sensor_model_time += std::chrono::duration<double, std::milli>(sensor_eval_end - sensor_eval_start).count();
}

// --------------------------------- RAY CASTING ---------------------------------
std::vector<float> ParticleFilter::calc_range_many(const Eigen::MatrixXd &queries)
{
    auto raycast_start = std::chrono::high_resolution_clock::now();
    
    std::vector<float> results(queries.rows());

    // --------------------------------- PARALLEL PROCESSING ---------------------------------
    if (USE_PARALLEL_RAYCASTING) {
        // Parallel ray casting with OpenMP
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2));
        }
    } else {
        // Sequential ray casting
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2));
        }
    }

    auto raycast_end = std::chrono::high_resolution_clock::now();
    timing_stats_.ray_casting_time += std::chrono::duration<double, std::milli>(raycast_end - raycast_start).count();
    
    return results;
}

float ParticleFilter::cast_ray(double x, double y, double angle)
{
    if (!map_initialized_)
        return MAX_RANGE_METERS;

    double dx = std::cos(angle) * map_resolution_;
    double dy = std::sin(angle) * map_resolution_;

    double current_x = x;
    double current_y = y;

    for (int step = 0; step < MAX_RANGE_PX; ++step)
    {
        current_x += dx;
        current_y += dy;

        // World to grid coordinate transformation
        int grid_x = static_cast<int>((current_x - map_origin_[0]) / map_resolution_);
        int grid_y = static_cast<int>((current_y - map_origin_[1]) / map_resolution_);

        // Map boundary collision
        if (grid_x < 0 || grid_x >= static_cast<int>(map_msg_->info.width) || grid_y < 0 ||
            grid_y >= static_cast<int>(map_msg_->info.height))
        {
            return step * map_resolution_;
        }

        // Obstacle collision detection
        int map_idx = grid_y * map_msg_->info.width + grid_x;
        if (map_idx >= 0 && map_idx < static_cast<int>(map_msg_->data.size()))
        {
            if (map_msg_->data[map_idx] > 50)
            {
                return step * map_resolution_;
            }
        }
    }

    return MAX_RANGE_METERS;
}

void ParticleFilter::MCL(const Eigen::Vector3d &action, const std::vector<float> &observation)
{
    auto mcl_start = std::chrono::high_resolution_clock::now();
    
    // Step 1: Multinomial resampling
    auto resample_start = std::chrono::high_resolution_clock::now();
    std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());
    Eigen::MatrixXd proposal_distribution(MAX_PARTICLES, 3);

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        int idx = particle_dist(rng_);
        proposal_distribution.row(i) = particles_.row(idx);
    }
    auto resample_end = std::chrono::high_resolution_clock::now();
    timing_stats_.resampling_time += std::chrono::duration<double, std::milli>(resample_end - resample_start).count();

    // Step 2: Motion prediction with noise
    auto motion_start = std::chrono::high_resolution_clock::now();
    motion_model(proposal_distribution, action);
    auto motion_end = std::chrono::high_resolution_clock::now();
    timing_stats_.motion_model_time += std::chrono::duration<double, std::milli>(motion_end - motion_start).count();

    // Step 3: Sensor likelihood evaluation (timing handled inside sensor_model function)
    sensor_model(proposal_distribution, observation, weights_);

    // Step 4: Weight normalization
    double sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (sum_weights > 0)
    {
        for (double &w : weights_)
        {
            w /= sum_weights;
        }
    }

    // Step 5: Update particle set
    particles_ = proposal_distribution;
    
    auto mcl_end = std::chrono::high_resolution_clock::now();
    timing_stats_.total_mcl_time += std::chrono::duration<double, std::milli>(mcl_end - mcl_start).count();
    timing_stats_.measurement_count++;
}

Eigen::Vector3d ParticleFilter::expected_pose()
{
    Eigen::Vector3d pose = Eigen::Vector3d::Zero();
    double sum_sin = 0.0, sum_cos = 0.0;
    
    // Calculate weighted mean for x, y coordinates
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        pose[0] += weights_[i] * particles_(i, 0);  // x
        pose[1] += weights_[i] * particles_(i, 1);  // y
        
        // For circular mean of angles
        sum_sin += weights_[i] * std::sin(particles_(i, 2));
        sum_cos += weights_[i] * std::cos(particles_(i, 2));
    }
    
    // Calculate circular mean for angle
    pose[2] = std::atan2(sum_sin, sum_cos);
    
    return pose;
}

// --------------------------------- MAIN UPDATE LOOP ---------------------------------
void ParticleFilter::update()
{
    if (!lidar_initialized_ || !odom_initialized_ || !map_initialized_)
    {
        return;
    }

    if (state_lock_.try_lock())
    {
        ++iters_;

        auto observation = downsampled_ranges_;
        auto action = odometry_data_;
        odometry_data_ = Eigen::Vector3d::Zero();

        // Execute complete MCL cycle
        MCL(action, observation);

        // Final pose estimate: weighted mean
        inferred_pose_ = expected_pose();
        
        // If using odometry tracking, update reference pose for correction
        // Enable when: 1) pose initialized from RViz, OR 2) global initialization done with valid MCL estimate
        bool can_use_odom_tracking = pose_initialized_from_rviz_ || 
                                   (map_initialized_ && iters_ > 0 && is_pose_valid(inferred_pose_));
        
        if (can_use_odom_tracking) {
            // Initialize odometry tracking if not already done but conditions are met
            if (!odom_tracking_active_ && is_pose_valid(inferred_pose_)) {
                initialize_odom_tracking(inferred_pose_, false);
                RCLCPP_INFO(this->get_logger(), "Odometry tracking initialized from GLOBAL initialization (not RViz)");
            }
            
            // Calculate correction between MCL estimate and odometry estimate
            Eigen::Vector3d correction = inferred_pose_ - odom_pose_;
            
            // Apply correction to odometry tracking - no coordinate conversion
            odom_reference_pose_ = inferred_pose_;
            odom_reference_odom_ = last_pose_;
            odom_pose_ = inferred_pose_;
            
            if (iters_ % 50 == 0) {
                const char* init_source = pose_initialized_from_rviz_ ? "RViz" : "Global";
                RCLCPP_INFO(this->get_logger(), "MCL correction [%s]: [%.3f, %.3f, %.3f]", 
                           init_source, correction[0], correction[1], correction[2]);
            }
        }

        state_lock_.unlock();

        // Output to navigation stack and visualization (using current best pose)
        Eigen::Vector3d pose_to_publish = get_current_pose();
        publish_tf(pose_to_publish, last_stamp_);

        if (iters_ % 50 == 0)
        {
            RCLCPP_INFO(this->get_logger(), "MCL iter %d: (%.3f, %.3f, %.3f)", iters_, pose_to_publish[0],
                        pose_to_publish[1], pose_to_publish[2]);
        }
        
        if (iters_ % 200 == 0)
        {
            print_performance_stats();
        }

        visualize();
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Concurrency error avoided");
    }
}

// --------------------------------- CONFIGURABLE TIMER UPDATE ---------------------------------
void ParticleFilter::timer_update()
{
    static int timer_call_count = 0;
    timer_call_count++;
    
    // Log first few calls and then every 100 calls
    if (timer_call_count <= 5 || timer_call_count % 100 == 0) {
        RCLCPP_INFO(this->get_logger(), "Timer update #%d - map_init: %s, odom_init: %s, lidar_init: %s", 
                    timer_call_count, 
                    map_initialized_ ? "true" : "false",
                    odom_initialized_ ? "true" : "false", 
                    lidar_initialized_ ? "true" : "false");
    }
    
    // Check if all required data is available
    if (!map_initialized_) {
        return;
    }
    
    // Allow timer to run even without odom for initial convergence
    bool has_odometry_for_motion = odom_initialized_;
    if (!has_odometry_for_motion) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                            "Running MCL without odometry - sensor-only updates for initial convergence");
    }
    
    rclcpp::Time current_time = this->get_clock()->now();
    
    // Handle simulation time that starts at 0 - use steady clock instead
    static auto steady_start_time = std::chrono::steady_clock::now();
    static auto last_steady_time = steady_start_time;
    
    auto current_steady_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_steady_time - last_steady_time).count();
    
    // Initialize time on first call
    static bool timer_initialized = false;
    if (!timer_initialized) {
        timer_initialized = true;
        last_steady_time = current_steady_time;
        return;
    }
    
    // Skip if time step is too large (allow very small dt for debugging)
    if (dt > 1.0) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                            "Skipping timer update - dt: %.6f", dt);
        return;
    }
    
    // For very small dt, don't apply motion model but allow sensor update
    bool apply_motion = (dt >= 0.001);
    
    static int lock_attempts = 0;
    static int successful_locks = 0;
    lock_attempts++;
    
    if (state_lock_.try_lock()) {
        successful_locks++;
        
        if (lock_attempts % 1000 == 0) {  // Every ~5 seconds
            RCLCPP_INFO(this->get_logger(), "Lock stats: %d/%d successful, ranges size: %zu", 
                        successful_locks, lock_attempts, downsampled_ranges_.size());
        }
        
        // Apply motion model only if odometry is available and there's motion
        if (has_odometry_for_motion && apply_motion && (std::abs(current_velocity_) > 0.005 || std::abs(current_angular_vel_) > 0.005)) {
            bicycle_motion_model(particles_, current_velocity_, current_angular_vel_, dt);
        }
        
        // Apply sensor update at higher frequency for better tracking
        static int sensor_update_counter = 0;
        sensor_update_counter++;
        int sensor_update_divisor = 2;  // Update sensor every 2nd timer call (200Hz -> 100Hz)
        
        if (lidar_initialized_ && !downsampled_ranges_.empty() && (sensor_update_counter % sensor_update_divisor == 0)) {
            ++iters_;
            
            auto observation = downsampled_ranges_;
            
            // Execute sensor model and resampling
            sensor_model(particles_, observation, weights_);
            
            // Weight normalization
            double sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
            if (sum_weights > 0) {
                for (double &w : weights_) {
                    w /= sum_weights;
                }
            }
            
            // Add adaptive small random motion to particles during early global initialization
            // to help with convergence when there's no odometry
            if (!pose_initialized_from_rviz_ && !has_odometry_for_motion && iters_ < 15) {
                // Reduce noise as iterations progress to prevent excessive spreading
                double noise_factor = std::max(0.1, 1.0 - (static_cast<double>(iters_) / 15.0));
                double pos_noise = 0.05 * noise_factor;  // Reduced from 0.1 to 0.05
                double angle_noise = 0.02 * noise_factor;  // Reduced from 0.05 to 0.02
                
                for (int i = 0; i < MAX_PARTICLES; ++i) {
                    particles_(i, 0) += normal_dist_(rng_) * pos_noise;
                    particles_(i, 1) += normal_dist_(rng_) * pos_noise;
                    particles_(i, 2) += normal_dist_(rng_) * angle_noise;
                    
                    // Normalize angle to [-π, π] range
                    while (particles_(i, 2) > M_PI) particles_(i, 2) -= 2.0 * M_PI;
                    while (particles_(i, 2) < -M_PI) particles_(i, 2) += 2.0 * M_PI;
                }
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000, 
                                    "Adding adaptive random motion for convergence (iter: %d, noise: %.3f)", 
                                    iters_, noise_factor);
            }
            
            // Effective sample size based resampling (more efficient)
            double sum_weights_sq = 0.0;
            for (const auto& w : weights_) {
                sum_weights_sq += w * w;
            }
            double n_eff = 1.0 / sum_weights_sq;
            // Adaptive resampling threshold based on convergence state
            double resample_threshold;
            if (pose_initialized_from_rviz_) {
                resample_threshold = MAX_PARTICLES * 0.25;  // More frequent resampling when initialized from RViz
            } else {
                // During global initialization, allow more iterations before resampling
                resample_threshold = MAX_PARTICLES * 0.1;   // Less frequent resampling for better convergence
            }
            
            if (n_eff < resample_threshold) {
                // Multinomial resampling
                std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());
                Eigen::MatrixXd resampled_particles(MAX_PARTICLES, 3);
                
                for (int i = 0; i < MAX_PARTICLES; ++i) {
                    int idx = particle_dist(rng_);
                    resampled_particles.row(i) = particles_.row(idx);
                }
                
                particles_ = resampled_particles;
                std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);
                
                const char* init_state = pose_initialized_from_rviz_ ? "RViz-init" : "Global-init";
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
                                    "Resampled particles [%s] - Neff: %.1f (threshold: %.1f)", 
                                    init_state, n_eff, resample_threshold);
            }
            
            // Update pose estimate
            inferred_pose_ = expected_pose();
            
            // Update odometry tracking reference if active and odometry is available
            bool can_use_odom_tracking = has_odometry_for_motion && (pose_initialized_from_rviz_ || 
                                       (map_initialized_ && iters_ > 0 && is_pose_valid(inferred_pose_)));
            
            if (can_use_odom_tracking) {
                if (!odom_tracking_active_ && is_pose_valid(inferred_pose_)) {
                    initialize_odom_tracking(inferred_pose_, false);
                    RCLCPP_INFO(this->get_logger(), "Odometry tracking initialized from MCL estimate");
                }
                
                Eigen::Vector3d correction = inferred_pose_ - odom_pose_;
                odom_reference_pose_ = inferred_pose_;
                odom_reference_odom_ = last_pose_;
                odom_pose_ = inferred_pose_;
                
                if (iters_ % 50 == 0) {
                    const char* init_source = pose_initialized_from_rviz_ ? "RViz" : "Global";
                    RCLCPP_INFO(this->get_logger(), "MCL correction [%s]: [%.3f, %.3f, %.3f]", 
                               init_source, correction[0], correction[1], correction[2]);
                }
            }
            
            if (iters_ % 50 == 0) {
                RCLCPP_INFO(this->get_logger(), "MCL iter %d: (%.3f, %.3f, %.3f)", iters_, 
                           inferred_pose_[0], inferred_pose_[1], inferred_pose_[2]);
            }
            
            if (iters_ % 200 == 0) {
                print_performance_stats();
            }
            
            visualize();
            
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                                "MCL update completed - iter: %d", iters_);
        }
        
        state_lock_.unlock();
    }
    
    // Always update steady time for next calculation
    last_steady_time = current_steady_time;
    
    // Always publish TF and odom at timer frequency if we have a valid pose
    if (map_initialized_) {
        Eigen::Vector3d current_pose = get_current_pose();
        rclcpp::Time timestamp = current_time;
        
        // Use odometry timestamp if available, otherwise use current time
        if (has_odometry_for_motion && last_stamp_.nanoseconds() != 0) {
            timestamp = last_stamp_;
        }
        
        // Always publish, even if pose is at origin during initialization
        publish_tf(current_pose, timestamp);
        
        // Debug logging for pose publishing
        static int publish_count = 0;
        publish_count++;
        if (publish_count <= 10 || publish_count % 100 == 0) {
            RCLCPP_INFO(this->get_logger(), "Publishing pose #%d: [%.3f, %.3f, %.3f] (odom_active: %s)", 
                        publish_count, current_pose[0], current_pose[1], current_pose[2],
                        odom_tracking_active_ ? "true" : "false");
        }
    }
}

void ParticleFilter::publish_map_periodically()
{
    // Periodically publish map to maintain persistent display in RViz
    if (map_initialized_ && map_pub_ && map_msg_) {
        map_pub_->publish(*map_msg_);
    }
}

// --------------------------------- OUTPUT & VISUALIZATION ---------------------------------
void ParticleFilter::publish_tf(const Eigen::Vector3d &pose, const rclcpp::Time &stamp)
{
    // Apply offset in vehicle coordinate frame (forward direction)
    double forward_offset = LIDAR_OFFSET_X;  // Use configured lidar offset
    double cos_theta = std::cos(pose[2]);
    double sin_theta = std::sin(pose[2]);
    
    // Transform to base_link position (lidar position - forward offset)
    double base_link_x = pose[0] - forward_offset * cos_theta;
    double base_link_y = pose[1] - forward_offset * sin_theta;
    
    // Publish map → base_link transform
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();
    t.header.frame_id = "map";
    t.child_frame_id = "base_link";
    t.transform.translation.x = base_link_x;
    t.transform.translation.y = base_link_y;
    t.transform.translation.z = 0.0;
    t.transform.rotation = angle_to_quaternion(pose[2]);

    pub_tf_->sendTransform(t);

    // Optional odometry output
    if (PUBLISH_ODOM && odom_pub_)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = base_link_x;
        odom.pose.pose.position.y = base_link_y;
        odom.pose.pose.orientation = angle_to_quaternion(pose[2]);
        odom.twist.twist.linear.x = current_speed_;
        odom_pub_->publish(odom);
    }
}


Eigen::Vector3d ParticleFilter::get_current_pose()
{
    // Priority 1: Use odometry-based tracking if active and valid
    if (odom_tracking_active_ && is_pose_valid(odom_pose_))
        return odom_pose_;
    
    // Priority 2: Use particle filter estimate if valid
    if (is_pose_valid(inferred_pose_))
        return inferred_pose_;
    
    // Priority 3: During initialization without pose estimate, use center of particles
    if (map_initialized_ && particles_.rows() > 0) {
        Eigen::Vector3d particle_center = particles_.colwise().mean();
        if (is_pose_valid(particle_center)) {
            return particle_center;
        }
    }
    
    // Priority 4: Fallback to last known good pose
    if (is_pose_valid(last_pose_))
        return last_pose_;
    
    // Default to origin
    return Eigen::Vector3d::Zero();
}

bool ParticleFilter::is_pose_valid(const Eigen::Vector3d& pose)
{
    return utils::validation::is_pose_valid(pose, MAX_POSE_RANGE);
}

void ParticleFilter::visualize()
{
    if (!DO_VIZ)
        return;

    // RViz pose visualization (with vehicle frame offset)
    if (pose_pub_ && pose_pub_->get_subscription_count() > 0)
    {
        // Apply same vehicle frame offset as TF
        double forward_offset = LIDAR_OFFSET_X;
        double cos_theta = std::cos(inferred_pose_[2]);
        double sin_theta = std::sin(inferred_pose_[2]);
        
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = this->get_clock()->now();
        ps.header.frame_id = "map";
        ps.pose.position.x = inferred_pose_[0] - forward_offset * cos_theta;
        ps.pose.position.y = inferred_pose_[1] - forward_offset * sin_theta;
        ps.pose.orientation = angle_to_quaternion(inferred_pose_[2]);
        pose_pub_->publish(ps);
    }

    // RViz particle cloud (downsampled for performance)
    if (particle_pub_ && particle_pub_->get_subscription_count() > 0)
    {
        if (MAX_PARTICLES > MAX_VIZ_PARTICLES)
        {
            // Weighted downsampling
            std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());
            Eigen::MatrixXd viz_particles(MAX_VIZ_PARTICLES, 3);

            for (int i = 0; i < MAX_VIZ_PARTICLES; ++i)
            {
                int idx = particle_dist(rng_);
                viz_particles.row(i) = particles_.row(idx);
            }

            publish_particles(viz_particles);
        }
        else
        {
            publish_particles(particles_);
        }
    }
}

void ParticleFilter::publish_particles(const Eigen::MatrixXd &particles_to_pub)
{
    // Apply vehicle frame offset to all particles
    Eigen::MatrixXd offset_particles = particles_to_pub;
    double forward_offset = LIDAR_OFFSET_X;
    
    for (int i = 0; i < offset_particles.rows(); ++i) {
        double cos_theta = std::cos(offset_particles(i, 2));
        double sin_theta = std::sin(offset_particles(i, 2));
        offset_particles(i, 0) -= forward_offset * cos_theta;
        offset_particles(i, 1) -= forward_offset * sin_theta;
    }
    
    auto pa = utils::particles_to_pose_array(offset_particles);
    pa.header.stamp = this->get_clock()->now();
    pa.header.frame_id = "map";
    particle_pub_->publish(pa);
}

// --------------------------------- UTILITY FUNCTIONS ---------------------------------
double ParticleFilter::quaternion_to_angle(const geometry_msgs::msg::Quaternion &q)
{
    return utils::geometry::quaternion_to_yaw(q);
}

geometry_msgs::msg::Quaternion ParticleFilter::angle_to_quaternion(double angle)
{
    return utils::geometry::yaw_to_quaternion(angle);
}

Eigen::Matrix2d ParticleFilter::rotation_matrix(double angle)
{
    return utils::geometry::rotation_matrix(angle);
}

// --------------------------------- PERFORMANCE PROFILING ---------------------------------
void ParticleFilter::print_performance_stats()
{
    // Create a lambda that captures the logger
    auto logger_func = [this](const std::string& msg) {
        RCLCPP_INFO(this->get_logger(), "%s", msg.c_str());
    };
    
    // Print performance stats using utils
    timing_stats_.print_stats(logger_func);
    
    // Print additional particle filter specific info
    if (timing_stats_.measurement_count > 0) {
        RCLCPP_INFO(this->get_logger(), 
            "Particles: %d, Rays/particle: %zu, Total rays: %d", 
            MAX_PARTICLES, downsampled_angles_.size(), MAX_PARTICLES * static_cast<int>(downsampled_angles_.size()));
        RCLCPP_INFO(this->get_logger(), "=====================================");
    }
    
    reset_performance_stats();
}

void ParticleFilter::reset_performance_stats()
{
    timing_stats_.reset();
}

// --------------------------------- ODOMETRY-BASED TRACKING IMPLEMENTATION ---------------------------------
void ParticleFilter::initialize_odom_tracking(const Eigen::Vector3d& initial_pose, bool from_rviz)
{
    RCLCPP_INFO(this->get_logger(), "Initializing odometry tracking from pose: [%.3f, %.3f, %.3f]", 
                initial_pose[0], initial_pose[1], initial_pose[2]);
    
    // Use initial pose directly - no coordinate conversion
    odom_pose_ = initial_pose;
    odom_reference_pose_ = initial_pose;
    
    // Store current odometry reading as reference
    if (last_pose_.norm() > 0) {
        odom_reference_odom_ = last_pose_;
    }
    
    pose_initialized_from_rviz_ = from_rviz;
    odom_tracking_active_ = true;
    RCLCPP_INFO(this->get_logger(), "Odometry tracking initialized successfully");
}

void ParticleFilter::update_odom_pose(const nav_msgs::msg::Odometry::SharedPtr& msg)
{
    if (!odom_tracking_active_) return;
    
    // Current odometry reading
    Eigen::Vector3d current_odom(msg->pose.pose.position.x, msg->pose.pose.position.y,
                                 quaternion_to_angle(msg->pose.pose.orientation));
    
    // Calculate odometry displacement since reference
    Eigen::Vector3d odom_delta = current_odom - odom_reference_odom_;
    
    // Apply displacement directly - no coordinate conversion
    odom_pose_ = odom_reference_pose_ + odom_delta;
}

Eigen::Vector3d ParticleFilter::odom_to_rear_axle(const Eigen::Vector3d& odom_pose)
{
    return utils::coordinates::odom_to_rear_axle(odom_pose, WHEELBASE);
}

Eigen::Vector3d ParticleFilter::rear_axle_to_odom(const Eigen::Vector3d& rear_axle_pose)
{
    return utils::coordinates::rear_axle_to_odom(rear_axle_pose, WHEELBASE);
}

} // namespace particle_filter_cpp

// --------------------------------- PROGRAM ENTRY POINT ---------------------------------
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<particle_filter_cpp::ParticleFilter>());
    rclcpp::shutdown();
    return 0;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(particle_filter_cpp::ParticleFilter)
