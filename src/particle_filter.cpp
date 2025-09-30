// ================================================================================================
// PARTICLE FILTER IMPLEMENTATION - Monte Carlo Localization (MCL)
// ================================================================================================
// Features: Multinomial resampling, velocity motion model, beam sensor model, ray casting
// ================================================================================================

#include "particle_filter_cpp/particle_filter.hpp"
#include "particle_filter_cpp/utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

namespace particle_filter_cpp
{

// ================================================================================================
// CONSTRUCTOR & INITIALIZATION
// ================================================================================================

/**
 * @brief Initializes particle filter with parameters, publishers, and subscribers
 */
ParticleFilter::ParticleFilter(const rclcpp::NodeOptions &options)
    : Node("particle_filter", options), rng_(std::random_device{}()), normal_dist_(0.0, 1.0)
{
    // === PARAMETER DECLARATIONS ===
    // Core algorithm parameters
    this->declare_parameter("angle_step", 18);
    this->declare_parameter("max_particles", 4000);
    this->declare_parameter("max_viz_particles", 60);
    this->declare_parameter("squash_factor", 2.2);
    this->declare_parameter("max_range", 12.0);
    this->declare_parameter("max_pose_range", 10000.0);
    
    // Sensor model parameters
    this->declare_parameter("z_short", 0.01);
    this->declare_parameter("z_max", 0.07);
    this->declare_parameter("z_rand", 0.12);
    this->declare_parameter("z_hit", 0.80);
    this->declare_parameter("sigma_hit", 8.0);
    
    // Motion model parameters
    this->declare_parameter("motion_dispersion_x", 0.02);
    this->declare_parameter("motion_dispersion_y", 0.01);
    this->declare_parameter("motion_dispersion_theta", 0.05);

    // Robot geometry
    this->declare_parameter("wheelbase", 0.324);

    
    // ROS interface
    this->declare_parameter("scan_topic", "/scan");
    this->declare_parameter("odom_topic", "/odom");
    this->declare_parameter("publish_odom", true);
    this->declare_parameter("viz", true);
    this->declare_parameter("timer_frequency", 35.0);
    
    // Performance
    this->declare_parameter("use_parallel_raycasting", true);
    this->declare_parameter("num_threads", 0); // 0 = auto-detect

    // TF frames
    this->declare_parameter("map_frame", "map");
    this->declare_parameter("odom_frame", "odom");
    this->declare_parameter("base_frame", "base_link"); 
    this->declare_parameter("laser_frame", "laser");
    
    // TF publishing control
    this->declare_parameter("publish_map_odom_tf", true);
    this->declare_parameter("publish_odom_base_tf", true);

    // === PARAMETER RETRIEVAL ===
    // Core algorithm parameters
    ANGLE_STEP = this->get_parameter("angle_step").as_int();
    MAX_PARTICLES = this->get_parameter("max_particles").as_int();
    MAX_VIZ_PARTICLES = this->get_parameter("max_viz_particles").as_int();
    INV_SQUASH_FACTOR = 1.0 / this->get_parameter("squash_factor").as_double();
    MAX_RANGE_METERS = this->get_parameter("max_range").as_double();
    MAX_POSE_RANGE = this->get_parameter("max_pose_range").as_double();

    // Sensor model parameters
    Z_SHORT = this->get_parameter("z_short").as_double();
    Z_MAX = this->get_parameter("z_max").as_double();
    Z_RAND = this->get_parameter("z_rand").as_double();
    Z_HIT = this->get_parameter("z_hit").as_double();
    SIGMA_HIT = this->get_parameter("sigma_hit").as_double();

    // Motion model parameters
    MOTION_DISPERSION_X = this->get_parameter("motion_dispersion_x").as_double();
    MOTION_DISPERSION_Y = this->get_parameter("motion_dispersion_y").as_double();
    MOTION_DISPERSION_THETA = this->get_parameter("motion_dispersion_theta").as_double();

    // Robot geometry
    WHEELBASE = this->get_parameter("wheelbase").as_double();

    // ROS interface
    PUBLISH_ODOM = this->get_parameter("publish_odom").as_bool();
    DO_VIZ = this->get_parameter("viz").as_bool();
    TIMER_FREQUENCY = this->get_parameter("timer_frequency").as_double();
    RCLCPP_INFO(this->get_logger(), "Loaded timer_frequency: %.1f Hz", TIMER_FREQUENCY);

    // Performance
    USE_PARALLEL_RAYCASTING = this->get_parameter("use_parallel_raycasting").as_bool();
    NUM_THREADS = this->get_parameter("num_threads").as_int();

    // TF frames
    MAP_FRAME = this->get_parameter("map_frame").as_string();
    ODOM_FRAME = this->get_parameter("odom_frame").as_string();
    BASE_FRAME = this->get_parameter("base_frame").as_string();
    LASER_FRAME = this->get_parameter("laser_frame").as_string();

    // TF publishing control
    PUBLISH_MAP_ODOM_TF = this->get_parameter("publish_map_odom_tf").as_bool();
    PUBLISH_ODOM_BASE_TF = this->get_parameter("publish_odom_base_tf").as_bool();

    // State initialization
    MAX_RANGE_PX_ = 0;
    map_initialized_ = false;
    lidar_initialized_ = false;
    odom_initialized_ = false;
    first_sensor_update_ = true;
    current_velocity_ = 0.0;
    current_angular_vel_ = 0.0;
    has_new_lidar_data_ = false;
    last_lidar_time_ = rclcpp::Time(0);
    mcl_processing_time_ = 0.0;
    
    // Simple state tracking

    // Threading setup - no startup throttling like old working version
    if (USE_PARALLEL_RAYCASTING) {
        if (NUM_THREADS == 0) {
            NUM_THREADS = omp_get_max_threads();
        }
        omp_set_num_threads(NUM_THREADS);  // Use full thread count immediately
    }

    // Particle initialization
    particles_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);
    weights_.resize(MAX_PARTICLES, 1.0 / MAX_PARTICLES);

    // Motion cache and performance optimizations
    local_deltas_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);
    proposal_distribution_ = Eigen::MatrixXd::Zero(MAX_PARTICLES, 3);

    // Publishers
    if (DO_VIZ)
    {
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 1);
        particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
    }

    if (PUBLISH_ODOM)
    {
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", 1);
    }

    // Map publisher
    map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/map", rclcpp::QoS(1).transient_local());

    // TF broadcaster and listener
    pub_tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    // Subscribers
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

    // Map service client
    map_client_ = this->create_client<nav_msgs::srv::GetMap>("/map_server/map");

    // Load map
    get_omap();
    initialize_global();

    // Update timer - simple like old working version
    int timer_interval_ms = static_cast<int>(1000.0 / TIMER_FREQUENCY);
    update_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(timer_interval_ms),
        std::bind(&ParticleFilter::timer_update, this)
    );

    // Map publisher timer
    map_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(200),
        std::bind(&ParticleFilter::publish_map_periodically, this)
    );


    RCLCPP_INFO(this->get_logger(), "Particle filter initialized - %.1fHz, %s threading (%d threads)",
        TIMER_FREQUENCY, USE_PARALLEL_RAYCASTING ? "parallel" : "sequential",
        USE_PARALLEL_RAYCASTING ? NUM_THREADS : 1);
}

// ================================================================================================
// MAP LOADING & PREPROCESSING
// ================================================================================================
/**
 * @brief Loads occupancy grid map from map server and extracts free space
 */
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

        MAX_RANGE_PX_ = static_cast<int>(MAX_RANGE_METERS / map_msg_->info.resolution);

        map_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "Map loaded and published");

        // Publish map
        if (map_pub_) {
            map_pub_->publish(*map_msg_);
        }

        // Generate sensor model lookup table
        precompute_sensor_model();
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get map from map server");
    }
}

// ================================================================================================
// SENSOR MODEL PRECOMPUTATION
// ================================================================================================
/**
 * @brief Precomputes sensor model lookup table for fast likelihood evaluation
 */
void ParticleFilter::precompute_sensor_model()
{
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(map_lock_);
        local_map = map_msg_;
    }

    if (!local_map || local_map->info.resolution <= 0.0)
    {
        RCLCPP_ERROR(this->get_logger(), "Invalid map resolution: %.6f",
                     local_map ? local_map->info.resolution : 0.0);
        return;
    }

    int table_width = MAX_RANGE_PX_ + 1;
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
            if (r == MAX_RANGE_PX_)
            {
                prob += Z_MAX;
            }

            // Z_RAND: Uniform distribution
            if (r < MAX_RANGE_PX_)
            {
                prob += Z_RAND * 1.0 / static_cast<double>(MAX_RANGE_PX_);
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
    RCLCPP_INFO(this->get_logger(), "Sensor model ready (%ld ms)", duration.count());
}

// ================================================================================================
// SENSOR CALLBACKS
// ================================================================================================
/**
 * @brief Processes lidar scan data and downsamples for particle filter
 */
void ParticleFilter::lidarCB(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    if (downsampled_angles_.empty())
    {
        // Create downsampled angles directly from scan parameters
        for (size_t i = 0; i < msg->ranges.size(); i += ANGLE_STEP)
        {
            downsampled_angles_.push_back(msg->angle_min + i * msg->angle_increment);
        }

        RCLCPP_INFO(this->get_logger(), "LiDAR initialized - %zu angles", downsampled_angles_.size());
    }

    // Extract downsampled measurements and mark new lidar data - protected by mutex
    {
        std::lock_guard<std::mutex> lock(lidar_lock_);
        downsampled_ranges_.clear();
        for (size_t i = 0; i < msg->ranges.size(); i += ANGLE_STEP)
        {
            downsampled_ranges_.push_back(msg->ranges[i]);
        }

        last_lidar_time_ = msg->header.stamp;
        has_new_lidar_data_ = true;

        // RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        //     "LiDAR callback: new data received, timestamp: %d.%09u",
        //     msg->header.stamp.sec, msg->header.stamp.nanosec);

        lidar_initialized_ = true;
    }

    // // Trigger MCL update based on update_from parameter
    //     update();
    // }
}

/**
 * @brief Processes odometry data and triggers MCL update
 */
void ParticleFilter::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Protect shared velocity variables with mutex
    {
        std::lock_guard<std::mutex> lock(odom_lock_);

        // Store velocity information for publishing
        current_velocity_ = msg->twist.twist.linear.x;
        current_angular_vel_ = msg->twist.twist.angular.z;

        // Mark odometry as initialized
        if (!odom_initialized_) {
            RCLCPP_INFO(this->get_logger(), "Odometry initialized");
            odom_initialized_ = true;
        }
    }
}

// ================================================================================================
// INTERACTIVE INITIALIZATION
// ================================================================================================
/**
 * @brief Initializes particles around manually clicked pose in RViz
 */
void ParticleFilter::clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    Eigen::Vector3d base_pose(msg->pose.pose.position.x, msg->pose.pose.position.y,
                              utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation));

    // Convert base_link pose to laser frame for particles initialization
    Eigen::Vector3d laser_pose;
    try {
        // Create pose in base_link frame
        geometry_msgs::msg::PoseStamped base_pose_msg;
        base_pose_msg.header.frame_id = BASE_FRAME;
        base_pose_msg.pose.position.x = base_pose[0];
        base_pose_msg.pose.position.y = base_pose[1];
        base_pose_msg.pose.position.z = 0.0;
        base_pose_msg.pose.orientation = utils::geometry::yaw_to_quaternion(base_pose[2]);

        // Transform to laser frame
        geometry_msgs::msg::PoseStamped laser_pose_msg;
        tf_buffer_->transform(base_pose_msg, laser_pose_msg, LASER_FRAME);

        laser_pose = Eigen::Vector3d(
            laser_pose_msg.pose.position.x,
            laser_pose_msg.pose.position.y,
            utils::geometry::quaternion_to_yaw(laser_pose_msg.pose.orientation)
        );
    }
    catch (tf2::TransformException &ex) {
        // Fallback: manual conversion using default offset
        RCLCPP_WARN(this->get_logger(), "TF transform failed, using manual conversion: %s", ex.what());
        double offset_x = 0.28;
        double cos_theta = std::cos(base_pose[2]);
        double sin_theta = std::sin(base_pose[2]);

        laser_pose = Eigen::Vector3d(
            base_pose[0] + offset_x * cos_theta,
            base_pose[1] + offset_x * sin_theta,
            base_pose[2]
        );
    }

    // Initialize particle filter around clicked pose (in laser frame)
    initialize_particles_pose(laser_pose);

    RCLCPP_INFO(this->get_logger(), "Pose initialized from RViz at base_link [%.3f, %.3f, %.3f] -> laser [%.3f, %.3f, %.3f]",
                base_pose[0], base_pose[1], base_pose[2],
                laser_pose[0], laser_pose[1], laser_pose[2]);

    // Trigger immediate visualization update (use base_link frame for visualization)
    visualize(base_pose, this->get_clock()->now());
}

/**
 * @brief Triggers global particle initialization when point is clicked
 */
void ParticleFilter::clicked_point(const geometry_msgs::msg::PointStamped::SharedPtr /*msg*/)
{
    initialize_global();
}

// ================================================================================================
// PARTICLE INITIALIZATION
// ================================================================================================
/**
 * @brief Initializes particles around specified pose with Gaussian distribution
 */
void ParticleFilter::initialize_particles_pose(const Eigen::Vector3d &pose)
{
    RCLCPP_INFO(this->get_logger(), "Initializing particles at [%.3f, %.3f, %.3f]", 
                pose[0], pose[1], pose[2]);

    std::lock_guard<std::mutex> lock(state_lock_);
    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

    // Use tighter distribution for faster convergence after manual pose setting
    double pos_std = 0.1;   // Reduced from 0.5m to 0.1m (±10cm)
    double angle_std = 0.1; // Reduced from 0.4rad to 0.1rad (±5.7°)

    // Generate all noise values at once
    std::vector<double> noise_x_values(MAX_PARTICLES);
    std::vector<double> noise_y_values(MAX_PARTICLES);
    std::vector<double> noise_theta_values(MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(rng_lock_);
        for (int i = 0; i < MAX_PARTICLES; ++i)
        {
            noise_x_values[i] = normal_dist_(rng_) * pos_std;
            noise_y_values[i] = normal_dist_(rng_) * pos_std;
            noise_theta_values[i] = normal_dist_(rng_) * angle_std;
        }
    }

    // Apply noise to particles without lock
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        particles_(i, 0) = pose[0] + noise_x_values[i];
        particles_(i, 1) = pose[1] + noise_y_values[i];
        particles_(i, 2) = pose[2] + noise_theta_values[i];

        // Normalize angle
        particles_(i, 2) = utils::geometry::normalize_angle(particles_(i, 2));
    }
}

/**
 * @brief Initializes particles uniformly across all free space in map
 */
void ParticleFilter::initialize_global()
{
    if (!map_initialized_)
        return;

    RCLCPP_INFO(this->get_logger(), "Global initialization started");

    // 1. Copy map data once with minimal lock time
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(map_lock_);
        local_map = map_msg_;
    }

    if (!local_map) return;

    // 2. Extract free space cells without lock (read-only operation)
    int height = local_map->info.height;
    int width = local_map->info.width;
    double resolution = local_map->info.resolution;
    double origin_x = local_map->info.origin.position.x;
    double origin_y = local_map->info.origin.position.y;

    std::vector<std::pair<int, int>> permissible_positions;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int idx = i * width + j;
            if (idx < static_cast<int>(local_map->data.size()) && local_map->data[idx] == 0)
            {
                permissible_positions.emplace_back(i, j);
            }
        }
    }

    if (permissible_positions.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "No free space found in map!");
        return;
    }

    // 3. Generate particles without lock (using local data)
    std::uniform_int_distribution<int> pos_dist(0, permissible_positions.size() - 1);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    // 4. Generate random values at once
    std::vector<int> indices(MAX_PARTICLES);
    std::vector<double> angles(MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(rng_lock_);
        for (int i = 0; i < MAX_PARTICLES; ++i)
        {
            indices[i] = pos_dist(rng_);
            angles[i] = angle_dist(rng_);
        }
    }

    // 5. Update particle state with minimal lock time
    {
        std::lock_guard<std::mutex> lock(state_lock_);

        for (int i = 0; i < MAX_PARTICLES; ++i)
        {
            auto pos = permissible_positions[indices[i]];
            particles_(i, 0) = pos.second * resolution + origin_x;
            particles_(i, 1) = pos.first * resolution + origin_y;
            particles_(i, 2) = angles[i];
        }

        std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

        // Calculate expected pose from initialized particles
        Eigen::Vector3d initial_pose = expected_pose();

        RCLCPP_INFO(this->get_logger(), "Initialized %d particles globally at [%.3f, %.3f, %.3f]",
                    MAX_PARTICLES, initial_pose[0], initial_pose[1], initial_pose[2]);
    }
}

// ================================================================================================
// MCL ALGORITHM CORE
// ================================================================================================

/**
 * @brief Applies motion model to particles with Gaussian noise
 */
void ParticleFilter::motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action)
{
    // Vectorized motion model implementation
    // Transform the action into the coordinate space of each particle

    // Pre-compute trigonometric values for all particles (vectorized approach)
    Eigen::VectorXd cos_thetas = proposal_dist.col(2).array().cos();
    Eigen::VectorXd sin_thetas = proposal_dist.col(2).array().sin();

    // Apply motion transformation: local → global coordinates (vectorized)
    Eigen::VectorXd global_dx = cos_thetas * action[0] - sin_thetas * action[1];
    Eigen::VectorXd global_dy = sin_thetas * action[0] + cos_thetas * action[1];

    // Apply displacement
    proposal_dist.col(0) += global_dx;
    proposal_dist.col(1) += global_dy;
    proposal_dist.col(2).array() += action[2];

    // Add Gaussian noise - generate all noise values at once
    std::vector<double> noise_x_values(MAX_PARTICLES);
    std::vector<double> noise_y_values(MAX_PARTICLES);
    std::vector<double> noise_theta_values(MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(rng_lock_);
        for (int i = 0; i < MAX_PARTICLES; ++i)
        {
            noise_x_values[i] = normal_dist_(rng_) * MOTION_DISPERSION_X;
            noise_y_values[i] = normal_dist_(rng_) * MOTION_DISPERSION_Y;
            noise_theta_values[i] = normal_dist_(rng_) * MOTION_DISPERSION_THETA;
        }
    }

    // Apply noise without lock
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        proposal_dist(i, 0) += noise_x_values[i];
        proposal_dist(i, 1) += noise_y_values[i];
        proposal_dist(i, 2) += noise_theta_values[i];
    }

}

/**
 * @brief Evaluates sensor model likelihood using beam model and lookup table
 */
void ParticleFilter::sensor_model(const Eigen::MatrixXd &proposal_dist, const std::vector<float> &obs,
                                  std::vector<double> &weights)
{
    const int num_rays = downsampled_angles_.size();
    const int total_queries = num_rays * MAX_PARTICLES;

    // === INITIALIZATION: First-time memory allocation ===
    initialize_sensor_arrays(num_rays, total_queries);

    // === RAY QUERY GENERATION ===
    auto query_start = std::chrono::high_resolution_clock::now();
    generate_ray_queries(proposal_dist, num_rays);
    timing_stats_.query_prep_time += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - query_start).count();

    // === RAY CASTING ===
    ranges_ = calc_range_many(queries_);

    // === WEIGHT CALCULATION ===
    auto sensor_eval_start = std::chrono::high_resolution_clock::now();
    calculate_particle_weights(obs, num_rays, weights);
    timing_stats_.sensor_model_time += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - sensor_eval_start).count();
}

/**
 * @brief Initializes arrays for sensor model computation
 */
void ParticleFilter::initialize_sensor_arrays(int num_rays, int total_queries)
{
    if (first_sensor_update_) {
        queries_ = Eigen::MatrixXd::Zero(total_queries, 3);
        ranges_.resize(total_queries);

        // Pre-allocate sensor model vectors
        obs_px_.resize(num_rays);
        ranges_px_.resize(total_queries);

        // Pre-compute tiled angles for efficiency
        tiled_angles_.resize(total_queries);
        for (int i = 0; i < MAX_PARTICLES; ++i) {
            std::copy(downsampled_angles_.begin(), downsampled_angles_.end(),
                     tiled_angles_.begin() + i * num_rays);
        }
        first_sensor_update_ = false;
    }
}

/**
 * @brief Generates ray queries for batch ray casting
 */
void ParticleFilter::generate_ray_queries(const Eigen::MatrixXd &proposal_dist, int num_rays)
{
    for (int i = 0; i < MAX_PARTICLES; ++i) {
        const int base_idx = i * num_rays;
        const double x = proposal_dist(i, 0);
        const double y = proposal_dist(i, 1);
        const double theta = proposal_dist(i, 2);

        for (int j = 0; j < num_rays; ++j) {
            const int idx = base_idx + j;
            queries_(idx, 0) = x;
            queries_(idx, 1) = y;
            queries_(idx, 2) = theta + downsampled_angles_[j];
        }
    }
}

/**
 * @brief Calculates particle weights using sensor model lookup table
 */
void ParticleFilter::calculate_particle_weights(const std::vector<float> &obs, int num_rays,
                                               std::vector<double> &weights)
{
    // Thread-safe access to map resolution
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(map_lock_);
        local_map = map_msg_;
    }
    if (!local_map) return;
    double resolution = local_map->info.resolution;

    // Convert observations to pixel units (pre-allocated)
    for (size_t i = 0; i < obs.size(); ++i) {
        obs_px_[i] = std::min(static_cast<double>(MAX_RANGE_PX_), obs[i] / resolution);
    }

    // Convert expected ranges to pixel units (pre-allocated)
    for (size_t i = 0; i < ranges_.size(); ++i) {
        ranges_px_[i] = std::min(static_cast<double>(MAX_RANGE_PX_), ranges_[i] / resolution);
    }

    // Compute particle weights using pre-computed sensor model lookup table
    for (int i = 0; i < MAX_PARTICLES; ++i) {
        double weight = 1.0;
        const int base_idx = i * num_rays;

        for (int j = 0; j < num_rays; ++j) {
            const int obs_idx = std::max(0, std::min(static_cast<int>(std::round(obs_px_[j])), MAX_RANGE_PX_));
            const int range_idx = std::max(0, std::min(static_cast<int>(std::round(ranges_px_[base_idx + j])), MAX_RANGE_PX_));

            weight *= sensor_model_table_(obs_idx, range_idx);
        }

        // Apply squash factor AFTER computing the product - optimized
        if (weight > 0.0) {
            weights[i] = std::exp(INV_SQUASH_FACTOR * std::log(weight));
        } else {
            weights[i] = 0.0;
        }
    }
}

// ================================================================================================
// RAY CASTING
// ================================================================================================
/**
 * @brief Performs batch ray casting for multiple queries
 */
std::vector<float> ParticleFilter::calc_range_many(const Eigen::MatrixXd &queries)
{
    auto raycast_start = std::chrono::high_resolution_clock::now();

    std::vector<float> results(queries.rows());

    // Get map once for all ray casting operations
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(map_lock_);
        local_map = map_msg_;
    }

    if (!local_map || !map_initialized_) {
        std::fill(results.begin(), results.end(), MAX_RANGE_METERS);
        return results;
    }

    if (USE_PARALLEL_RAYCASTING) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map);
        }
    } else {
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map);
        }
    }

    auto raycast_end = std::chrono::high_resolution_clock::now();
    timing_stats_.ray_casting_time += std::chrono::duration<double, std::milli>(raycast_end - raycast_start).count();

    return results;
}

/**
 * @brief Casts single ray to find obstacle distance
 */
float ParticleFilter::cast_ray(double x, double y, double angle,
                               const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map)
{
    if (!local_map)
        return MAX_RANGE_METERS;

    double resolution = local_map->info.resolution;
    double origin_x = local_map->info.origin.position.x;
    double origin_y = local_map->info.origin.position.y;

    double dx = std::cos(angle) * resolution;
    double dy = std::sin(angle) * resolution;

    for (int step = 0; step < MAX_RANGE_PX_; ++step)
    {
        x += dx;
        y += dy;

        // World to grid coordinate transformation
        int grid_x = static_cast<int>((x - origin_x) / resolution);
        int grid_y = static_cast<int>((y - origin_y) / resolution);

        // Map boundary collision
        if (grid_x < 0 || grid_x >= static_cast<int>(local_map->info.width) || grid_y < 0 ||
            grid_y >= static_cast<int>(local_map->info.height))
        {
            return step * resolution;
        }

        // Check for obstacles
        int map_idx = grid_y * local_map->info.width + grid_x;
        if (map_idx >= 0 && map_idx < static_cast<int>(local_map->data.size()))
        {
            if (local_map->data[map_idx] > 50)
            {
                return step * resolution;
            }
        }
    }

    return MAX_RANGE_METERS;
}

/**
 * @brief Executes complete MCL cycle: resample, predict, update weights
 */
void ParticleFilter::MCL(const Eigen::Vector3d &action, const std::vector<float> &observation)
{
    auto mcl_start = std::chrono::high_resolution_clock::now();
    
    // 1. Multinomial resampling - generate all indices at once
    auto resample_start = std::chrono::high_resolution_clock::now();
    std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());
    std::vector<int> resample_indices(MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(rng_lock_);
        for (int i = 0; i < MAX_PARTICLES; ++i)
        {
            resample_indices[i] = particle_dist(rng_);
        }
    }

    // Copy particles without lock
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        proposal_distribution_.row(i) = particles_.row(resample_indices[i]);
    }
    auto resample_end = std::chrono::high_resolution_clock::now();
    timing_stats_.resampling_time += std::chrono::duration<double, std::milli>(resample_end - resample_start).count();

    // 2. Motion prediction
    auto motion_start = std::chrono::high_resolution_clock::now();
    motion_model(proposal_distribution_, action);
    auto motion_end = std::chrono::high_resolution_clock::now();
    timing_stats_.motion_model_time += std::chrono::duration<double, std::milli>(motion_end - motion_start).count();

    // 3. Sensor likelihood evaluation
    sensor_model(proposal_distribution_, observation, weights_);

    // 4. Weight normalization with particle diversity check
    double sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (sum_weights > 0)
    {
        for (double &w : weights_)
        {
            w /= sum_weights;
        }

        // Simple resampling without emergency recovery
    }

    // 5. Update particle set - using efficient swap
    particles_.swap(proposal_distribution_);
    
    auto mcl_end = std::chrono::high_resolution_clock::now();
    timing_stats_.total_mcl_time += std::chrono::duration<double, std::milli>(mcl_end - mcl_start).count();
    timing_stats_.measurement_count++;
}

/**
 * @brief Computes weighted mean pose from particles
 */
Eigen::Vector3d ParticleFilter::expected_pose()
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
// POSE SMOOTHING
// ================================================================================================


// ================================================================================================
// TIMER UPDATE
// ================================================================================================
// --------------------------------- MAIN UPDATE LOOP ---------------------------------
/**
 * @brief Main update loop that runs MCL algorithm
 */
void ParticleFilter::timer_update()
{
    if (!lidar_initialized_ || !odom_initialized_ || !map_initialized_)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Waiting for initialization - LiDAR: %s, Odom: %s, Map: %s",
            lidar_initialized_ ? "OK" : "NO",
            odom_initialized_ ? "OK" : "NO",
            map_initialized_ ? "OK" : "NO");
        return;
    }

    // 1. 센서 데이터 복사 (최소 락 시간)
    std::vector<float> observation;
    Eigen::Vector3d action;
    rclcpp::Time lidar_timestamp;
    double velocity, angular_vel;

    // 센서 데이터 빠르게 복사
    {
        std::lock_guard<std::mutex> lock(lidar_lock_);
        // if (!has_new_lidar_data_) {
        //     // No new LiDAR data since last update
        //     RCLCPP_WARN(this->get_logger(), "MCL update skipped - no new LiDAR data");
        //     return;
        // }
        has_new_lidar_data_ = false;
        observation = downsampled_ranges_;
        lidar_timestamp = last_lidar_time_;
    }

    {
        std::lock_guard<std::mutex> lock(odom_lock_);
        velocity = current_velocity_;
        angular_vel = current_angular_vel_;
    }

    // 2. Calculate odometry motion between lidar frames using TF (outside of locks)
    action = calculate_lidar_frame_motion(lidar_timestamp);

    // 3. MCL 상태 락 확보 및 실행
    if (!state_lock_.try_lock()) {
        RCLCPP_INFO(this->get_logger(), "MCL update skipped - previous update still running");
        return;
    }

    // 반복 횟수 관리
    static int current_iters = 0;
    ++current_iters;

    // 4. MCL 알고리즘 실행 (state_lock_ 보호 하에)
    MCL(action, observation);

    // 5. 결과 계산 및 좌표 변환
    Eigen::Vector3d final_pose_laser = expected_pose();
    Eigen::Vector3d final_pose_base = apply_tf_offset(final_pose_laser);

    state_lock_.unlock();

    // 5. 출력 및 시각화 (락 없이) - base_link 좌표 사용
    publish_tf(final_pose_base, lidar_timestamp);

    if (current_iters % 10 == 0)
    {
        // RCLCPP_INFO(this->get_logger(), "MCL iteration %d, pose: (%.3f, %.3f, %.3f)",
        //             current_iters, final_pose_base[0], final_pose_base[1], final_pose_base[2]);
    }

    if (current_iters % 100 == 0)
    {
        // Print performance statistics using utils TimingStats
        timing_stats_.print_stats([this](const std::string& msg) {
            RCLCPP_INFO(this->get_logger(), "%s", msg.c_str());
        });
    }

    visualize(final_pose_base, lidar_timestamp);

    // Publish odometry
    if (PUBLISH_ODOM)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = lidar_timestamp;
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";

        // Use pre-converted base_link coordinate (no TF needed)
        odom.pose.pose.position.x = final_pose_base[0];
        odom.pose.pose.position.y = final_pose_base[1];
        odom.pose.pose.position.z = 0.0;
        odom.pose.pose.orientation = utils::geometry::yaw_to_quaternion(final_pose_base[2]);

        // Set velocity (using pre-copied data)
        odom.twist.twist.linear.x = velocity;
        odom.twist.twist.angular.z = angular_vel;

        odom_pub_->publish(odom);
    }
}


/**
 * @brief Periodically publishes map for RViz visualization
 */
void ParticleFilter::publish_map_periodically()
{
    // Maintain persistent map display in RViz
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(map_lock_);
        local_map = map_msg_;
    }

    if (map_initialized_ && map_pub_ && local_map) {
        map_pub_->publish(*local_map);
    }
}


// ================================================================================================
// OUTPUT & VISUALIZATION
// ================================================================================================
/**
 * @brief Publishes TF transforms for map-odom-base_link chain
 */
void ParticleFilter::publish_tf(const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp)
{
    // base_link_pose is already in base_link frame (no conversion needed)

    // === TF TRANSFORM PUBLISHING ===
    // Real mode: Publish map->odom and odom->base_link
    // Sim mode:  Don't publish TF (simulator handles map->base_link directly)
    
    if (PUBLISH_MAP_ODOM_TF) {
        geometry_msgs::msg::TransformStamped map_to_odom;
        map_to_odom.header.stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();
        map_to_odom.header.frame_id = MAP_FRAME;
        map_to_odom.child_frame_id = ODOM_FRAME;

        if (odom_initialized_ && base_link_pose.norm() > 0) {
            // Calculate map->odom transform: T_map_odom = T_map_base * T_base_odom^(-1)
            double mcl_x = base_link_pose[0], mcl_y = base_link_pose[1], mcl_yaw = base_link_pose[2];

            // Get odometry from tf (odom to base_link) at lidar timestamp
            double odom_x, odom_y, odom_yaw;
            try {
                auto odom_transform = tf_buffer_->lookupTransform(ODOM_FRAME, BASE_FRAME, stamp);
                odom_x = odom_transform.transform.translation.x;
                odom_y = odom_transform.transform.translation.y;
                odom_yaw = utils::geometry::quaternion_to_yaw(odom_transform.transform.rotation);
            } catch (tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "Could not get odom->base_link transform: %s. Using current pose fallback.", ex.what());
                odom_x = base_link_pose[0];
                odom_y = base_link_pose[1];
                odom_yaw = base_link_pose[2];
            }

            // Inverse odom transform
            double cos_odom_inv = std::cos(-odom_yaw), sin_odom_inv = std::sin(-odom_yaw);
            double inv_odom_x = -(odom_x * cos_odom_inv - odom_y * sin_odom_inv);
            double inv_odom_y = -(odom_x * sin_odom_inv + odom_y * cos_odom_inv);
            double inv_odom_yaw = -odom_yaw;

            // Compose transforms
            double cos_mcl = std::cos(mcl_yaw), sin_mcl = std::sin(mcl_yaw);
            map_to_odom.transform.translation.x = mcl_x + inv_odom_x * cos_mcl - inv_odom_y * sin_mcl;
            map_to_odom.transform.translation.y = mcl_y + inv_odom_x * sin_mcl + inv_odom_y * cos_mcl;
            map_to_odom.transform.translation.z = 0.0;
            map_to_odom.transform.rotation = utils::geometry::yaw_to_quaternion(
                utils::geometry::normalize_angle(mcl_yaw + inv_odom_yaw));
        } else {
            // Identity transform fallback
            map_to_odom.transform.translation.x = 0.0;
            map_to_odom.transform.translation.y = 0.0;
            map_to_odom.transform.translation.z = 0.0;
            map_to_odom.transform.rotation = utils::geometry::yaw_to_quaternion(0.0);
            RCLCPP_WARN(this->get_logger(), "Odom not initialized, publishing identity map->odom transform");
        }
        pub_tf_->sendTransform(map_to_odom);
    }

    if (PUBLISH_ODOM_BASE_TF && odom_initialized_ && base_link_pose.norm() > 0) {
        RCLCPP_INFO(this->get_logger(), "Publishing odom->base_link TF at [%.3f, %.3f, %.3f]",
                    base_link_pose[0], base_link_pose[1], base_link_pose[2]);
        geometry_msgs::msg::TransformStamped odom_to_base;
        odom_to_base.header.stamp = stamp;
        odom_to_base.header.frame_id = ODOM_FRAME;
        odom_to_base.child_frame_id = BASE_FRAME;
        odom_to_base.transform.translation.x = base_link_pose[0];
        odom_to_base.transform.translation.y = base_link_pose[1];
        odom_to_base.transform.translation.z = 0.0;
        odom_to_base.transform.rotation = utils::geometry::yaw_to_quaternion(base_link_pose[2]);
        pub_tf_->sendTransform(odom_to_base);
    }

    // Odometry publishing moved to timer_update for controlled frequency
}

/**
 * @brief Validates pose for reasonable bounds
 */
bool ParticleFilter::is_pose_valid(const Eigen::Vector3d& pose)
{
    return utils::validation::is_pose_valid(pose, MAX_POSE_RANGE);
}

/**
 * @brief Publishes visualization data for RViz
 */
void ParticleFilter::visualize(const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp)
{
    if (!DO_VIZ)
        return;

    // Use provided timestamp or fallback to current time
    rclcpp::Time viz_stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();

    // RViz pose visualization (already in base_link frame)
    if (pose_pub_ && pose_pub_->get_subscription_count() > 0)
    {
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = viz_stamp;
        ps.header.frame_id = "map";
        ps.pose.position.x = base_link_pose[0];
        ps.pose.position.y = base_link_pose[1];
        ps.pose.orientation = utils::geometry::yaw_to_quaternion(base_link_pose[2]);
        pose_pub_->publish(ps);
    }

    // RViz particle cloud (downsampled for performance)
    if (particle_pub_ && particle_pub_->get_subscription_count() > 0)
    {
        if (MAX_PARTICLES > MAX_VIZ_PARTICLES)
        {
            // Weighted downsampling - generate all indices at once
            std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());
            Eigen::MatrixXd viz_particles(MAX_VIZ_PARTICLES, 3);
            std::vector<int> indices(MAX_VIZ_PARTICLES);

            {
                std::lock_guard<std::mutex> lock(rng_lock_);
                for (int i = 0; i < MAX_VIZ_PARTICLES; ++i)
                {
                    indices[i] = particle_dist(rng_);
                }
            }

            // Copy particles without lock
            for (int i = 0; i < MAX_VIZ_PARTICLES; ++i)
            {
                viz_particles.row(i) = particles_.row(indices[i]);
            }

            publish_particles(viz_particles, viz_stamp);
        }
        else
        {
            publish_particles(particles_, viz_stamp);
        }
    }
}

/**
 * @brief Publishes particle cloud for visualization
 */
void ParticleFilter::publish_particles(const Eigen::MatrixXd &particles_to_pub, const rclcpp::Time &stamp)
{
    // Apply vehicle frame offset to all particles
    Eigen::MatrixXd offset_particles = particles_to_pub;

    try {
        // Get TF transform once for all particles (static transform)
        auto transform = tf_buffer_->lookupTransform(BASE_FRAME, LASER_FRAME, tf2::TimePointZero);
        double offset_x = transform.transform.translation.x;
        double offset_y = transform.transform.translation.y;

        // Apply same offset to all particles
        for (int i = 0; i < offset_particles.rows(); ++i) {
            double cos_theta = std::cos(offset_particles(i, 2));
            double sin_theta = std::sin(offset_particles(i, 2));

            double new_x = offset_particles(i, 0) - offset_x * cos_theta + offset_y * sin_theta;
            double new_y = offset_particles(i, 1) - offset_x * sin_theta - offset_y * cos_theta;

            offset_particles(i, 0) = new_x;
            offset_particles(i, 1) = new_y;
        }
    }
    catch (tf2::TransformException &ex) {
        // Fallback: use default F1Tenth offset
        static bool warning_shown = false;
        if (!warning_shown) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed for particles, using default offset: %s", ex.what());
            warning_shown = true;
        }

        double default_offset_x = 0.28;
        for (int i = 0; i < offset_particles.rows(); ++i) {
            double cos_theta = std::cos(offset_particles(i, 2));
            double sin_theta = std::sin(offset_particles(i, 2));

            offset_particles(i, 0) -= default_offset_x * cos_theta;
            offset_particles(i, 1) -= default_offset_x * sin_theta;
        }
    }

    auto pa = utils::particles_to_pose_array(offset_particles);
    pa.header.stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();
    pa.header.frame_id = "map";
    particle_pub_->publish(pa);
}



// ================================================================================================
// TF UTILITIES
// ================================================================================================
/**
 * @brief Applies TF offset from laser frame to base_link frame
 */
Eigen::Vector3d ParticleFilter::apply_tf_offset(const Eigen::Vector3d& pose_in_laser_frame)
{
    try {
        // Create pose in laser frame
        geometry_msgs::msg::PoseStamped laser_pose;
        laser_pose.header.frame_id = LASER_FRAME;
        laser_pose.pose.position.x = pose_in_laser_frame[0];
        laser_pose.pose.position.y = pose_in_laser_frame[1];
        laser_pose.pose.position.z = 0.0;
        laser_pose.pose.orientation = utils::geometry::yaw_to_quaternion(pose_in_laser_frame[2]);

        // Transform to base_link frame using TF system
        geometry_msgs::msg::PoseStamped base_pose;
        tf_buffer_->transform(laser_pose, base_pose, BASE_FRAME);

        // Convert back to Eigen::Vector3d
        return Eigen::Vector3d(
            base_pose.pose.position.x,
            base_pose.pose.position.y,
            utils::geometry::quaternion_to_yaw(base_pose.pose.orientation)
        );
    }
    catch (tf2::TransformException &ex) {
        // Fallback: use default F1Tenth offset if TF lookup fails
        static bool warning_shown = false;
        if (!warning_shown) {
            RCLCPP_WARN(this->get_logger(), "TF transform failed, using default F1Tenth offset (0.27m): %s", ex.what());
            warning_shown = true;
        }

        // Manual offset calculation as fallback
        double offset_x = 0.28;  // Default F1Tenth lidar offset
        double cos_theta = std::cos(pose_in_laser_frame[2]);
        double sin_theta = std::sin(pose_in_laser_frame[2]);

        return Eigen::Vector3d(
            pose_in_laser_frame[0] - offset_x * cos_theta,
            pose_in_laser_frame[1] - offset_x * sin_theta,
            pose_in_laser_frame[2]
        );
    }
}

/**
 * @brief Calculates odometry motion between consecutive lidar frames using TF
 */
Eigen::Vector3d ParticleFilter::calculate_lidar_frame_motion(const rclcpp::Time& current_lidar_stamp)
{
    // Static variable to store the last lidar timestamp
    static rclcpp::Time last_processed_lidar_stamp = rclcpp::Time(0);

    // If this is the first call, just initialize and return zero motion
    if (last_processed_lidar_stamp.nanoseconds() == 0) {
        last_processed_lidar_stamp = current_lidar_stamp;
        return Eigen::Vector3d::Zero();
    }

    try {
        // Get odom->base_link transform at current lidar timestamp
        auto current_transform = tf_buffer_->lookupTransform(
            ODOM_FRAME, BASE_FRAME, current_lidar_stamp);

        // Get odom->base_link transform at previous lidar timestamp
        auto previous_transform = tf_buffer_->lookupTransform(
            ODOM_FRAME, BASE_FRAME, last_processed_lidar_stamp);

        // Extract poses
        Eigen::Vector3d current_pose(
            current_transform.transform.translation.x,
            current_transform.transform.translation.y,
            utils::geometry::quaternion_to_yaw(current_transform.transform.rotation)
        );

        Eigen::Vector3d previous_pose(
            previous_transform.transform.translation.x,
            previous_transform.transform.translation.y,
            utils::geometry::quaternion_to_yaw(previous_transform.transform.rotation)
        );

        // Calculate global displacement
        Eigen::Vector2d delta_global = current_pose.head<2>() - previous_pose.head<2>();
        double delta_theta = current_pose[2] - previous_pose[2];

        // Transform global displacement to robot-local coordinates using previous pose
        Eigen::Matrix2d rot = utils::geometry::rotation_matrix(-previous_pose[2]);
        Eigen::Vector2d delta_local = rot * delta_global;

        // Update last processed timestamp
        last_processed_lidar_stamp = current_lidar_stamp;

        // Return motion in robot frame: [forward, lateral, rotation]
        return Eigen::Vector3d(delta_local[0], delta_local[1], delta_theta);

    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(),
            "Could not get transform for lidar frame motion calculation: %s", ex.what());
        return Eigen::Vector3d::Zero();
    }
}


} // namespace particle_filter_cpp

// ================================================================================================
// PROGRAM ENTRY POINT
// ================================================================================================
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<particle_filter_cpp::ParticleFilter>());
    rclcpp::shutdown();
    return 0;
}
