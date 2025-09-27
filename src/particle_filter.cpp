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
#include <numeric>
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
    : Node("particle_filter", options), rng_(std::random_device{}()), uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0)
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
    this->declare_parameter("timer_frequency", 100.0);
    
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
    MAX_RANGE_PX = 0;
    iters_ = 0;
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
    particle_indices_.resize(MAX_PARTICLES);
    std::iota(particle_indices_.begin(), particle_indices_.end(), 0);

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
        map_resolution_ = map_msg_->info.resolution;
        map_origin_ = Eigen::Vector3d(map_msg_->info.origin.position.x, map_msg_->info.origin.position.y,
                                      utils::geometry::quaternion_to_yaw(map_msg_->info.origin.orientation));

        MAX_RANGE_PX = static_cast<int>(MAX_RANGE_METERS / map_resolution_);

        // Extract free space for particle initialization
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
    if (laser_angles_.empty())
    {
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

        RCLCPP_INFO(this->get_logger(), "LiDAR initialized - %zu angles", downsampled_angles_.size());
    }

    // Extract downsampled measurements
    downsampled_ranges_.clear();
    for (size_t i = 0; i < msg->ranges.size(); i += ANGLE_STEP)
    {
        downsampled_ranges_.push_back(msg->ranges[i]);
    }

    // Mark new lidar data available - protected by mutex
    {
        std::lock_guard<std::mutex> lock(state_lock_);
        last_lidar_time_ = msg->header.stamp;
        has_new_lidar_data_ = true;

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "LiDAR callback: new data received, timestamp: %d.%09u",
            msg->header.stamp.sec, msg->header.stamp.nanosec);
    }
    lidar_initialized_ = true;
}

/**
 * @brief Processes odometry data and triggers MCL update
 */
void ParticleFilter::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Store velocity information
    current_velocity_ = msg->twist.twist.linear.x;
    current_angular_vel_ = msg->twist.twist.angular.z;

    // Store pose data
    Eigen::Vector3d position(msg->pose.pose.position.x, msg->pose.pose.position.y,
                             utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation));


    if (last_pose_.norm() > 0)
    {
        // Transform global displacement to robot-local coordinates
        Eigen::Matrix2d rot = utils::geometry::rotation_matrix(-last_pose_[2]);
        Eigen::Vector2d delta = position.head<2>() - last_pose_.head<2>();
        Eigen::Vector2d local_delta = rot * delta;

        // Use the motion directly for MCL update
        odometry_data_ = Eigen::Vector3d(local_delta[0], local_delta[1], position[2] - last_pose_[2]);

        last_pose_ = position;
        last_stamp_ = msg->header.stamp;
        odom_initialized_ = true;

        // Trigger MCL update on every odometry message
        update();
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Odometry initialized");
        last_pose_ = position;
        last_stamp_ = msg->header.stamp;
        odom_initialized_ = true;
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
    Eigen::Vector3d pose(msg->pose.pose.position.x, msg->pose.pose.position.y,
                         utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation));
    
    // Initialize particle filter around clicked pose
    initialize_particles_pose(pose);

    // Initialize odometry-based tracking from this pose

    // Set inferred pose immediately for visualization
    inferred_pose_ = pose;

    // Simple initialization

    RCLCPP_INFO(this->get_logger(), "Pose initialized from RViz at [%.3f, %.3f, %.3f]",
                pose[0], pose[1], pose[2]);

    // Trigger immediate visualization update
    visualize(this->get_clock()->now());
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

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        particles_(i, 0) = pose[0] + normal_dist_(rng_) * pos_std;
        particles_(i, 1) = pose[1] + normal_dist_(rng_) * pos_std;
        particles_(i, 2) = pose[2] + normal_dist_(rng_) * angle_std;

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

    std::lock_guard<std::mutex> lock(state_lock_);

    // Extract free space cells
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
        RCLCPP_ERROR(this->get_logger(), "No free space found in map!");
        return;
    }

    // Sample particles uniformly over free space
    std::uniform_int_distribution<int> pos_dist(0, permissible_positions.size() - 1);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        int idx = pos_dist(rng_);
        auto pos = permissible_positions[idx];

        particles_(i, 0) = pos.second * map_resolution_ + map_origin_[0];
        particles_(i, 1) = pos.first * map_resolution_ + map_origin_[1];
        particles_(i, 2) = angle_dist(rng_);
    }

    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

    // Calculate expected pose from initialized particles for odometry tracking
    Eigen::Vector3d initial_pose = expected_pose();

    // Initialize odometry tracking from global initialization (not from RViz)

    RCLCPP_INFO(this->get_logger(), "Initialized %d particles globally with odometry tracking at [%.3f, %.3f, %.3f]",
                MAX_PARTICLES, initial_pose[0], initial_pose[1], initial_pose[2]);
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

    // Add Gaussian noise
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        proposal_dist(i, 0) += normal_dist_(rng_) * MOTION_DISPERSION_X;
        proposal_dist(i, 1) += normal_dist_(rng_) * MOTION_DISPERSION_Y;
        proposal_dist(i, 2) += normal_dist_(rng_) * MOTION_DISPERSION_THETA;
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
    // Convert observations to pixel units
    obs_px_.resize(obs.size());
    for (size_t i = 0; i < obs.size(); ++i) {
        obs_px_[i] = std::min(static_cast<double>(MAX_RANGE_PX), obs[i] / map_resolution_);
    }

    // Convert expected ranges to pixel units
    ranges_px_.resize(ranges_.size());
    for (size_t i = 0; i < ranges_.size(); ++i) {
        ranges_px_[i] = std::min(static_cast<double>(MAX_RANGE_PX), ranges_[i] / map_resolution_);
    }

    // Compute particle weights using pre-computed sensor model lookup table
    for (int i = 0; i < MAX_PARTICLES; ++i) {
        double weight = 1.0;
        const int base_idx = i * num_rays;

        for (int j = 0; j < num_rays; ++j) {
            const int obs_idx = std::max(0, std::min(static_cast<int>(std::round(obs_px_[j])), MAX_RANGE_PX));
            const int range_idx = std::max(0, std::min(static_cast<int>(std::round(ranges_px_[base_idx + j])), MAX_RANGE_PX));

            weight *= sensor_model_table_(obs_idx, range_idx);
        }

        // Apply squash factor AFTER computing the product
        weights[i] = std::pow(weight, INV_SQUASH_FACTOR);
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

    if (USE_PARALLEL_RAYCASTING) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2));
        }
    } else {
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2));
        }
    }

    auto raycast_end = std::chrono::high_resolution_clock::now();
    timing_stats_.ray_casting_time += std::chrono::duration<double, std::milli>(raycast_end - raycast_start).count();
    
    return results;
}

/**
 * @brief Casts single ray to find obstacle distance
 */
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

        // Check for obstacles
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

/**
 * @brief Executes complete MCL cycle: resample, predict, update weights
 */
void ParticleFilter::MCL(const Eigen::Vector3d &action, const std::vector<float> &observation)
{
    auto mcl_start = std::chrono::high_resolution_clock::now();
    
    // 1. Multinomial resampling - using pre-allocated memory
    auto resample_start = std::chrono::high_resolution_clock::now();
    std::discrete_distribution<int> particle_dist(weights_.begin(), weights_.end());

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        int idx = particle_dist(rng_);
        proposal_distribution_.row(i) = particles_.row(idx);
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

        // Execute complete MCL cycle - pass action directly
        MCL(action, observation);

        // Final pose estimate: weighted mean
        inferred_pose_ = expected_pose();

        state_lock_.unlock();

        // Output to navigation stack and visualization
        publish_tf(inferred_pose_, last_lidar_time_);

        if (iters_ % 10 == 0)
        {
            RCLCPP_INFO(this->get_logger(), "MCL iteration %d, pose: (%.3f, %.3f, %.3f)", iters_, inferred_pose_[0],
                        inferred_pose_[1], inferred_pose_[2]);
        }

        if (iters_ % 100 == 0)
        {
            // Print performance statistics using utils TimingStats
            timing_stats_.print_stats([this](const std::string& msg) {
                RCLCPP_INFO(this->get_logger(), "%s", msg.c_str());
            });
        }

        visualize();
    }
    else
    {
        RCLCPP_INFO(this->get_logger(), "Concurrency error avoided");
    }
}

/**
 * @brief Timer callback for publishing odometry
 */
void ParticleFilter::timer_update()
{
    // MCL update is now triggered by odometry callback
    // Timer only handles publishing and visualization

    // Publish odometry
    if (PUBLISH_ODOM && odom_pub_)
    {
        nav_msgs::msg::Odometry odom;
        odom.header.stamp = this->now();
        odom.header.frame_id = "map";
        odom.child_frame_id = "base_link";

        // Use TF lookup for smooth interpolated pose (eliminates MCL vibration)
        // TF provides same global accuracy but with smooth interpolation between updates
        try {
            auto transform = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);

            odom.pose.pose.position.x = transform.transform.translation.x;
            odom.pose.pose.position.y = transform.transform.translation.y;
            odom.pose.pose.position.z = transform.transform.translation.z;
            odom.pose.pose.orientation = transform.transform.rotation;
        }
        catch (const tf2::TransformException& ex) {
            // Fallback to raw MCL output if TF lookup fails
            Eigen::Vector3d current_pose = get_current_pose();
            odom.pose.pose.position.x = current_pose[0];
            odom.pose.pose.position.y = current_pose[1];
            odom.pose.pose.position.z = 0.0;
            odom.pose.pose.orientation = utils::geometry::yaw_to_quaternion(current_pose[2]);

            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "TF lookup failed for /pf/pose/odom, using raw MCL: %s", ex.what());
        }

        // Set velocity
        odom.twist.twist.linear.x = current_velocity_;
        odom.twist.twist.angular.z = current_angular_vel_;

        odom_pub_->publish(odom);
    }
}

/**
 * @brief Periodically publishes map for RViz visualization
 */
void ParticleFilter::publish_map_periodically()
{
    // Maintain persistent map display in RViz
    if (map_initialized_ && map_pub_ && map_msg_) {
        map_pub_->publish(*map_msg_);
    }
}


// ================================================================================================
// OUTPUT & VISUALIZATION
// ================================================================================================
/**
 * @brief Publishes TF transforms for map-odom-base_link chain
 */
void ParticleFilter::publish_tf(const Eigen::Vector3d &pose, const rclcpp::Time &stamp)
{
    Eigen::Vector3d base_link_pose = apply_tf_offset(pose);

    // === TF TRANSFORM PUBLISHING ===
    // Real mode: Publish map->odom and odom->base_link
    // Sim mode:  Don't publish TF (simulator handles map->base_link directly)
    
    if (PUBLISH_MAP_ODOM_TF) {
        geometry_msgs::msg::TransformStamped map_to_odom;
        map_to_odom.header.stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();
        map_to_odom.header.frame_id = MAP_FRAME;
        map_to_odom.child_frame_id = ODOM_FRAME;

        if (odom_initialized_ && last_pose_.norm() > 0) {
            // Calculate map->odom transform: T_map_odom = T_map_base * T_base_odom^(-1)
            double mcl_x = pose[0], mcl_y = pose[1], mcl_yaw = pose[2];

            // Get odometry from tf (odom to base_link) at lidar timestamp
            double odom_x, odom_y, odom_yaw;
            try {
                auto odom_transform = tf_buffer_->lookupTransform(ODOM_FRAME, BASE_FRAME, stamp);
                odom_x = odom_transform.transform.translation.x;
                odom_y = odom_transform.transform.translation.y;
                odom_yaw = utils::geometry::quaternion_to_yaw(odom_transform.transform.rotation);
            } catch (tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "Could not get odom->base_link transform: %s. Using last_pose_ fallback.", ex.what());
                odom_x = last_pose_[0];
                odom_y = last_pose_[1];
                odom_yaw = last_pose_[2];
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
        }
        pub_tf_->sendTransform(map_to_odom);
    }

    if (PUBLISH_ODOM_BASE_TF && odom_initialized_ && last_pose_.norm() > 0) {
        geometry_msgs::msg::TransformStamped odom_to_base;
        odom_to_base.header.stamp = stamp;
        odom_to_base.header.frame_id = ODOM_FRAME;
        odom_to_base.child_frame_id = BASE_FRAME;
        odom_to_base.transform.translation.x = last_pose_[0];
        odom_to_base.transform.translation.y = last_pose_[1];
        odom_to_base.transform.translation.z = 0.0;
        odom_to_base.transform.rotation = utils::geometry::yaw_to_quaternion(last_pose_[2]);
        pub_tf_->sendTransform(odom_to_base);
    }

    // Odometry publishing moved to timer_update for controlled frequency
}


/**
 * @brief Returns current best pose estimate with fallback logic
 */
Eigen::Vector3d ParticleFilter::get_current_pose()
{
    // Use particle filter estimate primarily - avoid jumping between sources
    if (is_pose_valid(inferred_pose_))
        return inferred_pose_;

    // During initialization, use center of particles
    if (map_initialized_ && particles_.rows() > 0) {
        Eigen::Vector3d particle_center = particles_.colwise().mean();
        if (is_pose_valid(particle_center)) {
            return particle_center;
        }
    }

    // Fallback to last known good pose
    if (is_pose_valid(last_pose_))
        return last_pose_;

    // Default to origin
    return Eigen::Vector3d::Zero();
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
void ParticleFilter::visualize(const rclcpp::Time &stamp)
{
    if (!DO_VIZ)
        return;

    // Use provided timestamp or fallback to current time
    rclcpp::Time viz_stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();

    // RViz pose visualization (with vehicle frame offset)
    if (pose_pub_ && pose_pub_->get_subscription_count() > 0)
    {
        // Apply vehicle frame offset using TF
        Eigen::Vector3d offset_pose = apply_tf_offset(inferred_pose_);
        
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = viz_stamp;
        ps.header.frame_id = "map";
        ps.pose.position.x = offset_pose[0];
        ps.pose.position.y = offset_pose[1];
        ps.pose.orientation = utils::geometry::yaw_to_quaternion(inferred_pose_[2]);
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

    for (int i = 0; i < offset_particles.rows(); ++i) {
        Eigen::Vector3d particle_pose(offset_particles(i, 0), offset_particles(i, 1), offset_particles(i, 2));
        Eigen::Vector3d offset_pose = apply_tf_offset(particle_pose);
        offset_particles(i, 0) = offset_pose[0];
        offset_particles(i, 1) = offset_pose[1];
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
    // Get the offset from F1Tenth system's static transform
    static double lidar_offset_x = 0.27;  // Default fallback
    static double lidar_offset_y = 0.0;
    static bool offset_read = false;
    static int tf_retry_count = 0;
    static std::chrono::steady_clock::time_point last_tf_attempt = std::chrono::steady_clock::now();

    // Implement backoff strategy to prevent excessive TF lookups during startup
    if (!offset_read && tf_retry_count < 10) {
        auto now = std::chrono::steady_clock::now();
        auto time_since_last_attempt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tf_attempt);

        // Exponential backoff: wait longer between attempts
        int backoff_ms = 100 * (1 << std::min(tf_retry_count, 4)); // 100, 200, 400, 800, 1600ms max

        if (time_since_last_attempt.count() >= backoff_ms) {
            try {
                // Use shorter timeout to prevent blocking
                auto transform = tf_buffer_->lookupTransform(
                    BASE_FRAME, LASER_FRAME, tf2::TimePointZero, tf2::Duration(std::chrono::milliseconds(50)));

                lidar_offset_x = transform.transform.translation.x;
                lidar_offset_y = transform.transform.translation.y;
                offset_read = true;

                RCLCPP_INFO(this->get_logger(), "Using F1Tenth TF offset: x=%.3fm, y=%.3fm",
                           lidar_offset_x, lidar_offset_y);
            }
            catch (tf2::TransformException &ex) {
                tf_retry_count++;
                last_tf_attempt = now;

                if (tf_retry_count >= 10) {
                    RCLCPP_WARN(this->get_logger(), "Could not read F1Tenth TF after %d attempts, using default 0.27m: %s",
                               tf_retry_count, ex.what());
                    offset_read = true;  // Stop trying after max attempts
                }
            }
        }
    }

    // Apply offset: laser frame pose → base_link frame pose
    double cos_theta = std::cos(pose_in_laser_frame[2]);
    double sin_theta = std::sin(pose_in_laser_frame[2]);

    Eigen::Vector3d base_link_pose;
    base_link_pose[0] = pose_in_laser_frame[0] - lidar_offset_x * cos_theta + lidar_offset_y * sin_theta;
    base_link_pose[1] = pose_in_laser_frame[1] - lidar_offset_x * sin_theta - lidar_offset_y * cos_theta;
    base_link_pose[2] = pose_in_laser_frame[2];

    return base_link_pose;
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
