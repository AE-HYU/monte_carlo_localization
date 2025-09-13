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

// --------------------------------- CONSTRUCTOR & INITIALIZATION ---------------------------------
ParticleFilter::ParticleFilter(const rclcpp::NodeOptions &options)
    : Node("particle_filter", options), rng_(std::random_device{}()), uniform_dist_(0.0, 1.0), normal_dist_(0.0, 1.0)
{
    // ROS2 parameter declarations
    this->declare_parameter("angle_step", 18);
    this->declare_parameter("max_particles", 2000);
    this->declare_parameter("max_viz_particles", 60);
    this->declare_parameter("squash_factor", 2.2);
    this->declare_parameter("max_range", 12.0);
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

    // Retrieve parameter values
    ANGLE_STEP = this->get_parameter("angle_step").as_int();
    MAX_PARTICLES = this->get_parameter("max_particles").as_int();
    MAX_VIZ_PARTICLES = this->get_parameter("max_viz_particles").as_int();
    INV_SQUASH_FACTOR = 1.0 / this->get_parameter("squash_factor").as_double();
    MAX_RANGE_METERS = this->get_parameter("max_range").as_double();
    PUBLISH_ODOM = this->get_parameter("publish_odom").as_bool();
    DO_VIZ = this->get_parameter("viz").as_bool();
    TIMER_FREQUENCY = this->get_parameter("timer_frequency").as_double();
    USE_PARALLEL_RAYCASTING = this->get_parameter("use_parallel_raycasting").as_bool();
    NUM_THREADS = this->get_parameter("num_threads").as_int();
    MAX_POSE_RANGE = this->get_parameter("max_pose_range").as_double();

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
    iters_ = 0;
    map_initialized_ = false;
    lidar_initialized_ = false;
    odom_initialized_ = false;
    first_sensor_update_ = true;
    current_velocity_ = 0.0;
    current_angular_vel_ = 0.0;
    
    // Odometry tracking initialization
    odom_pose_ = Eigen::Vector3d::Zero();
    odom_reference_pose_ = Eigen::Vector3d::Zero();
    odom_reference_odom_ = Eigen::Vector3d::Zero();
    pose_initialized_from_rviz_ = false;
    odom_tracking_active_ = false;
    
    // Setup OpenMP threading
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

    RCLCPP_INFO(this->get_logger(), "Particle filter initialized - %.1fHz, %s threading (%d threads)", 
        TIMER_FREQUENCY, USE_PARALLEL_RAYCASTING ? "parallel" : "sequential", 
        USE_PARALLEL_RAYCASTING ? NUM_THREADS : 1);
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
                                      utils::geometry::quaternion_to_yaw(map_msg_->info.origin.orientation));

        MAX_RANGE_PX = static_cast<int>(MAX_RANGE_METERS / map_resolution_);


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
        RCLCPP_INFO(this->get_logger(), "Map loaded and published");

        // Publish map immediately
        if (map_pub_) {
            map_pub_->publish(*map_msg_);
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

// --------------------------------- SENSOR CALLBACKS ---------------------------------
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

    lidar_initialized_ = true;
}

void ParticleFilter::odomCB(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Store velocity information
    current_velocity_ = msg->twist.twist.linear.x;
    current_angular_vel_ = msg->twist.twist.angular.z;

    // Update odometry tracking if active
    bool can_use_odom_tracking = pose_initialized_from_rviz_ || 
                               (map_initialized_ && iters_ > 0 && is_pose_valid(inferred_pose_));
    
    if (can_use_odom_tracking && odom_tracking_active_) {
        update_odom_pose(msg);
    }

    // Store pose data
    Eigen::Vector3d position(msg->pose.pose.position.x, msg->pose.pose.position.y,
                             utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation));

    if (last_pose_.norm() <= 0)
    {
        RCLCPP_INFO(this->get_logger(), "Odometry initialized");
        last_pose_ = position;
    }
    
    last_pose_ = position;
    last_stamp_ = msg->header.stamp;
    odom_initialized_ = true;
}

// --------------------------------- INTERACTIVE INITIALIZATION ---------------------------------
void ParticleFilter::clicked_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    Eigen::Vector3d pose(msg->pose.pose.position.x, msg->pose.pose.position.y,
                         utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation));
    
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
    RCLCPP_INFO(this->get_logger(), "Initializing particles at [%.3f, %.3f, %.3f]", 
                pose[0], pose[1], pose[2]);

    std::lock_guard<std::mutex> lock(state_lock_);
    std::fill(weights_.begin(), weights_.end(), 1.0 / MAX_PARTICLES);

    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        particles_(i, 0) = pose[0] + normal_dist_(rng_) * 0.5;
        particles_(i, 1) = pose[1] + normal_dist_(rng_) * 0.5;
        particles_(i, 2) = pose[2] + normal_dist_(rng_) * 0.4;
        
        // Normalize angle
        particles_(i, 2) = utils::geometry::normalize_angle(particles_(i, 2));
    }
}

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

    RCLCPP_INFO(this->get_logger(), "Initialized %d particles globally", MAX_PARTICLES);
}

// --------------------------------- MCL ALGORITHM CORE ---------------------------------
void ParticleFilter::motion_model(Eigen::MatrixXd &proposal_dist, const Eigen::Vector3d &action)
{
    // action[0] = forward displacement, action[2] = angular displacement
    double dt = 0.01;
    double velocity = 0.0;
    double angular_velocity = 0.0;
    
    double forward_displacement = action[0];
    double angular_displacement = action[2];
    
    if (std::abs(forward_displacement) > 0.001) {
        if (std::abs(forward_displacement) < 0.1) {
            dt = std::abs(forward_displacement) / 1.0;
        } else {
            dt = std::abs(forward_displacement) / 5.0;
        }
        dt = std::max(0.001, std::min(dt, 0.1));
        velocity = forward_displacement / dt;
    }
    
    if (std::abs(angular_displacement) > 0.001) {
        angular_velocity = angular_displacement / dt;
    }

    // Apply bicycle model kinematics
    for (int i = 0; i < MAX_PARTICLES; ++i)
    {
        double x = proposal_dist(i, 0);
        double y = proposal_dist(i, 1);
        double theta = proposal_dist(i, 2);
        
        if (std::abs(angular_velocity) < 1e-6) {
            // Straight line motion
            proposal_dist(i, 0) = x + velocity * dt * std::cos(theta);
            proposal_dist(i, 1) = y + velocity * dt * std::sin(theta);
            proposal_dist(i, 2) = theta;
        } else {
            // Curved motion
            double radius = velocity / angular_velocity;
            double delta_theta = angular_velocity * dt;
            
            proposal_dist(i, 0) = x + radius * (std::sin(theta + delta_theta) - std::sin(theta));
            proposal_dist(i, 1) = y - radius * (std::cos(theta + delta_theta) - std::cos(theta));
            proposal_dist(i, 2) = theta + delta_theta;
        }
        
        // Add noise
        proposal_dist(i, 0) += normal_dist_(rng_) * MOTION_DISPERSION_X;
        proposal_dist(i, 1) += normal_dist_(rng_) * MOTION_DISPERSION_Y;
        proposal_dist(i, 2) += normal_dist_(rng_) * MOTION_DISPERSION_THETA;
        
        // Normalize angle
        proposal_dist(i, 2) = utils::geometry::normalize_angle(proposal_dist(i, 2));
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


// --------------------------------- CONFIGURABLE TIMER UPDATE ---------------------------------
void ParticleFilter::timer_update()
{
    if (!map_initialized_) {
        return;
    }
    
    bool has_odometry_for_motion = odom_initialized_;
    if (!has_odometry_for_motion) {
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 10000, 
                            "Running MCL without odometry");
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
    
    // Skip if time step is too large
    if (dt > 1.0) {
        return;
    }
    
    bool apply_motion = (dt >= 0.0001);
    
    if (state_lock_.try_lock()) {
        
        if (lidar_initialized_ && !downsampled_ranges_.empty()) {
            ++iters_;
            
            Eigen::Vector3d action = Eigen::Vector3d::Zero();
            if (has_odometry_for_motion && apply_motion && 
                (std::abs(current_velocity_) > 0.0001 || std::abs(current_angular_vel_) > 0.0001)) {
                action[0] = current_velocity_ * dt;
                action[1] = 0.0;
                action[2] = current_angular_vel_ * dt;
            } else if (!has_odometry_for_motion && !pose_initialized_from_rviz_ && iters_ < 15) {
                double noise_factor = std::max(0.1, 1.0 - (static_cast<double>(iters_) / 15.0));
                action[0] = normal_dist_(rng_) * 0.02 * noise_factor;
                action[1] = normal_dist_(rng_) * 0.01 * noise_factor;
                action[2] = normal_dist_(rng_) * 0.05 * noise_factor;
            }
            
            auto observation = downsampled_ranges_;
            
            // Execute MCL pipeline
            MCL(action, observation);
            inferred_pose_ = expected_pose();
            
            // Update odometry tracking
            bool can_use_odom_tracking = has_odometry_for_motion && 
                (pose_initialized_from_rviz_ || (map_initialized_ && iters_ > 0 && is_pose_valid(inferred_pose_)));
            
            if (can_use_odom_tracking) {
                if (!odom_tracking_active_ && is_pose_valid(inferred_pose_)) {
                    initialize_odom_tracking(inferred_pose_, false);
                    RCLCPP_INFO(this->get_logger(), "Odometry tracking initialized");
                }
                
                odom_reference_pose_ = inferred_pose_;
                odom_reference_odom_ = last_pose_;
                odom_pose_ = inferred_pose_;
            }
            
            if (iters_ % 100 == 0) {
                RCLCPP_INFO(this->get_logger(), "MCL iter %d: [%.2f, %.2f, %.2f]", iters_, 
                           inferred_pose_[0], inferred_pose_[1], inferred_pose_[2]);
            }
            
            if (iters_ % 200 == 0) {
                // Print performance stats
                auto logger_func = [this](const std::string& msg) {
                    RCLCPP_INFO(this->get_logger(), "%s", msg.c_str());
                };
                timing_stats_.print_stats(logger_func);
                
                if (timing_stats_.measurement_count > 0) {
                    RCLCPP_INFO(this->get_logger(), 
                        "Particles: %d, Rays/particle: %zu, Total rays: %d", 
                        MAX_PARTICLES, downsampled_angles_.size(), MAX_PARTICLES * static_cast<int>(downsampled_angles_.size()));
                }
                timing_stats_.reset();
            }
            
            visualize();
        }
        
        state_lock_.unlock();
    }
    
    // Always update steady time for next calculation
    last_steady_time = current_steady_time;
    
    // Publish TF and odometry
    if (map_initialized_) {
        Eigen::Vector3d current_pose = get_current_pose();
        rclcpp::Time timestamp = (has_odometry_for_motion && last_stamp_.nanoseconds() != 0) ? 
                                last_stamp_ : current_time;
        
        publish_tf(current_pose, timestamp);
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
    // Apply vehicle frame offset (lidar -> base_link)
    Eigen::Vector3d base_link_pose = utils::geometry::apply_vehicle_offset(pose, LIDAR_OFFSET_X);
    double base_link_x = base_link_pose[0];
    double base_link_y = base_link_pose[1];
    
    // Publish map → base_link transform
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = (stamp.nanoseconds() != 0) ? stamp : this->get_clock()->now();
    t.header.frame_id = "map";
    t.child_frame_id = "base_link";
    t.transform.translation.x = base_link_x;
    t.transform.translation.y = base_link_y;
    t.transform.translation.z = 0.0;
    t.transform.rotation = utils::geometry::yaw_to_quaternion(pose[2]);

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
        odom.pose.pose.orientation = utils::geometry::yaw_to_quaternion(pose[2]);
        odom.twist.twist.linear.x = current_velocity_;
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
        // Apply vehicle frame offset
        Eigen::Vector3d offset_pose = utils::geometry::apply_vehicle_offset(inferred_pose_, LIDAR_OFFSET_X);
        
        geometry_msgs::msg::PoseStamped ps;
        ps.header.stamp = this->get_clock()->now();
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
    
    for (int i = 0; i < offset_particles.rows(); ++i) {
        Eigen::Vector3d particle_pose(offset_particles(i, 0), offset_particles(i, 1), offset_particles(i, 2));
        Eigen::Vector3d offset_pose = utils::geometry::apply_vehicle_offset(particle_pose, LIDAR_OFFSET_X);
        offset_particles(i, 0) = offset_pose[0];
        offset_particles(i, 1) = offset_pose[1];
    }
    
    auto pa = utils::particles_to_pose_array(offset_particles);
    pa.header.stamp = this->get_clock()->now();
    pa.header.frame_id = "map";
    particle_pub_->publish(pa);
}



// --------------------------------- ODOMETRY-BASED TRACKING IMPLEMENTATION ---------------------------------
void ParticleFilter::initialize_odom_tracking(const Eigen::Vector3d& initial_pose, bool from_rviz)
{
    RCLCPP_INFO(this->get_logger(), "Odometry tracking init: [%.3f, %.3f, %.3f]", 
                initial_pose[0], initial_pose[1], initial_pose[2]);
    
    odom_pose_ = initial_pose;
    odom_reference_pose_ = initial_pose;
    
    if (last_pose_.norm() > 0) {
        odom_reference_odom_ = last_pose_;
    }
    
    pose_initialized_from_rviz_ = from_rviz;
    odom_tracking_active_ = true;
}

void ParticleFilter::update_odom_pose(const nav_msgs::msg::Odometry::SharedPtr& msg)
{
    if (!odom_tracking_active_) return;
    
    Eigen::Vector3d current_odom(msg->pose.pose.position.x, msg->pose.pose.position.y,
                                 utils::geometry::quaternion_to_yaw(msg->pose.pose.orientation));
    
    Eigen::Vector3d odom_delta = current_odom - odom_reference_odom_;
    odom_pose_ = odom_reference_pose_ + odom_delta;
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
