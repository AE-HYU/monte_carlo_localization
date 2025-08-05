#include "particle_filter_cpp/particle_filter.hpp"
#include "particle_filter_cpp/utils.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <algorithm>
#include <numeric>

namespace particle_filter_cpp
{

ParticleFilter::ParticleFilter(const rclcpp::NodeOptions& options)
    : Node("particle_filter", options),
      gen_(std::random_device{}()),
      uniform_dist_(0.0, 1.0),
      normal_dist_(0.0, 1.0)
{
    // Declare parameters with default values
    this->declare_parameter("num_particles", 4000);
    this->declare_parameter("max_viz_particles", 60);
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
    this->declare_parameter("theta_discretization", 112);
    this->declare_parameter("publish_odom", true);
    this->declare_parameter("viz", true);

    // Get parameters
    num_particles_ = this->get_parameter("num_particles").as_int();
    max_viz_particles_ = this->get_parameter("max_viz_particles").as_int();
    max_range_ = this->get_parameter("max_range").as_double();
    motion_dispersion_x_ = this->get_parameter("motion_dispersion_x").as_double();
    motion_dispersion_y_ = this->get_parameter("motion_dispersion_y").as_double();
    motion_dispersion_theta_ = this->get_parameter("motion_dispersion_theta").as_double();
    z_hit_ = this->get_parameter("z_hit").as_double();
    z_short_ = this->get_parameter("z_short").as_double();
    z_max_ = this->get_parameter("z_max").as_double();
    z_rand_ = this->get_parameter("z_rand").as_double();
    sigma_hit_ = this->get_parameter("sigma_hit").as_double();
    scan_topic_ = this->get_parameter("scan_topic").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    range_method_ = this->get_parameter("range_method").as_string();
    theta_discretization_ = this->get_parameter("theta_discretization").as_int();
    publish_odom_ = this->get_parameter("publish_odom").as_bool();
    viz_ = this->get_parameter("viz").as_bool();

    // Initialize state
    map_initialized_ = false;
    odom_initialized_ = false;
    scan_initialized_ = false;
    particles_initialized_ = false;
    
#ifdef USE_RANGELIBC
    rangelib_initialized_ = false;
#endif

    // Initialize particles and weights
    particles_.resize(num_particles_);
    weights_.resize(num_particles_, 1.0 / num_particles_);
    current_pose_ = Eigen::Vector3d::Zero();

    // Initialize TF
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Initialize publishers
    particle_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/pf/viz/particles", 1);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/pf/viz/inferred_pose", 1);
    
    if (publish_odom_) {
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/pf/pose/odom", 1);
    }

    // Initialize subscribers
    laser_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic_, 1, std::bind(&ParticleFilter::laser_callback, this, std::placeholders::_1));
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 1, std::bind(&ParticleFilter::odom_callback, this, std::placeholders::_1));
    
    initial_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1, std::bind(&ParticleFilter::initial_pose_callback, this, std::placeholders::_1));

    // Initialize map service client
    map_client_ = this->create_client<nav_msgs::srv::GetMap>("/map_server/map");
    
    // Get the map
    get_map();

    RCLCPP_INFO(this->get_logger(), "Particle Filter initialized with %d particles", num_particles_);
}

void ParticleFilter::get_map()
{
    RCLCPP_INFO(this->get_logger(), "Requesting map from map server...");
    
    while (!map_client_->wait_for_service(std::chrono::seconds(1))) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for map service");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Map service not available, waiting...");
    }

    auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();
    auto future = map_client_->async_send_request(request);

    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS) {
        map_ = future.get()->map;
        max_range_px_ = static_cast<int>(max_range_ / map_.info.resolution);
        map_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "Map received: %dx%d, resolution: %.3f", 
                    map_.info.width, map_.info.height, map_.info.resolution);
        
#ifdef USE_RANGELIBC
        // Initialize RangeLibc
        initialize_rangelib();
        precompute_sensor_model();
#endif
        
        // Initialize particles uniformly across free space
        initialize_particles_uniform();
    } else {
        RCLCPP_ERROR(this->get_logger(), "Failed to get map from map server");
    }
}

void ParticleFilter::laser_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!scan_initialized_) {
        RCLCPP_INFO(this->get_logger(), "Received first laser scan");
        
#ifdef USE_RANGELIBC
        // Initialize laser angles for RangeLibc
        int num_beams = msg->ranges.size();
        laser_angles_.resize(num_beams);
        for (int i = 0; i < num_beams; ++i) {
            laser_angles_[i] = msg->angle_min + i * msg->angle_increment;
        }
        
        // Create downsampled angles (every 18th beam like Python version)
        int angle_step = 18;
        downsampled_angles_.clear();
        for (int i = 0; i < num_beams; i += angle_step) {
            downsampled_angles_.push_back(laser_angles_[i]);
        }
        
        RCLCPP_INFO(this->get_logger(), "Using %zu downsampled laser beams", downsampled_angles_.size());
#endif
        
        scan_initialized_ = true;
    }
    
    last_scan_ = *msg;
    
    // Update filter if all components are ready
    if (map_initialized_ && odom_initialized_ && particles_initialized_) {
        update_filter();
    }
}

void ParticleFilter::odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (!odom_initialized_) {
        RCLCPP_INFO(this->get_logger(), "Received first odometry");
        last_odom_ = *msg;
        odom_initialized_ = true;
        return;
    }
    
    // Apply motion model
    motion_model(*msg);
    last_odom_ = *msg;
}

void ParticleFilter::initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Received initial pose estimate");
    initialize_particles_pose(msg->pose.pose);
}

void ParticleFilter::motion_model(const nav_msgs::msg::Odometry& odom)
{
    if (!particles_initialized_) return;

    // Compute odometry delta
    double dx = odom.pose.pose.position.x - last_odom_.pose.pose.position.x;
    double dy = odom.pose.pose.position.y - last_odom_.pose.pose.position.y;
    double dtheta = utils::quaternion_to_yaw(odom.pose.pose.orientation) - 
                    utils::quaternion_to_yaw(last_odom_.pose.pose.orientation);
    dtheta = utils::normalize_angle(dtheta);

    // Apply motion model to each particle
    for (auto& particle : particles_) {
        // Add noise to motion
        double noisy_dx = dx + normal_dist_(gen_) * motion_dispersion_x_;
        double noisy_dy = dy + normal_dist_(gen_) * motion_dispersion_y_;
        double noisy_dtheta = dtheta + normal_dist_(gen_) * motion_dispersion_theta_;
        
        // Apply motion in particle's local coordinate frame
        double cos_theta = std::cos(particle.theta);
        double sin_theta = std::sin(particle.theta);
        
        particle.x += cos_theta * noisy_dx - sin_theta * noisy_dy;
        particle.y += sin_theta * noisy_dx + cos_theta * noisy_dy;
        particle.theta += noisy_dtheta;
        particle.theta = utils::normalize_angle(particle.theta);
    }
}

void ParticleFilter::sensor_model(const sensor_msgs::msg::LaserScan& scan)
{
    if (!particles_initialized_ || !map_initialized_) return;

    // Update particle weights based on laser scan
    for (size_t i = 0; i < particles_.size(); ++i) {
        weights_[i] = compute_likelihood(scan.ranges, particles_[i]);
    }
    
    normalize_weights();
}

void ParticleFilter::resampling()
{
    if (!particles_initialized_) return;

    // Use systematic resampling
    auto indices = utils::systematic_resampling(weights_, num_particles_);
    
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles_);
    
    for (int idx : indices) {
        new_particles.push_back(particles_[idx]);
    }
    
    particles_ = std::move(new_particles);
    
    // Reset weights to uniform
    std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
}

void ParticleFilter::update_filter()
{
    if (!scan_initialized_ || !particles_initialized_) return;

    // Apply sensor model
    sensor_model(last_scan_);
    
    // Check if resampling is needed
    double eff_sample_size = utils::compute_effective_sample_size(weights_);
    if (eff_sample_size < num_particles_ / 2.0) {
        resampling();
    }
    
    // Compute expected pose
    current_pose_ = compute_expected_pose();
    
    // Publish results
    if (viz_) {
        publish_particles();
        publish_pose();
    }
    publish_transform();
}

void ParticleFilter::initialize_particles_uniform()
{
    if (!map_initialized_) return;

    RCLCPP_INFO(this->get_logger(), "Initializing particles uniformly across free space");
    
    // Get free space points from map
    auto free_points = utils::get_free_space_points(map_);
    
    if (free_points.empty()) {
        RCLCPP_WARN(this->get_logger(), "No free space found in map, initializing particles at origin");
        for (auto& particle : particles_) {
            particle = Particle(0.0, 0.0, uniform_dist_(gen_) * 2.0 * M_PI);
        }
    } else {
        // Randomly sample from free space
        std::uniform_int_distribution<size_t> point_dist(0, free_points.size() - 1);
        
        for (auto& particle : particles_) {
            auto point = free_points[point_dist(gen_)];
            particle = Particle(point.x(), point.y(), uniform_dist_(gen_) * 2.0 * M_PI);
        }
    }
    
    particles_initialized_ = true;
    std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
    
    RCLCPP_INFO(this->get_logger(), "Particles initialized");
}

void ParticleFilter::initialize_particles_pose(const geometry_msgs::msg::Pose& pose)
{
    RCLCPP_INFO(this->get_logger(), "Initializing particles around given pose");
    
    double x = pose.position.x;
    double y = pose.position.y;
    double theta = utils::quaternion_to_yaw(pose.orientation);
    
    // Initialize particles with Gaussian distribution around given pose
    std::normal_distribution<double> x_dist(x, 0.5);
    std::normal_distribution<double> y_dist(y, 0.5);
    std::normal_distribution<double> theta_dist(theta, 0.4);
    
    for (auto& particle : particles_) {
        particle.x = x_dist(gen_);
        particle.y = y_dist(gen_);
        particle.theta = utils::normalize_angle(theta_dist(gen_));
        particle.weight = 1.0 / num_particles_;
    }
    
    particles_initialized_ = true;
    std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
}

Eigen::Vector3d ParticleFilter::compute_expected_pose()
{
    if (!particles_initialized_) return Eigen::Vector3d::Zero();

    double x_sum = 0.0, y_sum = 0.0;
    double cos_sum = 0.0, sin_sum = 0.0;
    
    for (size_t i = 0; i < particles_.size(); ++i) {
        x_sum += particles_[i].x * weights_[i];
        y_sum += particles_[i].y * weights_[i];
        cos_sum += std::cos(particles_[i].theta) * weights_[i];
        sin_sum += std::sin(particles_[i].theta) * weights_[i];
    }
    
    double theta = std::atan2(sin_sum, cos_sum);
    return Eigen::Vector3d(x_sum, y_sum, theta);
}

void ParticleFilter::publish_particles()
{
    if (particle_pub_->get_subscription_count() == 0) return;

    auto pose_array = utils::particles_to_pose_array(particles_);
    pose_array.header.stamp = this->get_clock()->now();
    pose_array.header.frame_id = "map";
    
    particle_pub_->publish(pose_array);
}

void ParticleFilter::publish_pose()
{
    if (pose_pub_->get_subscription_count() == 0) return;

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = this->get_clock()->now();
    pose_msg.header.frame_id = "map";
    pose_msg.pose.position.x = current_pose_[0];
    pose_msg.pose.position.y = current_pose_[1];
    pose_msg.pose.orientation = utils::yaw_to_quaternion(current_pose_[2]);
    
    pose_pub_->publish(pose_msg);
}

void ParticleFilter::publish_transform()
{
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = this->get_clock()->now();
    transform.header.frame_id = "map";
    transform.child_frame_id = "base_link";
    transform.transform.translation.x = current_pose_[0];
    transform.transform.translation.y = current_pose_[1];
    transform.transform.translation.z = 0.0;
    transform.transform.rotation = utils::yaw_to_quaternion(current_pose_[2]);
    
    tf_broadcaster_->sendTransform(transform);
}

double ParticleFilter::compute_likelihood(const std::vector<float>& ranges, const Particle& particle)
{
#ifdef USE_RANGELIBC
    if (!rangelib_initialized_ || downsampled_angles_.empty()) {
        return 1.0; // Fallback if RangeLibc not available
    }
    
    try {
        // Prepare queries for RangeLibc (x, y, theta for each ray)
        size_t num_rays = downsampled_angles_.size();
        queries_.resize(num_rays * 3);
        ranges_.resize(num_rays);
        
        // Set particle pose for all rays
        for (size_t i = 0; i < num_rays; ++i) {
            queries_[i * 3 + 0] = particle.x;                           // x
            queries_[i * 3 + 1] = particle.y;                           // y
            queries_[i * 3 + 2] = particle.theta + downsampled_angles_[i]; // theta
        }
        
        // Perform ray casting
        range_method_ptr_->calc_range_many(queries_, ranges_);
        
        // Get downsampled observed ranges
        std::vector<float> observed_ranges;
        int angle_step = 18;
        for (size_t i = 0; i < ranges.size(); i += angle_step) {
            if (i < ranges.size()) {
                observed_ranges.push_back(ranges[i]);
            }
        }
        
        // Ensure we have matching number of observed and computed ranges
        size_t min_size = std::min(observed_ranges.size(), ranges_.size());
        
        // Use RangeLibc's sensor model evaluation if available
        std::vector<float> weights(1, 0.0f);
        try {
            range_method_ptr_->eval_sensor_model(observed_ranges, ranges_, weights, min_size, 1);
            return static_cast<double>(weights[0]);
        } catch (const std::exception& e) {
            // Fallback to manual sensor model computation
            double likelihood = 1.0;
            for (size_t i = 0; i < min_size; ++i) {
                double obs_range = observed_ranges[i];
                double exp_range = ranges_[i];
                
                // Apply sensor model components
                double prob = 0.0;
                double z = obs_range - exp_range;
                
                // Hit component (Gaussian)
                prob += z_hit_ * std::exp(-(z * z) / (2.0 * sigma_hit_ * sigma_hit_)) / 
                        (sigma_hit_ * std::sqrt(2.0 * M_PI));
                
                // Short component
                if (obs_range < exp_range) {
                    prob += 2.0 * z_short_ * (exp_range - obs_range) / exp_range;
                }
                
                // Max range component
                if (std::abs(obs_range - max_range_) < 0.01) {
                    prob += z_max_;
                }
                
                // Random component
                if (obs_range < max_range_) {
                    prob += z_rand_ / max_range_;
                }
                
                likelihood *= std::max(prob, 1e-6); // Avoid zero likelihood
            }
            
            return likelihood;
        }
        
    } catch (const std::exception& e) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Error in RangeLibc likelihood computation: %s", e.what());
        return 1.0;
    }
#else
    // Fallback implementation without RangeLibc
    return particle_filter_cpp::utils::cast_rays(
        Eigen::Vector3d(particle.x, particle.y, particle.theta),
        std::vector<double>(downsampled_angles_.begin(), downsampled_angles_.end()),
        max_range_, map_).size() > 0 ? 1.0 : 0.1;
#endif
}

void ParticleFilter::normalize_weights()
{
    double sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    if (sum > 0.0) {
        for (auto& weight : weights_) {
            weight /= sum;
        }
    } else {
        std::fill(weights_.begin(), weights_.end(), 1.0 / num_particles_);
    }
}

#ifdef USE_RANGELIBC
void ParticleFilter::initialize_rangelib()
{
    RCLCPP_INFO(this->get_logger(), "Initializing RangeLibc with method: %s", range_method_.c_str());
    
    try {
        // Convert ROS map to RangeLibc format
        RangeLib::OccupancyGrid grid;
        grid.width = map_.info.width;
        grid.height = map_.info.height;
        grid.resolution = map_.info.resolution;
        grid.origin_x = map_.info.origin.position.x;
        grid.origin_y = map_.info.origin.position.y;
        
        // Copy map data
        grid.data.resize(map_.data.size());
        for (size_t i = 0; i < map_.data.size(); ++i) {
            grid.data[i] = static_cast<unsigned char>(map_.data[i]);
        }
        
        // Initialize the appropriate range method
        if (range_method_ == "bl") {
            range_method_ptr_ = std::make_unique<RangeLib::BresenhamsLine>(grid, max_range_px_);
        } else if (range_method_ == "cddt") {
            range_method_ptr_ = std::make_unique<RangeLib::CDDTCast>(grid, max_range_px_, theta_discretization_);
        } else if (range_method_ == "pcddt") {
            auto cddt_ptr = std::make_unique<RangeLib::CDDTCast>(grid, max_range_px_, theta_discretization_);
            cddt_ptr->prune();
            range_method_ptr_ = std::move(cddt_ptr);
        } else if (range_method_ == "rm") {
            range_method_ptr_ = std::make_unique<RangeLib::RayMarching>(grid, max_range_px_);
        } else if (range_method_ == "rmgpu") {
            range_method_ptr_ = std::make_unique<RangeLib::RayMarchingGPU>(grid, max_range_px_);
        } else if (range_method_ == "glt") {
            range_method_ptr_ = std::make_unique<RangeLib::GiantLUTCast>(grid, max_range_px_, theta_discretization_);
        } else {
            RCLCPP_WARN(this->get_logger(), "Unknown range method '%s', using 'cddt'", range_method_.c_str());
            range_method_ptr_ = std::make_unique<RangeLib::CDDTCast>(grid, max_range_px_, theta_discretization_);
        }
        
        rangelib_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "RangeLibc initialized successfully");
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize RangeLibc: %s", e.what());
        rangelib_initialized_ = false;
    }
}

void ParticleFilter::precompute_sensor_model()
{
    if (!rangelib_initialized_) return;
    
    RCLCPP_INFO(this->get_logger(), "Precomputing sensor model");
    
    // Create sensor model table
    int table_width = max_range_px_ + 1;
    std::vector<std::vector<float>> sensor_model_table(table_width, std::vector<float>(table_width, 0.0f));
    
    // Precompute sensor model probabilities
    for (int d = 0; d < table_width; ++d) {
        float norm = 0.0f;
        
        for (int r = 0; r < table_width; ++r) {
            float prob = 0.0f;
            float z = static_cast<float>(r - d);
            
            // Hit probability (Gaussian around expected range)
            prob += z_hit_ * std::exp(-(z * z) / (2.0f * sigma_hit_ * sigma_hit_)) / 
                    (sigma_hit_ * std::sqrt(2.0f * M_PI));
            
            // Short range probability
            if (r < d) {
                prob += 2.0f * z_short_ * (d - r) / static_cast<float>(d);
            }
            
            // Max range probability
            if (r == max_range_px_) {
                prob += z_max_;
            }
            
            // Random measurement probability
            if (r < max_range_px_) {
                prob += z_rand_ / static_cast<float>(max_range_px_);
            }
            
            norm += prob;
            sensor_model_table[r][d] = prob;
        }
        
        // Normalize column
        if (norm > 0.0f) {
            for (int r = 0; r < table_width; ++r) {
                sensor_model_table[r][d] /= norm;
            }
        }
    }
    
    // Upload sensor model to RangeLibc if supported
    try {
        range_method_ptr_->set_sensor_model(sensor_model_table);
        RCLCPP_INFO(this->get_logger(), "Sensor model uploaded to RangeLibc");
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "Could not upload sensor model to RangeLibc: %s", e.what());
    }
}
#endif

} // namespace particle_filter_cpp

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(particle_filter_cpp::ParticleFilter)