#include "particle_filter_cpp/modules/mcl_algorithm.hpp"
#include <chrono>
#include <cmath>

namespace particle_filter_cpp
{
namespace modules
{

MCLAlgorithm::MCLAlgorithm(const MCLParams& params, int random_seed)
    : params_(params), initialized_(false), adaptive_params_(false), rng_(random_seed)
{
    // Initialize all components
    initializer_ = std::make_unique<ParticleInitializer>(rng_);
    motion_model_ = std::make_unique<MotionModel>(params_.motion_params, rng_);
    sensor_model_ = std::make_unique<SensorModel>(params_.sensor_params);
    resampler_ = std::make_unique<ParticleResampler>(params_.resampling_params, rng_);
    pose_estimator_ = std::make_unique<PoseEstimator>(params_.pose_params);
    
    // Initialize particle containers
    particles_.resize(params_.num_particles);
    weights_.resize(params_.num_particles, 1.0 / params_.num_particles);
    estimated_pose_ = Eigen::Vector3d::Zero();
    
    // Initialize statistics
    statistics_.effective_sample_size = params_.num_particles;
    statistics_.resample_count = 0;
    statistics_.pose_uncertainty = 1.0;
    statistics_.computation_time_ms = 0.0;
    statistics_.converged = false;
    statistics_.iteration_count = 0;
    
    last_update_time_ = std::chrono::steady_clock::now();
}

void MCLAlgorithm::initialize_global(const MapInfo& map_info)
{
    map_info_ = map_info;
    
    // Initialize sensor model with map
    sensor_model_->initialize_map(map_info);
    
    // Initialize RangeLibc if available
#ifdef USE_RANGELIBC
    initialize_rangelibc();
#endif
    
    // Initialize particles globally
    initializer_->initialize_global(particles_, weights_, map_info);
    
    initialized_ = true;
    
    // Reset statistics
    statistics_.resample_count = 0;
    statistics_.iteration_count = 0;
}

void MCLAlgorithm::initialize_pose(const Eigen::Vector3d& pose, const Eigen::Vector3d& std_dev)
{
    if (!initialized_) {
        // Need map info for sensor model
        return;
    }
    
    // Initialize particles around given pose
    initializer_->initialize_pose(particles_, weights_, pose, std_dev);
    
    // Reset statistics
    statistics_.resample_count = 0;
    statistics_.iteration_count = 0;
}

bool MCLAlgorithm::update(const OdometryData& odom_data, const LaserScanData& scan_data)
{
    if (!initialized_) {
        printf("MCL: Not initialized, skipping update\n");
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Check if update is needed
    if (!should_update(odom_data)) {
        printf("MCL: Update not needed, skipping\n");
        return false;
    }
    
    printf("\n=== MCL UPDATE #%lu ===\n", statistics_.iteration_count);
    
    // Before update: show current state
    printf("BEFORE - ESS: %.1f/%d, pose: (%.2f, %.2f, %.2f)\n", 
           statistics_.effective_sample_size, params_.num_particles,
           estimated_pose_[0], estimated_pose_[1], estimated_pose_[2]);
    
    // Show weight distribution before resampling
    double min_weight = *std::min_element(weights_.begin(), weights_.end());
    double max_weight = *std::max_element(weights_.begin(), weights_.end());
    double sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    printf("WEIGHTS - min: %.6f, max: %.6f, sum: %.6f\n", min_weight, max_weight, sum_weights);
    
    // Perform MCL update steps in Python order:
    // 1. RESAMPLE FIRST (like Python)
    printf("1. RESAMPLING...\n");
    resampling_update();
    
    // 2. Motion model
    printf("2. MOTION MODEL...\n");
    motion_update(odom_data);
    
    // 3. Sensor model  
    printf("3. SENSOR MODEL...\n");
    sensor_update(scan_data);
    
    // 4. Pose estimation
    printf("4. POSE ESTIMATION...\n");
    pose_estimation_update();
    
    // Update statistics and timing
    update_statistics();
    
    auto end_time = std::chrono::steady_clock::now();
    statistics_.computation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // After update: show results
    printf("AFTER - ESS: %.1f/%d, pose: (%.2f, %.2f, %.2f)\n", 
           statistics_.effective_sample_size, params_.num_particles,
           estimated_pose_[0], estimated_pose_[1], estimated_pose_[2]);
    
    // Show final weight distribution
    min_weight = *std::min_element(weights_.begin(), weights_.end());
    max_weight = *std::max_element(weights_.begin(), weights_.end());
    sum_weights = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    printf("FINAL WEIGHTS - min: %.6f, max: %.6f, sum: %.6f\n", min_weight, max_weight, sum_weights);
    printf("Computation time: %.2f ms\n", statistics_.computation_time_ms);
    printf("=== END MCL UPDATE ===\n\n");
    
    // Adaptive parameter adjustment
    if (adaptive_params_) {
        adapt_parameters();
    }
    
    last_odom_ = odom_data;
    last_update_time_ = end_time;
    statistics_.iteration_count++;
    
    return true;
}

void MCLAlgorithm::motion_update(const OdometryData& odom_data)
{
    // Only apply motion if we have valid previous odometry data
    // Skip motion update on first call (like Python reference implementation)
    if (statistics_.iteration_count > 0 && last_odom_.timestamp > 0.0) {
        // Compute odometry delta
        auto delta = motion_model_->compute_odometry_delta(odom_data, last_odom_);
        printf("   Motion delta: (%.4f, %.4f, %.4f)\n", delta[0], delta[1], delta[2]);
        
        // Show a few particles before motion
        printf("   Before motion - P[0]: (%.2f, %.2f, %.2f), P[1]: (%.2f, %.2f, %.2f)\n",
               particles_[0].x, particles_[0].y, particles_[0].theta,
               particles_[1].x, particles_[1].y, particles_[1].theta);
        
        motion_model_->update_particles(particles_, odom_data, last_odom_);
        
        // Show same particles after motion
        printf("   After motion  - P[0]: (%.2f, %.2f, %.2f), P[1]: (%.2f, %.2f, %.2f)\n",
               particles_[0].x, particles_[0].y, particles_[0].theta,
               particles_[1].x, particles_[1].y, particles_[1].theta);
    } else {
        printf("   Skipping motion update (iteration: %lu, last_timestamp: %.3f)\n", 
               statistics_.iteration_count, last_odom_.timestamp);
    }
}

void MCLAlgorithm::sensor_update(const LaserScanData& scan_data)
{
    // Show scan data info
    printf("   Scan data: %zu ranges, angle_min=%.3f, angle_max=%.3f\n", 
           scan_data.ranges.size(), scan_data.angle_min, scan_data.angle_max);
    
    // Show weights before sensor model
    double sum_before = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    printf("   Weights before sensor: sum=%.6f\n", sum_before);
    
    sensor_model_->update_weights(particles_, weights_, scan_data);
    
    // Show weights after sensor model but before normalization
    double sum_after = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    double min_weight = *std::min_element(weights_.begin(), weights_.end());
    double max_weight = *std::max_element(weights_.begin(), weights_.end());
    printf("   Weights after sensor: sum=%.6f, min=%.6f, max=%.6f\n", 
           sum_after, min_weight, max_weight);
    
    // Normalize weights
    resampler_->normalize_weights(weights_);
    
    // Show final normalized weights
    double sum_normalized = std::accumulate(weights_.begin(), weights_.end(), 0.0);
    printf("   Weights after normalize: sum=%.6f\n", sum_normalized);
}

void MCLAlgorithm::resampling_update()
{
    double ess = resampler_->compute_effective_sample_size(weights_);
    bool needs_resample = resampler_->needs_resampling(weights_);
    printf("   ESS: %.1f/%d, needs_resample: %s\n", 
           ess, params_.num_particles, needs_resample ? "YES" : "NO");
    
    if (needs_resample) {
        // Show some particles before resampling
        printf("   Before resample - P[0]: (%.2f, %.2f, %.2f) w=%.6f\n",
               particles_[0].x, particles_[0].y, particles_[0].theta, weights_[0]);
        printf("   Before resample - P[1]: (%.2f, %.2f, %.2f) w=%.6f\n",
               particles_[1].x, particles_[1].y, particles_[1].theta, weights_[1]);
        
        resampler_->resample(particles_, weights_);
        statistics_.resample_count = resampler_->get_resample_count();
        
        // Show same particles after resampling
        printf("   After resample  - P[0]: (%.2f, %.2f, %.2f) w=%.6f\n",
               particles_[0].x, particles_[0].y, particles_[0].theta, weights_[0]);
        printf("   After resample  - P[1]: (%.2f, %.2f, %.2f) w=%.6f\n",
               particles_[1].x, particles_[1].y, particles_[1].theta, weights_[1]);
    }
}

void MCLAlgorithm::pose_estimation_update()
{
    estimated_pose_ = pose_estimator_->compute_expected_pose(particles_, weights_);
}

bool MCLAlgorithm::should_update(const OdometryData& odom_data)
{
    (void)odom_data;  // Suppress unused parameter warning
    // Always update like the Python reference code
    return true;
}

double MCLAlgorithm::compute_motion_delta(const OdometryData& current, const OdometryData& previous)
{
    Eigen::Vector3d delta = motion_model_->compute_odometry_delta(current, previous);
    
    // Compute magnitude of motion
    double linear_delta = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
    double angular_delta = std::abs(delta[2]);
    
    return linear_delta + angular_delta * 0.5; // Weight angular motion less
}

void MCLAlgorithm::update_statistics()
{
    // Compute effective sample size
    statistics_.effective_sample_size = resampler_->compute_effective_sample_size(weights_);
    
    // Compute pose uncertainty (trace of covariance matrix)
    auto pose_stats = pose_estimator_->compute_statistics(particles_, weights_);
    statistics_.pose_uncertainty = pose_stats.covariance.trace();
    statistics_.converged = pose_stats.converged;
}

void MCLAlgorithm::adapt_parameters()
{
    // Adaptive parameter adjustment based on performance
    double ess_ratio = statistics_.effective_sample_size / params_.num_particles;
    
    // Adjust resampling threshold based on ESS
    if (ess_ratio < 0.25) {
        // Low diversity, reduce resampling threshold
        auto resampling_params = resampler_->get_parameters();
        resampling_params.ess_threshold_ratio = std::max(0.1, resampling_params.ess_threshold_ratio * 0.9);
        resampler_->set_parameters(resampling_params);
    } else if (ess_ratio > 0.75) {
        // High diversity, increase resampling threshold
        auto resampling_params = resampler_->get_parameters();
        resampling_params.ess_threshold_ratio = std::min(0.9, resampling_params.ess_threshold_ratio * 1.1);
        resampler_->set_parameters(resampling_params);
    }
    
    // Adjust motion noise based on convergence
    if (statistics_.converged) {
        // Reduce motion noise when converged
        auto motion_params = motion_model_->get_parameters();
        motion_params.dispersion_x *= 0.95;
        motion_params.dispersion_y *= 0.95;
        motion_params.dispersion_theta *= 0.95;
        motion_model_->set_parameters(motion_params);
    } else if (statistics_.pose_uncertainty > 10.0) {
        // Increase motion noise when uncertain
        auto motion_params = motion_model_->get_parameters();
        motion_params.dispersion_x *= 1.05;
        motion_params.dispersion_y *= 1.05;
        motion_params.dispersion_theta *= 1.05;
        motion_model_->set_parameters(motion_params);
    }
}

bool MCLAlgorithm::needs_initialization() const
{
    if (!initialized_) return true;
    
    // Check if particles are too dispersed or all weights are too low
    double max_weight = *std::max_element(weights_.begin(), weights_.end());
    
    return (statistics_.effective_sample_size < params_.num_particles * 0.1) || 
           (max_weight < 1e-6);
}

double MCLAlgorithm::get_pose_uncertainty() const
{
    return statistics_.pose_uncertainty;
}

void MCLAlgorithm::set_map(const MapInfo& map_info)
{
    map_info_ = map_info;
    sensor_model_->initialize_map(map_info);
}

void MCLAlgorithm::reset()
{
    initialized_ = false;
    estimated_pose_ = Eigen::Vector3d::Zero();
    
    // Reset weights to uniform
    double uniform_weight = 1.0 / params_.num_particles;
    std::fill(weights_.begin(), weights_.end(), uniform_weight);
    
    // Reset statistics
    statistics_.effective_sample_size = params_.num_particles;
    statistics_.resample_count = 0;
    statistics_.pose_uncertainty = 1.0;
    statistics_.computation_time_ms = 0.0;
    statistics_.converged = false;
    statistics_.iteration_count = 0;
    
    // Reset resampler statistics
    resampler_->reset_statistics();
}

void MCLAlgorithm::set_params(const MCLParams& params)
{
    params_ = params;
    
    // Update component parameters
    motion_model_->set_parameters(params_.motion_params);
    sensor_model_->set_parameters(params_.sensor_params);
    resampler_->set_parameters(params_.resampling_params);
    pose_estimator_->set_parameters(params_.pose_params);
    
    // Resize particle containers if needed
    if (static_cast<int>(particles_.size()) != params_.num_particles) {
        particles_.resize(params_.num_particles);
        weights_.resize(params_.num_particles, 1.0 / params_.num_particles);
        initialized_ = false; // Need to reinitialize with new particle count
    }
}

#ifdef USE_RANGELIBC
void MCLAlgorithm::initialize_rangelibc()
{
    if (!map_info_.data.empty()) {
        printf("Initializing RangeLibc with map: %dx%d, resolution=%.3f\n", 
               map_info_.width, map_info_.height, map_info_.resolution);
        
        // Create RangeLibc map
        int max_range_px = static_cast<int>(params_.sensor_params.max_range / map_info_.resolution);
        printf("Max range pixels: %d\n", max_range_px);
        
        // Initialize range method based on configuration
        std::string range_method = params_.sensor_params.range_method;
        printf("Range method: %s\n", range_method.c_str());
        
        if (range_method == "cddt" || range_method == "pcddt") {
            auto cddt_method = std::make_unique<RangeLib::CDDTCast>(
                map_info_.data.data(), 
                map_info_.width, 
                map_info_.height,
                map_info_.resolution,
                max_range_px,
                params_.sensor_params.theta_discretization
            );
            
            if (range_method == "pcddt") {
                cddt_method->prune();
            }
            
            sensor_model_->set_range_method(std::move(cddt_method));
            printf("RangeLibc CDDT initialized successfully\n");
        } else if (range_method == "rm") {
            auto rm_method = std::make_unique<RangeLib::RayMarching>(
                map_info_.data.data(),
                map_info_.width,
                map_info_.height,
                map_info_.resolution,
                max_range_px
            );
            sensor_model_->set_range_method(std::move(rm_method));
            printf("RangeLibc RayMarching initialized successfully\n");
        } else if (range_method == "bl") {
            auto bl_method = std::make_unique<RangeLib::BresenhamsLine>(
                map_info_.data.data(),
                map_info_.width,
                map_info_.height,
                map_info_.resolution,
                max_range_px
            );
            sensor_model_->set_range_method(std::move(bl_method));
            printf("RangeLibc Bresenham initialized successfully\n");
        } else {
            printf("Unknown range method: %s, using fallback\n", range_method.c_str());
        }
    } else {
        printf("No map data available for RangeLibc initialization\n");
    }
}
#endif

} // namespace modules
} // namespace particle_filter_cpp