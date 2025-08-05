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
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Check if update is needed
    if (!should_update(odom_data)) {
        return false;
    }
    
    // Perform MCL update steps
    motion_update(odom_data);
    sensor_update(scan_data);
    resampling_update();
    pose_estimation_update();
    
    // Update statistics and timing
    update_statistics();
    
    auto end_time = std::chrono::steady_clock::now();
    statistics_.computation_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
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
    motion_model_->update_particles(particles_, odom_data, last_odom_);
}

void MCLAlgorithm::sensor_update(const LaserScanData& scan_data)
{
    sensor_model_->update_weights(particles_, weights_, scan_data);
    
    // Normalize weights
    resampler_->normalize_weights(weights_);
}

void MCLAlgorithm::resampling_update()
{
    if (resampler_->needs_resampling(weights_)) {
        resampler_->resample(particles_, weights_);
        statistics_.resample_count = resampler_->get_resample_count();
    }
}

void MCLAlgorithm::pose_estimation_update()
{
    estimated_pose_ = pose_estimator_->compute_expected_pose(particles_, weights_);
}

bool MCLAlgorithm::should_update(const OdometryData& odom_data)
{
    if (statistics_.iteration_count == 0) {
        return true; // Always update on first iteration
    }
    
    // Check motion threshold
    double motion_delta = compute_motion_delta(odom_data, last_odom_);
    
    return motion_delta > params_.update_min_distance;
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

} // namespace modules
} // namespace particle_filter_cpp