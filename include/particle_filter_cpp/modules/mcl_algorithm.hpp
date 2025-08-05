#ifndef PARTICLE_FILTER_CPP__MODULES__MCL_ALGORITHM_HPP_
#define PARTICLE_FILTER_CPP__MODULES__MCL_ALGORITHM_HPP_

#include "particle.hpp"
#include "initialization.hpp"
#include "motion_model.hpp"
#include "sensor_model.hpp"
#include "resampling.hpp"
#include "pose_estimation.hpp"
#include <memory>
#include <random>

namespace particle_filter_cpp
{
namespace modules
{

struct MCLParams
{
    int num_particles;
    MotionModelParams motion_params;
    SensorModelParams sensor_params;
    ResamplingParams resampling_params;
    PoseEstimationParams pose_params;
    
    // Algorithm parameters
    double update_min_distance;  // Minimum distance to trigger update
    double update_min_angle;     // Minimum angular change to trigger update
    int max_iterations;          // Maximum MCL iterations per update
    bool enable_timing;          // Enable performance timing
};

struct MCLStatistics
{
    double effective_sample_size;
    int resample_count;
    double pose_uncertainty;
    double computation_time_ms;
    bool converged;
    int iteration_count;
};

class MCLAlgorithm
{
public:
    explicit MCLAlgorithm(const MCLParams& params, int random_seed = 42);
    ~MCLAlgorithm() = default;
    
    // Initialize the particle filter
    void initialize_global(const MapInfo& map_info);
    void initialize_pose(const Eigen::Vector3d& pose, const Eigen::Vector3d& std_dev);
    
    // Main MCL update step
    bool update(const OdometryData& odom_data, const LaserScanData& scan_data);
    
    // Get current state
    Eigen::Vector3d get_estimated_pose() const { return estimated_pose_; }
    const ParticleSet& get_particles() const { return particles_; }
    const WeightVector& get_weights() const { return weights_; }
    MCLStatistics get_statistics() const { return statistics_; }
    
    // Parameter updates
    void set_params(const MCLParams& params);
    const MCLParams& get_params() const { return params_; }
    
    // State queries
    bool is_initialized() const { return initialized_; }
    bool needs_initialization() const;
    double get_pose_uncertainty() const;
    
    // Advanced features
    void set_map(const MapInfo& map_info);
    void reset();
    void enable_adaptive_parameters(bool enable) { adaptive_params_ = enable; }

private:
    MCLParams params_;
    bool initialized_;
    bool adaptive_params_;
    
    // Core components
    std::unique_ptr<ParticleInitializer> initializer_;
    std::unique_ptr<MotionModel> motion_model_;
    std::unique_ptr<SensorModel> sensor_model_;
    std::unique_ptr<ParticleResampler> resampler_;
    std::unique_ptr<PoseEstimator> pose_estimator_;
    
    // State
    ParticleSet particles_;
    WeightVector weights_;
    Eigen::Vector3d estimated_pose_;
    OdometryData last_odom_;
    MapInfo map_info_;
    
    // Statistics and timing
    MCLStatistics statistics_;
    std::chrono::steady_clock::time_point last_update_time_;
    
    // Random number generation
    std::mt19937 rng_;
    
    // Internal update steps
    void motion_update(const OdometryData& odom_data);
    void sensor_update(const LaserScanData& scan_data);
    void resampling_update();
    void pose_estimation_update();
    
    // Adaptive parameter adjustment
    void adapt_parameters();
    
    // Utility functions
    bool should_update(const OdometryData& odom_data);
    void update_statistics();
    double compute_motion_delta(const OdometryData& current, const OdometryData& previous);
};

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__MCL_ALGORITHM_HPP_