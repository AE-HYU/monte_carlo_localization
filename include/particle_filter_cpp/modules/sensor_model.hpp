#ifndef PARTICLE_FILTER_CPP__MODULES__SENSOR_MODEL_HPP_
#define PARTICLE_FILTER_CPP__MODULES__SENSOR_MODEL_HPP_

#include "particle.hpp"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>

// Forward declarations for RangeLibc (optional)
#ifdef USE_RANGELIBC
namespace RangeLib {
    class RaycastBase;
}
#endif

namespace particle_filter_cpp
{
namespace modules
{

struct SensorModelParams
{
    double z_hit;     // Weight for correct measurements
    double z_short;   // Weight for short measurements  
    double z_max;     // Weight for max range measurements
    double z_rand;    // Weight for random measurements
    double sigma_hit; // Standard deviation for hit measurements
    double max_range; // Maximum sensor range
};

struct LaserScanData
{
    std::vector<float> ranges;
    std::vector<float> angles;
    double max_range;
    double angle_min;
    double angle_max;
    double angle_increment;
    double timestamp;
};

class SensorModel
{
public:
    SensorModel(const SensorModelParams& params);
    ~SensorModel() = default;
    
    // Initialize with map data
    void initialize_map(const MapInfo& map_info);
    
    // Precompute sensor model lookup table
    void precompute_sensor_model();
    
    // Update particle weights based on laser scan
    void update_weights(const ParticleSet& particles, WeightVector& weights,
                       const LaserScanData& scan_data);
    
    // Compute likelihood for single particle
    double compute_likelihood(const Particle& particle, 
                             const LaserScanData& scan_data);
    
    // Set range casting method (RangeLibc integration)
#ifdef USE_RANGELIBC
    void set_range_method(std::unique_ptr<RangeLib::RaycastBase> range_method);
#endif
    
    // Update parameters
    void set_parameters(const SensorModelParams& params);
    const SensorModelParams& get_parameters() const { return params_; }
    
    // Set downsampling parameters
    void set_angle_step(int step) { angle_step_ = step; }

private:
    SensorModelParams params_;
    MapInfo map_info_;
    bool map_initialized_;
    int angle_step_;  // Downsample every N-th laser beam
    
    // Sensor model lookup table
    std::vector<std::vector<float>> sensor_model_table_;
    bool sensor_model_precomputed_;
    
    // RangeLibc integration
#ifdef USE_RANGELIBC
    std::unique_ptr<RangeLib::RaycastBase> range_method_;
    std::vector<float> queries_;
    std::vector<float> computed_ranges_;
    bool rangelib_initialized_;
#endif
    
    // Ray casting methods
    std::vector<double> cast_rays_rangelib(const Particle& particle,
                                          const std::vector<float>& angles);
    std::vector<double> cast_rays_fallback(const Particle& particle,
                                          const std::vector<float>& angles);
    double cast_single_ray(const Eigen::Vector2d& start, double angle);
    
    // Sensor model computation
    double evaluate_sensor_model(const std::vector<float>& observed_ranges,
                                const std::vector<double>& computed_ranges);
    double get_sensor_model_prob(double observed, double computed);
    
    // Utility functions
    bool is_valid_point(const Eigen::Vector2d& point);
    double normalize_angle(double angle);
};

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__SENSOR_MODEL_HPP_