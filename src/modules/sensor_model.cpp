#include "particle_filter_cpp/modules/sensor_model.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdio>

// Conditional RangeLibc includes
#ifdef USE_RANGELIBC
#include "RangeLib.h"
#endif

namespace particle_filter_cpp
{
namespace modules
{

SensorModel::SensorModel(const SensorModelParams& params)
    : params_(params), map_initialized_(false), angle_step_(params.angle_step),
      sensor_model_precomputed_(false)
{
#ifdef USE_RANGELIBC
    rangelib_initialized_ = false;
#endif
}

void SensorModel::initialize_map(const MapInfo& map_info)
{
    map_info_ = map_info;
    map_initialized_ = true;
    
    // Precompute sensor model after map is loaded
    precompute_sensor_model();
}

void SensorModel::precompute_sensor_model()
{
    if (!map_initialized_) return;
    
    int max_range_px = static_cast<int>(params_.max_range / map_info_.resolution);
    int table_width = max_range_px + 1;
    
    sensor_model_table_.resize(table_width, std::vector<float>(table_width, 0.0f));
    
    // Precompute sensor model probabilities
    for (int d = 0; d < table_width; ++d) {
        float norm = 0.0f;
        
        for (int r = 0; r < table_width; ++r) {
            float prob = 0.0f;
            float z = static_cast<float>(r - d);
            
            // Hit probability (Gaussian around expected range)
            prob += params_.z_hit * std::exp(-(z * z) / (2.0f * params_.sigma_hit * params_.sigma_hit)) / 
                    (params_.sigma_hit * std::sqrt(2.0f * M_PI));
            
            // Short range probability
            if (r < d && d > 0) {
                prob += 2.0f * params_.z_short * (d - r) / static_cast<float>(d);
            }
            
            // Max range probability
            if (r == max_range_px) {
                prob += params_.z_max;
            }
            
            // Random measurement probability
            if (r < max_range_px) {
                prob += params_.z_rand / static_cast<float>(max_range_px);
            }
            
            norm += prob;
            sensor_model_table_[r][d] = prob;
        }
        
        // Normalize column
        if (norm > 0.0f) {
            for (int r = 0; r < table_width; ++r) {
                sensor_model_table_[r][d] /= norm;
            }
        }
    }
    
    sensor_model_precomputed_ = true;
    
#ifdef USE_RANGELIBC
    // Upload sensor model to RangeLibc if available
    if (range_method_) {
        try {
            range_method_->set_sensor_model(sensor_model_table_);
        } catch (const std::exception& e) {
            // RangeLibc doesn't support sensor model upload for this method
        }
    }
#endif
}

void SensorModel::update_weights(const ParticleSet& particles, WeightVector& weights,
                               const LaserScanData& scan_data)
{
    if (particles.size() != weights.size()) return;
    
    // Debug: Show RangeLibc status at the beginning
    static bool first_call = true;
    if (first_call) {
        printf("   SENSOR MODEL DEBUG:\n");
        printf("   - Map initialized: %s\n", map_initialized_ ? "YES" : "NO");
        printf("   - Sensor model precomputed: %s\n", sensor_model_precomputed_ ? "YES" : "NO");
#ifdef USE_RANGELIBC
        printf("   - RangeLibc compiled: YES\n");
        printf("   - RangeLibc initialized: %s\n", rangelib_initialized_ ? "YES" : "NO");
        printf("   - Range method available: %s\n", range_method_ ? "YES" : "NO");
#else
        printf("   - RangeLibc compiled: NO (using fallback)\n");
#endif
        printf("   - Angle step: %d\n", angle_step_);
        first_call = false;
    }
    
    double max_weight = 0.0;
    double min_weight = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < particles.size(); ++i) {
        weights[i] = compute_likelihood(particles[i], scan_data);
        max_weight = std::max(max_weight, weights[i]);
        min_weight = std::min(min_weight, weights[i]);
    }
    
    // Debug: Log weight statistics occasionally
    static int update_count = 0;
    update_count++;
    if (update_count % 100 == 0) {
        printf("Sensor model: min_weight=%.6f, max_weight=%.6f, ratio=%.6f\n", 
               min_weight, max_weight, max_weight > 0 ? min_weight / max_weight : 0.0);
    }
}

double SensorModel::compute_likelihood(const Particle& particle, 
                                     const LaserScanData& scan_data)
{
    if (!map_initialized_ || scan_data.ranges.empty()) {
        return 1.0;
    }
    
    // Downsample angles and ranges
    std::vector<float> downsampled_angles;
    std::vector<float> downsampled_ranges;
    
    for (size_t i = 0; i < scan_data.angles.size(); i += angle_step_) {
        if (i < scan_data.ranges.size()) {
            downsampled_angles.push_back(scan_data.angles[i]);
            downsampled_ranges.push_back(scan_data.ranges[i]);
        }
    }
    
    if (downsampled_angles.empty()) return 1.0;
    
    // Cast rays from particle pose
    std::vector<double> computed_ranges;
    
#ifdef USE_RANGELIBC
    if (rangelib_initialized_) {
        computed_ranges = cast_rays_rangelib(particle, downsampled_angles);
    } else {
        computed_ranges = cast_rays_fallback(particle, downsampled_angles);
    }
#else
    computed_ranges = cast_rays_fallback(particle, downsampled_angles);
#endif
    
    // Evaluate sensor model
    double likelihood = evaluate_sensor_model(downsampled_ranges, computed_ranges);
    
    // Apply squash factor (like Python reference)
    likelihood = std::pow(likelihood, 1.0 / params_.squash_factor);
    
    // Debug: Log occasionally for first few particles
    static int debug_count = 0;
    debug_count++;
    if (debug_count % 1000 == 0) {
        printf("Particle (%.2f, %.2f, %.2f): likelihood=%.6f, rangelib=%s\n", 
               particle.x, particle.y, particle.theta, likelihood,
#ifdef USE_RANGELIBC
               rangelib_initialized_ ? "yes" : "no"
#else
               "no"
#endif
               );
    }
    
    return likelihood;
}

#ifdef USE_RANGELIBC
void SensorModel::set_range_method(std::unique_ptr<RangeLib::RaycastBase> range_method)
{
    range_method_ = std::move(range_method);
    rangelib_initialized_ = (range_method_ != nullptr);
    
    if (rangelib_initialized_ && sensor_model_precomputed_) {
        try {
            range_method_->set_sensor_model(sensor_model_table_);
        } catch (const std::exception& e) {
            // Method doesn't support sensor model upload
        }
    }
}

std::vector<double> SensorModel::cast_rays_rangelib(const Particle& particle,
                                                  const std::vector<float>& angles)
{
    std::vector<double> ranges;
    
    if (!rangelib_initialized_ || angles.empty()) {
        return cast_rays_fallback(particle, angles);
    }
    
    try {
        // Prepare queries for RangeLibc
        size_t num_rays = angles.size();
        queries_.resize(num_rays * 3);
        computed_ranges_.resize(num_rays);
        
        // Set particle pose for all rays
        for (size_t i = 0; i < num_rays; ++i) {
            queries_[i * 3 + 0] = particle.x;
            queries_[i * 3 + 1] = particle.y;
            queries_[i * 3 + 2] = particle.theta + angles[i];
        }
        
        // Perform ray casting
        range_method_->calc_range_many(queries_, computed_ranges_);
        
        // Convert to double vector
        ranges.reserve(computed_ranges_.size());
        for (float range : computed_ranges_) {
            ranges.push_back(static_cast<double>(range) * map_info_.resolution);
        }
        
    } catch (const std::exception& e) {
        // Fallback to manual ray casting
        return cast_rays_fallback(particle, angles);
    }
    
    return ranges;
}
#endif

std::vector<double> SensorModel::cast_rays_fallback(const Particle& particle,
                                                  const std::vector<float>& angles)
{
    std::vector<double> ranges;
    ranges.reserve(angles.size());
    
    Eigen::Vector2d start(particle.x, particle.y);
    
    static int cast_count = 0;
    cast_count++;
    bool debug_this = (cast_count % 1000 == 0);
    
    if (debug_this) {
        printf("Fallback ray casting #%d for particle (%.2f, %.2f, %.2f), %zu angles\n",
               cast_count, particle.x, particle.y, particle.theta, angles.size());
    }
    
    for (size_t i = 0; i < angles.size(); ++i) {
        float angle_offset = angles[i];
        double absolute_angle = particle.theta + angle_offset;
        double range = cast_single_ray(start, absolute_angle);
        ranges.push_back(range);
        
        if (debug_this && i < 3) {
            printf("  Ray[%zu]: angle_offset=%.3f, abs_angle=%.3f, range=%.2f\n",
                   i, angle_offset, absolute_angle, range);
        }
    }
    
    return ranges;
}

double SensorModel::cast_single_ray(const Eigen::Vector2d& start, double angle)
{
    if (!map_initialized_) return params_.max_range;
    
    double step_size = map_info_.resolution / 2.0;
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    
    for (double range = 0.0; range < params_.max_range; range += step_size) {
        Eigen::Vector2d point = start + Eigen::Vector2d(range * cos_a, range * sin_a);
        
        if (!is_valid_point(point)) {
            return range;
        }
    }
    
    return params_.max_range;
}

double SensorModel::evaluate_sensor_model(const std::vector<float>& observed_ranges,
                                         const std::vector<double>& computed_ranges)
{
    if (!sensor_model_precomputed_ || observed_ranges.empty() || computed_ranges.empty()) {
        printf("Sensor model evaluation failed: precomputed=%s, obs_size=%zu, comp_size=%zu\n",
               sensor_model_precomputed_ ? "yes" : "no", observed_ranges.size(), computed_ranges.size());
        return 1.0;
    }
    
    size_t min_size = std::min(observed_ranges.size(), computed_ranges.size());
    double likelihood = 1.0;
    
    int max_range_px = static_cast<int>(params_.max_range / map_info_.resolution);
    
    // Debug: Check a few range comparisons
    static int eval_count = 0;
    eval_count++;
    bool debug_this = (eval_count % 500 == 0);
    
    for (size_t i = 0; i < min_size; ++i) {
        // Convert ranges to pixel units
        int obs_px = std::min(static_cast<int>(observed_ranges[i] / map_info_.resolution), max_range_px);
        int comp_px = std::min(static_cast<int>(computed_ranges[i] / map_info_.resolution), max_range_px);
        
        // Clamp to valid table indices
        obs_px = std::max(0, std::min(obs_px, static_cast<int>(sensor_model_table_.size()) - 1));
        comp_px = std::max(0, std::min(comp_px, static_cast<int>(sensor_model_table_[0].size()) - 1));
        
        double prob = sensor_model_table_[obs_px][comp_px];
        likelihood *= std::max(prob, 1e-6); // Avoid zero likelihood
        
        if (debug_this && i < 3) {
            printf("  Range[%zu]: obs=%.2f->%dpx, comp=%.2f->%dpx, prob=%.6f\n",
                   i, observed_ranges[i], obs_px, computed_ranges[i], comp_px, prob);
        }
    }
    
    if (debug_this) {
        printf("Sensor eval #%d: %zu ranges, final likelihood=%.6f\n", eval_count, min_size, likelihood);
    }
    
    return likelihood;
}

double SensorModel::get_sensor_model_prob(double observed, double computed)
{
    double prob = 0.0;
    double z = observed - computed;
    
    // Hit component (Gaussian)
    prob += params_.z_hit * std::exp(-(z * z) / (2.0 * params_.sigma_hit * params_.sigma_hit)) / 
            (params_.sigma_hit * std::sqrt(2.0 * M_PI));
    
    // Short component
    if (observed < computed) {
        prob += 2.0 * params_.z_short * (computed - observed) / computed;
    }
    
    // Max range component
    if (std::abs(observed - params_.max_range) < 0.01) {
        prob += params_.z_max;
    }
    
    // Random component
    if (observed < params_.max_range) {
        prob += params_.z_rand / params_.max_range;
    }
    
    return std::max(prob, 1e-6);
}

bool SensorModel::is_valid_point(const Eigen::Vector2d& point)
{
    if (!map_initialized_) return true;
    
    // Convert world coordinates to map coordinates
    double x_map = (point.x() - map_info_.origin.x()) / map_info_.resolution;
    double y_map = (point.y() - map_info_.origin.y()) / map_info_.resolution;
    
    int x = static_cast<int>(x_map);
    int y = static_cast<int>(y_map);
    
    if (x < 0 || x >= map_info_.width || y < 0 || y >= map_info_.height) {
        return false;
    }
    
    int index = y * map_info_.width + x;
    return map_info_.data[index] == 0; // 0 = free space
}

double SensorModel::normalize_angle(double angle)
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void SensorModel::set_parameters(const SensorModelParams& params)
{
    params_ = params;
    sensor_model_precomputed_ = false;
    
    if (map_initialized_) {
        precompute_sensor_model();
    }
}

} // namespace modules
} // namespace particle_filter_cpp