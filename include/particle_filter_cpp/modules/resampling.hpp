#ifndef PARTICLE_FILTER_CPP__MODULES__RESAMPLING_HPP_
#define PARTICLE_FILTER_CPP__MODULES__RESAMPLING_HPP_

#include "particle.hpp"
#include <random>
#include <vector>

namespace particle_filter_cpp
{
namespace modules
{

enum class ResamplingMethod
{
    SYSTEMATIC,
    LOW_VARIANCE,
    MULTINOMIAL,
    STRATIFIED
};

struct ResamplingParams
{
    ResamplingMethod method;
    double ess_threshold_ratio;  // Resample when ESS < N * threshold
    bool adaptive;              // Use adaptive resampling
};

class ParticleResampler
{
public:
    ParticleResampler(const ResamplingParams& params, std::mt19937& rng);
    
    // Check if resampling is needed based on effective sample size
    bool needs_resampling(const WeightVector& weights);
    
    // Perform resampling (modifies particles and weights in-place)
    void resample(ParticleSet& particles, WeightVector& weights);
    
    // Compute effective sample size
    double compute_effective_sample_size(const WeightVector& weights);
    
    // Normalize weights
    void normalize_weights(WeightVector& weights);
    
    // Different resampling algorithms
    std::vector<int> systematic_resampling(const WeightVector& weights, int num_samples);
    std::vector<int> low_variance_resampling(const WeightVector& weights, int num_samples);
    std::vector<int> multinomial_resampling(const WeightVector& weights, int num_samples);
    std::vector<int> stratified_resampling(const WeightVector& weights, int num_samples);
    
    // Update parameters
    void set_parameters(const ResamplingParams& params);
    const ResamplingParams& get_parameters() const { return params_; }
    
    // Statistics
    int get_resample_count() const { return resample_count_; }
    void reset_statistics() { resample_count_ = 0; }

private:
    ResamplingParams params_;
    std::mt19937& rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    
    // Statistics
    int resample_count_;
    
    // Helper functions
    std::vector<double> compute_cumulative_sum(const WeightVector& weights);
    std::vector<int> sample_from_cumsum(const std::vector<double>& cumsum, 
                                       int num_samples,
                                       ResamplingMethod method);
};

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__RESAMPLING_HPP_