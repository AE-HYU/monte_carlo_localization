#include "particle_filter_cpp/modules/resampling.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace particle_filter_cpp
{
namespace modules
{

ParticleResampler::ParticleResampler(const ResamplingParams& params, std::mt19937& rng)
    : params_(params), rng_(rng), uniform_dist_(0.0, 1.0), resample_count_(0)
{
}

bool ParticleResampler::needs_resampling(const WeightVector& weights)
{
    if (!params_.adaptive) {
        return true; // Always resample if not adaptive
    }
    
    double ess = compute_effective_sample_size(weights);
    double threshold = weights.size() * params_.ess_threshold_ratio;
    
    return ess < threshold;
}

void ParticleResampler::resample(ParticleSet& particles, WeightVector& weights)
{
    if (particles.empty() || weights.empty()) return;
    
    // Normalize weights first
    normalize_weights(weights);
    
    // Get resampling indices
    std::vector<int> indices;
    switch (params_.method) {
        case ResamplingMethod::SYSTEMATIC:
            indices = systematic_resampling(weights, particles.size());
            break;
        case ResamplingMethod::LOW_VARIANCE:
            indices = low_variance_resampling(weights, particles.size());
            break;
        case ResamplingMethod::MULTINOMIAL:
            indices = multinomial_resampling(weights, particles.size());
            break;
        case ResamplingMethod::STRATIFIED:
            indices = stratified_resampling(weights, particles.size());
            break;
    }
    
    // Create new particle set
    ParticleSet new_particles;
    new_particles.reserve(particles.size());
    
    for (int idx : indices) {
        new_particles.push_back(particles[idx]);
    }
    
    particles = std::move(new_particles);
    
    // Reset weights to uniform
    double uniform_weight = 1.0 / weights.size();
    std::fill(weights.begin(), weights.end(), uniform_weight);
    
    resample_count_++;
}

double ParticleResampler::compute_effective_sample_size(const WeightVector& weights)
{
    double sum_squared = 0.0;
    for (double weight : weights) {
        sum_squared += weight * weight;
    }
    return 1.0 / sum_squared;
}

void ParticleResampler::normalize_weights(WeightVector& weights)
{
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    
    if (sum > 1e-10) {
        for (double& weight : weights) {
            weight /= sum;
        }
    } else {
        // All weights are zero, set to uniform
        double uniform_weight = 1.0 / weights.size();
        std::fill(weights.begin(), weights.end(), uniform_weight);
    }
}

std::vector<int> ParticleResampler::systematic_resampling(const WeightVector& weights, int num_samples)
{
    std::vector<int> indices;
    indices.reserve(num_samples);
    
    auto cumsum = compute_cumulative_sum(weights);
    
    double step = 1.0 / num_samples;
    double start = uniform_dist_(rng_) * step;
    
    int current_index = 0;
    for (int i = 0; i < num_samples; ++i) {
        double target = start + i * step;
        
        while (current_index < static_cast<int>(cumsum.size()) - 1 && 
               cumsum[current_index] < target) {
            current_index++;
        }
        
        indices.push_back(current_index);
    }
    
    return indices;
}

std::vector<int> ParticleResampler::low_variance_resampling(const WeightVector& weights, int num_samples)
{
    // Low variance resampling is the same as systematic resampling
    return systematic_resampling(weights, num_samples);
}

std::vector<int> ParticleResampler::multinomial_resampling(const WeightVector& weights, int num_samples)
{
    std::vector<int> indices;
    indices.reserve(num_samples);
    
    auto cumsum = compute_cumulative_sum(weights);
    
    for (int i = 0; i < num_samples; ++i) {
        double rand_val = uniform_dist_(rng_);
        
        // Find the first index where cumsum > rand_val
        auto it = std::upper_bound(cumsum.begin(), cumsum.end(), rand_val);
        int index = std::distance(cumsum.begin(), it);
        
        // Clamp to valid range
        index = std::min(index, static_cast<int>(weights.size()) - 1);
        indices.push_back(index);
    }
    
    return indices;
}

std::vector<int> ParticleResampler::stratified_resampling(const WeightVector& weights, int num_samples)
{
    std::vector<int> indices;
    indices.reserve(num_samples);
    
    auto cumsum = compute_cumulative_sum(weights);
    
    double step = 1.0 / num_samples;
    
    for (int i = 0; i < num_samples; ++i) {
        double start = i * step;
        double end = (i + 1) * step;
        double rand_val = start + uniform_dist_(rng_) * (end - start);
        
        auto it = std::upper_bound(cumsum.begin(), cumsum.end(), rand_val);
        int index = std::distance(cumsum.begin(), it);
        
        index = std::min(index, static_cast<int>(weights.size()) - 1);
        indices.push_back(index);
    }
    
    return indices;
}

void ParticleResampler::set_parameters(const ResamplingParams& params)
{
    params_ = params;
}

std::vector<double> ParticleResampler::compute_cumulative_sum(const WeightVector& weights)
{
    std::vector<double> cumsum(weights.size());
    std::partial_sum(weights.begin(), weights.end(), cumsum.begin());
    return cumsum;
}

} // namespace modules
} // namespace particle_filter_cpp