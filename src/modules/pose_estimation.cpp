#include "particle_filter_cpp/modules/pose_estimation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace particle_filter_cpp
{
namespace modules
{

PoseEstimator::PoseEstimator(const PoseEstimationParams& params)
    : params_(params)
{
    // Set default parameters if not provided
    if (params_.method == PoseEstimationParams::Method::WEIGHTED_AVERAGE) {
        // Already set
    }
}

Eigen::Vector3d PoseEstimator::compute_expected_pose(const ParticleSet& particles,
                                                   const WeightVector& weights)
{
    switch (params_.method) {
        case PoseEstimationParams::Method::WEIGHTED_AVERAGE:
            return weighted_average(particles, weights);
        case PoseEstimationParams::Method::MAX_LIKELIHOOD:
            return max_weight_pose(particles, weights);
        case PoseEstimationParams::Method::CLUSTERED_AVERAGE:
            return gaussian_mixture_estimate(particles, weights);
        default:
            return weighted_average(particles, weights);
    }
}

PoseStatistics PoseEstimator::compute_statistics(const ParticleSet& particles,
                                               const WeightVector& weights)
{
    PoseStatistics stats;
    
    if (particles.empty() || weights.empty()) {
        stats.mean = Eigen::Vector3d::Zero();
        stats.covariance = Eigen::Matrix3d::Identity();
        stats.max_weight = 0.0;
        stats.max_weight_index = -1;
        stats.effective_sample_size = 0.0;
        stats.converged = false;
        return stats;
    }
    
    // Compute mean pose
    stats.mean = weighted_average(particles, weights);
    
    // Compute covariance
    stats.covariance = compute_covariance(particles, weights, stats.mean);
    
    // Find max weight
    auto max_it = std::max_element(weights.begin(), weights.end());
    stats.max_weight = *max_it;
    stats.max_weight_index = std::distance(weights.begin(), max_it);
    
    // Compute effective sample size
    double sum_squared = 0.0;
    for (double weight : weights) {
        sum_squared += weight * weight;
    }
    stats.effective_sample_size = 1.0 / sum_squared;
    
    // Check convergence
    stats.converged = is_converged(particles, weights);
    
    return stats;
}

Eigen::Matrix3d PoseEstimator::compute_covariance(const ParticleSet& particles,
                                                const WeightVector& weights,
                                                const Eigen::Vector3d& mean)
{
    if (particles.empty()) {
        return Eigen::Matrix3d::Identity();
    }
    
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    
    // Compute weighted covariance for x, y
    for (size_t i = 0; i < particles.size(); ++i) {
        Eigen::Vector2d diff_xy(particles[i].x - mean[0], particles[i].y - mean[1]);
        covariance.block<2,2>(0,0) += weights[i] * (diff_xy * diff_xy.transpose());
    }
    
    // Handle angular component separately (circular statistics)
    std::vector<double> angles;
    for (const auto& particle : particles) {
        angles.push_back(particle.theta);
    }
    
    double circular_var = compute_circular_variance(angles, weights, mean[2]);
    covariance(2, 2) = circular_var;
    
    return covariance;
}

Eigen::Vector3d PoseEstimator::weighted_average(const ParticleSet& particles,
                                              const WeightVector& weights)
{
    if (particles.empty() || weights.empty()) {
        return Eigen::Vector3d::Zero();
    }
    
    double x_sum = 0.0, y_sum = 0.0;
    double cos_sum = 0.0, sin_sum = 0.0;
    
    for (size_t i = 0; i < particles.size(); ++i) {
        x_sum += particles[i].x * weights[i];
        y_sum += particles[i].y * weights[i];
        cos_sum += std::cos(particles[i].theta) * weights[i];
        sin_sum += std::sin(particles[i].theta) * weights[i];
    }
    
    double theta = std::atan2(sin_sum, cos_sum);
    return Eigen::Vector3d(x_sum, y_sum, theta);
}

Eigen::Vector3d PoseEstimator::max_weight_pose(const ParticleSet& particles,
                                             const WeightVector& weights)
{
    if (particles.empty() || weights.empty()) {
        return Eigen::Vector3d::Zero();
    }
    
    auto max_it = std::max_element(weights.begin(), weights.end());
    size_t max_index = std::distance(weights.begin(), max_it);
    
    const auto& best_particle = particles[max_index];
    return Eigen::Vector3d(best_particle.x, best_particle.y, best_particle.theta);
}

Eigen::Vector3d PoseEstimator::gaussian_mixture_estimate(const ParticleSet& particles,
                                                       const WeightVector& weights)
{
    // For simplicity, this implementation falls back to weighted average
    // A full Gaussian mixture model would require clustering and EM algorithm
    return weighted_average(particles, weights);
}

double PoseEstimator::compute_particle_diversity(const ParticleSet& particles)
{
    if (particles.size() < 2) return 0.0;
    
    double total_distance = 0.0;
    int count = 0;
    
    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = i + 1; j < particles.size(); ++j) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double dtheta = angle_difference(particles[i].theta, particles[j].theta);
            
            total_distance += std::sqrt(dx*dx + dy*dy + dtheta*dtheta);
            count++;
        }
    }
    
    return total_distance / count;
}

bool PoseEstimator::is_converged(const ParticleSet& particles, const WeightVector& weights)
{
    // Simple convergence check based on particle diversity
    double diversity = compute_particle_diversity(particles);
    return diversity < params_.convergence_threshold;
}

void PoseEstimator::set_parameters(const PoseEstimationParams& params)
{
    params_ = params;
}

double PoseEstimator::compute_circular_mean(const std::vector<double>& angles,
                                          const WeightVector& weights)
{
    double cos_sum = 0.0, sin_sum = 0.0;
    
    for (size_t i = 0; i < angles.size() && i < weights.size(); ++i) {
        cos_sum += std::cos(angles[i]) * weights[i];
        sin_sum += std::sin(angles[i]) * weights[i];
    }
    
    return std::atan2(sin_sum, cos_sum);
}

double PoseEstimator::compute_circular_variance(const std::vector<double>& angles,
                                              const WeightVector& weights,
                                              double mean_angle)
{
    double variance = 0.0;
    
    for (size_t i = 0; i < angles.size() && i < weights.size(); ++i) {
        double diff = angle_difference(angles[i], mean_angle);
        variance += weights[i] * diff * diff;
    }
    
    return variance;
}

double PoseEstimator::normalize_angle(double angle)
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double PoseEstimator::angle_difference(double a1, double a2)
{
    double diff = a1 - a2;
    return normalize_angle(diff);
}

} // namespace modules
} // namespace particle_filter_cpp