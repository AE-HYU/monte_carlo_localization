#ifndef PARTICLE_FILTER_CPP__MODULES__POSE_ESTIMATION_HPP_
#define PARTICLE_FILTER_CPP__MODULES__POSE_ESTIMATION_HPP_

#include "particle.hpp"
#include <eigen3/Eigen/Dense>

namespace particle_filter_cpp
{
namespace modules
{

enum class PoseEstimationMethod
{
    WEIGHTED_AVERAGE,
    MAX_WEIGHT,
    GAUSSIAN_MIXTURE,
    KERNEL_DENSITY
};

struct PoseEstimationParams
{
    PoseEstimationMethod method;
    double convergence_threshold;  // For clustering methods
    int max_iterations;           // For iterative methods
};

struct PoseStatistics
{
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    double max_weight;
    int max_weight_index;
    double effective_sample_size;
    bool converged;
};

class PoseEstimator
{
public:
    explicit PoseEstimator(const PoseEstimationParams& params = PoseEstimationParams{});
    
    // Compute expected pose from particle distribution
    Eigen::Vector3d compute_expected_pose(const ParticleSet& particles,
                                         const WeightVector& weights);
    
    // Compute pose statistics
    PoseStatistics compute_statistics(const ParticleSet& particles,
                                     const WeightVector& weights);
    
    // Compute covariance matrix
    Eigen::Matrix3d compute_covariance(const ParticleSet& particles,
                                      const WeightVector& weights,
                                      const Eigen::Vector3d& mean);
    
    // Different pose estimation methods
    Eigen::Vector3d weighted_average(const ParticleSet& particles,
                                    const WeightVector& weights);
    
    Eigen::Vector3d max_weight_pose(const ParticleSet& particles,
                                   const WeightVector& weights);
    
    Eigen::Vector3d gaussian_mixture_estimate(const ParticleSet& particles,
                                             const WeightVector& weights);
    
    // Utility functions
    double compute_particle_diversity(const ParticleSet& particles);
    bool is_converged(const ParticleSet& particles, const WeightVector& weights);
    
    // Update parameters
    void set_parameters(const PoseEstimationParams& params);
    const PoseEstimationParams& get_parameters() const { return params_; }

private:
    PoseEstimationParams params_;
    
    // Helper functions for circular statistics (handling angles)
    double compute_circular_mean(const std::vector<double>& angles,
                                const WeightVector& weights);
    double compute_circular_variance(const std::vector<double>& angles,
                                   const WeightVector& weights,
                                   double mean_angle);
    
    // Angle utilities
    double normalize_angle(double angle);
    double angle_difference(double a1, double a2);
};

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__POSE_ESTIMATION_HPP_