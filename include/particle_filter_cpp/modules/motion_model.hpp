#ifndef PARTICLE_FILTER_CPP__MODULES__MOTION_MODEL_HPP_
#define PARTICLE_FILTER_CPP__MODULES__MOTION_MODEL_HPP_

#include "particle.hpp"
#include "common_types.hpp"
#include <eigen3/Eigen/Dense>
#include <random>

namespace particle_filter_cpp
{
namespace modules
{

class MotionModel
{
public:
    MotionModel(const MotionModelParams& params, std::mt19937& rng);
    
    // Apply motion model to all particles
    void update_particles(ParticleSet& particles, 
                         const OdometryData& current_odom,
                         const OdometryData& previous_odom);
    
    // Apply motion model with explicit delta
    void update_particles_delta(ParticleSet& particles,
                               const Eigen::Vector3d& delta);
    
    // Compute odometry delta in local coordinate frame
    Eigen::Vector3d compute_odometry_delta(const OdometryData& current,
                                          const OdometryData& previous);
    
    // Update parameters
    void set_parameters(const MotionModelParams& params);
    const MotionModelParams& get_parameters() const { return params_; }

private:
    MotionModelParams params_;
    std::mt19937& rng_;
    std::normal_distribution<double> normal_dist_;
    
    // Apply motion with noise to single particle
    void apply_motion_with_noise(Particle& particle, const Eigen::Vector3d& delta);
    
    // Normalize angle to [-pi, pi]
    double normalize_angle(double angle);
};

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__MOTION_MODEL_HPP_