#include "particle_filter_cpp/modules/motion_model.hpp"
#include <cmath>

namespace particle_filter_cpp
{
namespace modules
{

MotionModel::MotionModel(const MotionModelParams& params, std::mt19937& rng)
    : params_(params), rng_(rng), normal_dist_(0.0, 1.0)
{
}

void MotionModel::update_particles(ParticleSet& particles, 
                                 const OdometryData& current_odom,
                                 const OdometryData& previous_odom)
{
    Eigen::Vector3d delta = compute_odometry_delta(current_odom, previous_odom);
    update_particles_delta(particles, delta);
}

void MotionModel::update_particles_delta(ParticleSet& particles,
                                       const Eigen::Vector3d& delta)
{
    for (auto& particle : particles) {
        apply_motion_with_noise(particle, delta);
    }
}

Eigen::Vector3d MotionModel::compute_odometry_delta(const OdometryData& current,
                                                  const OdometryData& previous)
{
    // Compute position change
    Eigen::Vector2d position_delta = current.pose.head<2>() - previous.pose.head<2>();
    
    // Transform to local coordinate frame of previous pose
    double prev_theta = previous.pose[2];
    double cos_theta = std::cos(-prev_theta);
    double sin_theta = std::sin(-prev_theta);
    
    Eigen::Matrix2d rotation;
    rotation << cos_theta, -sin_theta,
                sin_theta,  cos_theta;
    
    Eigen::Vector2d local_delta = rotation * position_delta;
    
    // Compute angular change
    double theta_delta = normalize_angle(current.pose[2] - previous.pose[2]);
    
    return Eigen::Vector3d(local_delta[0], local_delta[1], theta_delta);
}

void MotionModel::apply_motion_with_noise(Particle& particle, const Eigen::Vector3d& delta)
{
    // Add noise to the motion delta
    double noisy_dx = delta[0] + normal_dist_(rng_) * params_.dispersion_x;
    double noisy_dy = delta[1] + normal_dist_(rng_) * params_.dispersion_y;
    double noisy_dtheta = delta[2] + normal_dist_(rng_) * params_.dispersion_theta;
    
    // Apply motion in particle's local coordinate frame
    double cos_theta = std::cos(particle.theta);
    double sin_theta = std::sin(particle.theta);
    
    // Transform local motion to global coordinates
    particle.x += cos_theta * noisy_dx - sin_theta * noisy_dy;
    particle.y += sin_theta * noisy_dx + cos_theta * noisy_dy;
    particle.theta += noisy_dtheta;
    
    // Normalize angle
    particle.theta = normalize_angle(particle.theta);
}

double MotionModel::normalize_angle(double angle)
{
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void MotionModel::set_parameters(const MotionModelParams& params)
{
    params_ = params;
}

} // namespace modules
} // namespace particle_filter_cpp