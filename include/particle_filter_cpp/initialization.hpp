// ================================================================================================
// INITIALIZATION - Particle initialization functions
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__INITIALIZATION_HPP_
#define PARTICLE_FILTER_CPP__INITIALIZATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

namespace particle_filter_cpp
{

// Forward declaration
class ParticleFilter;

namespace initialization
{

/**
 * @brief Initialize particles uniformly across free space in map
 */
void initialize_global(ParticleFilter* node);

/**
 * @brief Initialize particles around a given pose with Gaussian noise
 */
void initialize_particles_pose(ParticleFilter* node, const Eigen::Vector3d &pose);

} // namespace initialization
} // namespace particle_filter_cpp

#endif
