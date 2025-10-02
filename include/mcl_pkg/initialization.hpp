// ================================================================================================
// INITIALIZATION - Particle initialization functions
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__INITIALIZATION_HPP_
#define PARTICLE_FILTER_CPP__INITIALIZATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

namespace mcl_pkg
{

// Forward declaration
class MCL;

namespace initialization
{

/**
 * @brief Initialize particles uniformly across free space in map
 */
void initialize_global(MCL* node);

/**
 * @brief Initialize particles around a given pose with Gaussian noise
 */
void initialize_particles_pose(MCL* node, const Eigen::Vector3d &pose);

} // namespace initialization
} // namespace mcl_pkg

#endif
