// ================================================================================================
// PARAMETER MANAGER - Parameter initialization, validation, and dynamic reconfiguration
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__PARAMETER_MANAGER_HPP_
#define PARTICLE_FILTER_CPP__PARAMETER_MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>

namespace particle_filter_cpp
{

// Forward declaration
class ParticleFilter;

namespace parameter_manager
{

/**
 * @brief Initialize and validate all MCL parameters with semantic checks
 *
 * Performs comprehensive parameter validation including:
 * - Range checks for particle counts, motion/sensor model parameters
 * - Automatic normalization of sensor model weights (must sum to 1.0)
 * - Frame name validation
 *
 * @param node Pointer to ParticleFilter node
 */
void initParameters(ParticleFilter* node);

/**
 * @brief Handle runtime parameter changes with validation
 *
 * Allows dynamic reconfiguration of:
 * - Motion model parameters (dispersion values)
 * - Sensor model parameters (weights, sigma) with auto-normalization
 * - Max range and visualization settings
 *
 * Automatically regenerates sensor model lookup table when needed.
 *
 * @param node Pointer to ParticleFilter node
 * @param parameters Vector of parameter changes
 * @return Result indicating success/failure with reason
 */
rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
    ParticleFilter* node,
    const std::vector<rclcpp::Parameter> &parameters);

} // namespace parameter_manager
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__PARAMETER_MANAGER_HPP_
