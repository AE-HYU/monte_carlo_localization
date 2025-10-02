// ================================================================================================
// PARAMETER MANAGER - Parameter initialization, validation, and dynamic reconfiguration
// ================================================================================================

#ifndef MCL_PKG__PARAMETER_MANAGER_HPP_
#define MCL_PKG__PARAMETER_MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>

namespace mcl_pkg
{

// Forward declaration
class MCL;

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
 * @param node Pointer to MCL node
 */
void initParameters(MCL* node);

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
 * @param node Pointer to MCL node
 * @param parameters Vector of parameter changes
 * @return Result indicating success/failure with reason
 */
rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
    MCL* node,
    const std::vector<rclcpp::Parameter> &parameters);

} // namespace parameter_manager
} // namespace mcl_pkg

#endif // MCL_PKG__PARAMETER_MANAGER_HPP_
