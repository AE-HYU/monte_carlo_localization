// ================================================================================================
// ODOMETRY MOTION MODEL - Header
// ================================================================================================
// Odometry-based motion model with Gaussian noise
// Applies differential motion in robot's local frame with configurable noise parameters
// ================================================================================================

#ifndef MCL_PKG__MOTION_MODEL__ODOMETRY_MOTION_MODEL_HPP_
#define MCL_PKG__MOTION_MODEL__ODOMETRY_MOTION_MODEL_HPP_

#include <Eigen/Dense>

namespace mcl_pkg {

// Forward declaration
class MCL;

namespace motion_model {

/**
 * @brief Applies odometry-based motion model to particles with Gaussian noise
 *
 * Transforms particles according to odometry action with added noise:
 * - Converts local frame motion (dx, dy, dtheta) to global coordinates
 * - Adds Gaussian noise based on MOTION_DISPERSION parameters
 *
 * @param node Pointer to MCL node containing particles and parameters
 * @param proposal_dist Particle poses (Nx3: x, y, theta) - modified in place
 * @param action Motion command (dx, dy, dtheta) in local frame
 */
void odometry_motion_update(MCL* node,
                            Eigen::MatrixXd& proposal_dist,
                            const Eigen::Vector3d& action);

} // namespace motion_model
} // namespace mcl_pkg

#endif  // MCL_PKG__MOTION_MODEL__ODOMETRY_MOTION_MODEL_HPP_
