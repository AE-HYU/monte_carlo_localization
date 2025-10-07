// ================================================================================================
// ODOMETRY MOTION MODEL - Header
// ================================================================================================
// RTR (Rotation-Translation-Rotation) based motion model from Probabilistic Robotics
// Chapter 5.4, Table 5.6: sample_motion_model_odometry
// Reference: Thrun, Burgard, Fox - "Probabilistic Robotics" (2005), pp. 134-136
// ================================================================================================

#ifndef MCL_PKG__MOTION_MODEL__ODOMETRY_MOTION_MODEL_HPP_
#define MCL_PKG__MOTION_MODEL__ODOMETRY_MOTION_MODEL_HPP_

#include <Eigen/Dense>

namespace mcl_pkg {

// Forward declaration
class MCL;

namespace motion_model {

/**
 * @brief Applies RTR-based odometry motion model to particles
 *
 * Implements the sample_motion_model_odometry algorithm from
 * "Probabilistic Robotics" by Thrun, Burgard, and Fox (2005).
 *
 * Motion decomposition:
 * 1. δ_rot1:  Initial rotation to face translation direction
 * 2. δ_trans: Forward translation distance
 * 3. δ_rot2:  Final rotation to achieve target heading
 *
 * Noise model (motion-proportional):
 * - α1: rotation → rotation noise
 * - α2: translation → rotation noise
 * - α3: translation → translation noise
 * - α4: rotation → translation noise
 *
 * @param node Pointer to MCL node containing particles and alpha parameters
 * @param proposal_dist Particle poses (Nx3: x, y, theta) - modified in place
 * @param action Motion command (dx, dy, dtheta) in local frame
 */
void odometry_motion_update(MCL* node,
                            Eigen::MatrixXd& proposal_dist,
                            const Eigen::Vector3d& action);

} // namespace motion_model
} // namespace mcl_pkg

#endif  // MCL_PKG__MOTION_MODEL__ODOMETRY_MOTION_MODEL_HPP_
