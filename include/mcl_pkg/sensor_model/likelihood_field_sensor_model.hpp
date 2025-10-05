// ================================================================================================
// LIKELIHOOD FIELD SENSOR MODEL - Header
// ================================================================================================
// Distance field based sensor model - no ray casting needed
// Multi-component model: z_hit (Gaussian) + z_short (Exponential) + z_max + z_rand
// ================================================================================================

#ifndef MCL_PKG__SENSOR_MODEL__LIKELIHOOD_FIELD_SENSOR_MODEL_HPP_
#define MCL_PKG__SENSOR_MODEL__LIKELIHOOD_FIELD_SENSOR_MODEL_HPP_

#include <vector>
#include <Eigen/Dense>
#include "nav_msgs/msg/occupancy_grid.hpp"

namespace mcl_pkg {

// Forward declaration
class MCL;

namespace sensor_model {

/**
 * @brief Precomputes distance field using efficient two-pass distance transform
 * Based on chamfer distance transform - O(N) complexity instead of O(N*M)
 *
 * @param node Pointer to MCL node
 */
void precompute_distance_field(MCL* node);

/**
 * @brief Precompute Gaussian likelihood lookup table for distance field
 *
 * @param node Pointer to MCL node
 */
void precompute_likelihood_lookup_table(MCL* node);

/**
 * @brief Likelihood field sensor model - no ray casting needed
 * Multi-component model: z_hit (Gaussian) + z_short (Exponential) + z_max + z_rand
 *
 * @param node Pointer to MCL node
 * @param proposal_dist Particle poses (Nx3: x, y, theta)
 * @param obs Observed laser ranges
 * @param weights Output particle weights
 */
void likelihood_field_sensor_update(MCL* node,
                                   const Eigen::MatrixXd& proposal_dist,
                                   const std::vector<float>& obs,
                                   std::vector<double>& weights);

} // namespace sensor_model
} // namespace mcl_pkg

#endif  // MCL_PKG__SENSOR_MODEL__LIKELIHOOD_FIELD_SENSOR_MODEL_HPP_
