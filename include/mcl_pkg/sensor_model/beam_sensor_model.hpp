// ================================================================================================
// BEAM SENSOR MODEL - Header
// ================================================================================================
// Beam-based range finder model with ray casting and multi-component noise model
// Components: z_hit (Gaussian), z_short (Exponential), z_max (Delta), z_rand (Uniform)
// ================================================================================================

#ifndef MCL_PKG__SENSOR_MODEL__BEAM_SENSOR_MODEL_HPP_
#define MCL_PKG__SENSOR_MODEL__BEAM_SENSOR_MODEL_HPP_

#include <vector>
#include <Eigen/Dense>
#include "nav_msgs/msg/occupancy_grid.hpp"

namespace mcl_pkg {

// Forward declaration
class MCL;

namespace sensor_model {

/**
 * @brief Precomputes beam sensor model lookup table for fast likelihood evaluation
 *
 * Generates probability table for beam sensor model:
 * P(z|z_exp) = z_hit*N(z;z_exp,σ) + z_short*λ*exp(-λ*z) + z_max*δ(z-z_max) + z_rand*U(0,z_max)
 *
 * @param node Pointer to MCL node containing sensor model parameters
 */
void precompute_beam_sensor_model(MCL* node);

/**
 * @brief Calculate particle weights using beam sensor model with ray casting
 *
 * Performs ray casting from each particle and evaluates observation likelihood
 * using pre-computed lookup table
 *
 * @param node Pointer to MCL node
 * @param proposal_dist Particle poses (Nx3: x, y, theta)
 * @param obs Observed laser ranges
 * @param weights Output particle weights
 */
void beam_sensor_update(MCL* node,
                       const Eigen::MatrixXd& proposal_dist,
                       const std::vector<float>& obs,
                       std::vector<double>& weights);

/**
 * @brief Initialize sensor arrays for beam model computation
 */
void initialize_beam_sensor_arrays(MCL* node, int num_rays, int total_queries);

/**
 * @brief Generate ray queries for batch ray casting
 */
void generate_beam_ray_queries(MCL* node,
                               const Eigen::MatrixXd& proposal_dist,
                               int num_rays);

/**
 * @brief Calculate particle weights using lookup table
 */
void calculate_beam_particle_weights(MCL* node,
                                     const std::vector<float>& obs,
                                     int num_rays,
                                     std::vector<double>& weights);

/**
 * @brief Performs batch ray casting for multiple queries
 * @param node Pointer to MCL node
 * @param queries Matrix of ray queries (Nx3: x, y, angle)
 * @return Vector of distances to obstacles for each query
 */
std::vector<float> calc_range_many(MCL* node, const Eigen::MatrixXd &queries);

/**
 * @brief Casts single ray to find obstacle distance
 * @param x Starting x position
 * @param y Starting y position
 * @param angle Ray angle in radians
 * @param local_map Occupancy grid map
 * @param max_range_px Maximum range in pixels
 * @return Distance to obstacle in meters
 */
float cast_ray(double x, double y, double angle,
               const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
               int max_range_px);

} // namespace sensor_model
} // namespace mcl_pkg

#endif  // MCL_PKG__SENSOR_MODEL__BEAM_SENSOR_MODEL_HPP_
