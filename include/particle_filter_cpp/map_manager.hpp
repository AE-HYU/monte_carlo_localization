// ================================================================================================
// MAP MANAGER - Map loading and sensor model precomputation
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__MAP_MANAGER_HPP_
#define PARTICLE_FILTER_CPP__MAP_MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace particle_filter_cpp
{

// Forward declaration
class ParticleFilter;

namespace map_manager
{

/**
 * @brief Synchronously request and load map from map server (legacy)
 *
 * Blocking call that waits for map server and loads map.
 * Used for backwards compatibility.
 *
 * @param node Pointer to ParticleFilter node
 */
void get_omap(ParticleFilter* node);

/**
 * @brief Asynchronously try to load map (non-blocking)
 *
 * Called periodically by timer until map is successfully loaded.
 * Enables graceful startup without dependencies.
 *
 * @param node Pointer to ParticleFilter node
 */
void try_load_map(ParticleFilter* node);

/**
 * @brief Precompute sensor model lookup table for all ranges
 *
 * Generates probability table for beam sensor model:
 * - p_hit: Gaussian around true range
 * - p_short: Exponential for early returns
 * - p_max: Probability of max range reading
 * - p_rand: Uniform random noise
 *
 * Lookup table indexed by: [observed_range_px][true_range_px]
 *
 * @param node Pointer to ParticleFilter node
 */
void precompute_sensor_model(ParticleFilter* node);

} // namespace map_manager
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MAP_MANAGER_HPP_
