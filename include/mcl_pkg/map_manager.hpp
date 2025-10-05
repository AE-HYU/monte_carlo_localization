// ================================================================================================
// MAP MANAGER - Map loading and sensor model precomputation
// ================================================================================================

#ifndef MCL_PKG__MAP_MANAGER_HPP_
#define MCL_PKG__MAP_MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/srv/get_map.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

namespace mcl_pkg
{

// Forward declaration
class MCL;

namespace map_manager
{

/**
 * @brief Synchronously request and load map from map server (legacy)
 *
 * Blocking call that waits for map server and loads map.
 * Used for backwards compatibility.
 *
 * @param node Pointer to MCL node
 */
void get_omap(MCL* node);

/**
 * @brief Asynchronously try to load map (non-blocking)
 *
 * Called periodically by timer until map is successfully loaded.
 * Enables graceful startup without dependencies.
 *
 * @param node Pointer to MCL node
 */
void try_load_map(MCL* node);

} // namespace map_manager
} // namespace mcl_pkg

#endif // MCL_PKG__MAP_MANAGER_HPP_
