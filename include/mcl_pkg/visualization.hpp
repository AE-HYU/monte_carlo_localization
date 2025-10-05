// ================================================================================================
// VISUALIZATION - TF publishing and RViz visualization
// ================================================================================================

#ifndef MCL_PKG__VISUALIZATION_HPP_
#define MCL_PKG__VISUALIZATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

namespace mcl_pkg
{

// Forward declaration
class MCL;

namespace visualization
{

/**
 * @brief Publish TF, odometry, and pose at high frequency (called every timer tick)
 */
void publish_localization(MCL* node, const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp);

/**
 * @brief Publish particle cloud visualization (called only on MCL updates)
 */
void publish_particles_viz(MCL* node, const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp);

} // namespace visualization
} // namespace mcl_pkg

#endif // MCL_PKG__VISUALIZATION_HPP_
