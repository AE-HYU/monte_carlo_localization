// ================================================================================================
// VISUALIZATION - TF publishing and RViz visualization
// ================================================================================================

#ifndef PARTICLE_FILTER_CPP__VISUALIZATION_HPP_
#define PARTICLE_FILTER_CPP__VISUALIZATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

namespace mcl_pkg
{

// Forward declaration
class MCL;

namespace visualization
{

/**
 * @brief Publish map->odom TF transform
 */
void publish_tf(MCL* node, const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp);

/**
 * @brief Publish visualization data for RViz (pose, particles)
 */
void visualize(MCL* node, const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp);

/**
 * @brief Publish particle array for RViz
 */
void publish_particles(MCL* node, const Eigen::MatrixXd &particles_to_pub, const rclcpp::Time &stamp);

} // namespace visualization
} // namespace mcl_pkg

#endif
