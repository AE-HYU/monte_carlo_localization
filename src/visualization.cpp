// ================================================================================================
// VISUALIZATION - TF publishing and RViz visualization
// ================================================================================================

#include "mcl_pkg/visualization.hpp"
#include "mcl_pkg/mcl.hpp"
#include "mcl_pkg/utils.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

namespace mcl_pkg
{
namespace visualization
{

// Internal helper function for TF publishing
static void publish_tf(MCL* node, const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp)
{
    if (!node->PUBLISH_MAP_ODOM_TF) {
        return;
    }

    rclcpp::Time tf_stamp = (stamp.nanoseconds() != 0) ? stamp : node->get_clock()->now();

    if (!node->odom_initialized_ || base_link_pose.norm() <= 0) {
        RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 1000,
            "Cannot publish TF - odom not initialized or invalid pose");
        return;
    }

    try {
        // Calculate: T_map_odom = T_map_base * T_base_odom^(-1)

        // Step 1: Create MCL's map->base_link transform
        tf2::Transform map_to_base;
        tf2::Quaternion q;
        q.setRPY(0, 0, base_link_pose[2]);
        map_to_base.setOrigin(tf2::Vector3(base_link_pose[0], base_link_pose[1], 0.0));
        map_to_base.setRotation(q);

        // Step 2: Get odom->base_link transform from TF tree
        geometry_msgs::msg::TransformStamped odom_to_base_msg;
        tf2::Transform odom_to_base;

        try {
            odom_to_base_msg = node->tf_buffer_->lookupTransform(
                node->ODOM_FRAME, node->BASE_FRAME, tf_stamp, tf2::durationFromSec(0.05));
            tf2::fromMsg(odom_to_base_msg.transform, odom_to_base);
        } catch (tf2::TransformException &ex) {
            // FALLBACK: Assume map = odom
            RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 5000,
                "TF lookup failed, assuming map=odom (simulation mode): %s", ex.what());
            odom_to_base = map_to_base;
        }

        // Step 3: Compute map->odom
        tf2::Transform map_to_odom = map_to_base * odom_to_base.inverse();

        // Step 4: Publish
        geometry_msgs::msg::TransformStamped map_to_odom_msg;
        map_to_odom_msg.header.stamp = tf_stamp;
        map_to_odom_msg.header.frame_id = node->MAP_FRAME;
        map_to_odom_msg.child_frame_id = node->ODOM_FRAME;
        map_to_odom_msg.transform = tf2::toMsg(map_to_odom);

        node->pub_tf_->sendTransform(map_to_odom_msg);

    } catch (const std::exception &ex) {
        RCLCPP_ERROR_THROTTLE(node->get_logger(), *node->get_clock(), 1000,
            "Unexpected error in TF publishing: %s", ex.what());
    }
}

void publish_localization(MCL* node, const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp)
{
    rclcpp::Time pub_stamp = (stamp.nanoseconds() != 0) ? stamp : node->get_clock()->now();

    // 1. Publish TF (map->odom)
    publish_tf(node, base_link_pose, pub_stamp);

    // 2. Publish Odometry message
    if (node->PUBLISH_ODOM && node->odom_pub_ && node->odom_pub_->get_subscription_count() > 0)
    {
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = pub_stamp;
        odom_msg.header.frame_id = node->MAP_FRAME;
        odom_msg.child_frame_id = node->BASE_FRAME;

        odom_msg.pose.pose.position.x = base_link_pose[0];
        odom_msg.pose.pose.position.y = base_link_pose[1];
        odom_msg.pose.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, base_link_pose[2]);
        odom_msg.pose.pose.orientation = tf2::toMsg(q);

        node->odom_pub_->publish(odom_msg);
    }

    // 3. Publish Pose visualization (RViz arrow)
    if (node->DO_VIZ && node->pose_pub_ && node->pose_pub_->get_subscription_count() > 0)
    {
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = pub_stamp;
        pose_msg.header.frame_id = node->MAP_FRAME;

        pose_msg.pose.position.x = base_link_pose[0];
        pose_msg.pose.position.y = base_link_pose[1];
        pose_msg.pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, base_link_pose[2]);
        pose_msg.pose.orientation = tf2::toMsg(q);

        node->pose_pub_->publish(pose_msg);
    }
}

// Internal helper function for particle publishing
static void publish_particles(MCL* node, const Eigen::MatrixXd &particles_to_pub, const rclcpp::Time &stamp)
{
    geometry_msgs::msg::PoseArray particle_array;
    particle_array.header.stamp = stamp;
    particle_array.header.frame_id = node->MAP_FRAME;

    int viz_count = std::min(node->MAX_VIZ_PARTICLES, static_cast<int>(particles_to_pub.rows()));

    for (int i = 0; i < viz_count; ++i)
    {
        Eigen::Vector3d laser_particle(particles_to_pub(i, 0), particles_to_pub(i, 1), particles_to_pub(i, 2));

        // Convert laser frame to base_link frame
        Eigen::Vector3d base_particle = utils::transforms::apply_laser_to_base_offset(
            laser_particle, node->laser_offset_x_, node->laser_offset_y_);

        geometry_msgs::msg::Pose pose;
        pose.position.x = base_particle[0];
        pose.position.y = base_particle[1];
        pose.position.z = 0.0;

        tf2::Quaternion q;
        q.setRPY(0, 0, base_particle[2]);
        pose.orientation = tf2::toMsg(q);

        particle_array.poses.push_back(pose);
    }

    node->particle_pub_->publish(particle_array);
}

void publish_particles_viz(MCL* node, const Eigen::Vector3d &base_link_pose, const rclcpp::Time &stamp)
{
    (void)base_link_pose;  // Unused parameter

    if (!node->DO_VIZ || !node->particle_pub_)
        return;

    rclcpp::Time viz_stamp = (stamp.nanoseconds() != 0) ? stamp : node->get_clock()->now();

    std::lock_guard<std::mutex> lock(node->state_lock_);
    publish_particles(node, node->particles_, viz_stamp);
}

} // namespace visualization
} // namespace mcl_pkg
