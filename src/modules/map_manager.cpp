// ================================================================================================
// MAP MANAGER IMPLEMENTATION - Map Loading and Sensor Model Precomputation
// ================================================================================================
// Extracted from particle_filter.cpp for better code organization
// ================================================================================================

#include "mcl_pkg/map_manager.hpp"
#include "mcl_pkg/mcl.hpp"
#include "mcl_pkg/initialization.hpp"
#include "mcl_pkg/sensor_model/beam_sensor_model.hpp"
#include "mcl_pkg/sensor_model/likelihood_field_sensor_model.hpp"
#include <chrono>
#include <cmath>

namespace mcl_pkg {
namespace map_manager {

// ================================================================================================
// MAP LOADING & PREPROCESSING
// ================================================================================================
/**
 * @brief Loads occupancy grid map from map server and extracts free space (BLOCKING - legacy)
 */
void get_omap(MCL* node)
{
    RCLCPP_INFO(node->get_logger(), "Requesting map from map server...");

    while (!node->map_client_->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok())
            return;
        RCLCPP_INFO(node->get_logger(), "Get map service not available, waiting...");
    }

    auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();
    auto future = node->map_client_->async_send_request(request);

    if (rclcpp::spin_until_future_complete(node->get_node_base_interface(), future) ==
        rclcpp::FutureReturnCode::SUCCESS)
    {
        node->map_msg_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(future.get()->map);

        node->MAX_RANGE_PX_ = static_cast<int>(node->MAX_RANGE_METERS / node->map_msg_->info.resolution);

        node->map_initialized_ = true;
        RCLCPP_INFO(node->get_logger(), "Map loaded and published");

        // Publish map
        if (node->map_pub_) {
            node->map_pub_->publish(*node->map_msg_);
        }

        // Generate sensor model lookup table (beam model)
        sensor_model::precompute_beam_sensor_model(node);

        // Generate distance field and lookup table (likelihood field model)
        if (node->SENSOR_MODEL_TYPE == "likelihood_field") {
            sensor_model::precompute_distance_field(node);
            sensor_model::precompute_likelihood_lookup_table(node);
        }
    }
    else
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to get map from map server");
    }
}

/**
 * @brief Non-blocking async map loading - AMCL style graceful initialization
 */
void try_load_map(MCL* node)
{
    // Already loaded - stop timer
    if (node->map_initialized_) {
        if (node->map_loader_timer_) {
            node->map_loader_timer_->cancel();
            RCLCPP_INFO(node->get_logger(), "Map loading complete - timer stopped");
        }
        return;
    }

    // Check if map service is ready (non-blocking)
    if (!node->map_client_->service_is_ready()) {
        RCLCPP_INFO_THROTTLE(node->get_logger(), *node->get_clock(), 2000,
            "Waiting for map server to be available...");
        return;
    }

    // Send async request
    RCLCPP_INFO(node->get_logger(), "Map service available - requesting map...");
    auto request = std::make_shared<nav_msgs::srv::GetMap::Request>();

    node->map_client_->async_send_request(request,
        [node](rclcpp::Client<nav_msgs::srv::GetMap>::SharedFuture future) {
            try {
                // Get map data
                node->map_msg_ = std::make_shared<nav_msgs::msg::OccupancyGrid>(future.get()->map);

                node->MAX_RANGE_PX_ = static_cast<int>(node->MAX_RANGE_METERS / node->map_msg_->info.resolution);

                // Precompute sensor model (beam model)
                sensor_model::precompute_beam_sensor_model(node);

                // Precompute distance field and lookup table (likelihood field model)
                if (node->SENSOR_MODEL_TYPE == "likelihood_field") {
                    sensor_model::precompute_distance_field(node);
                    sensor_model::precompute_likelihood_lookup_table(node);
                }

                // Publish map for RViz
                if (node->map_pub_) {
                    node->map_pub_->publish(*node->map_msg_);
                }

                // Mark as initialized
                node->map_initialized_ = true;

                // Initialize particles globally after map is ready
                initialization::initialize_global(node);

                RCLCPP_INFO(node->get_logger(),
                    "Map loaded successfully (%dx%d @ %.3fm/px) - MCL ready",
                    node->map_msg_->info.width, node->map_msg_->info.height, node->map_msg_->info.resolution);

                // Stop the loading timer
                if (node->map_loader_timer_) {
                    node->map_loader_timer_->cancel();
                }

            } catch (const std::exception& e) {
                RCLCPP_ERROR(node->get_logger(),
                    "Failed to load map: %s - will retry...", e.what());
                // Timer will automatically retry
            }
        });
}

// ================================================================================================
// SENSOR MODEL PRECOMPUTATION

} // namespace map_manager
} // namespace mcl_pkg
