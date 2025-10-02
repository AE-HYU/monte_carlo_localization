// ================================================================================================
// MAP MANAGER IMPLEMENTATION - Map Loading and Sensor Model Precomputation
// ================================================================================================
// Extracted from particle_filter.cpp for better code organization
// ================================================================================================

#include "particle_filter_cpp/map_manager.hpp"
#include "particle_filter_cpp/particle_filter.hpp"
#include "particle_filter_cpp/initialization.hpp"
#include <chrono>
#include <cmath>

namespace particle_filter_cpp {
namespace map_manager {

// ================================================================================================
// MAP LOADING & PREPROCESSING
// ================================================================================================
/**
 * @brief Loads occupancy grid map from map server and extracts free space (BLOCKING - legacy)
 */
void get_omap(ParticleFilter* node)
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

        // Generate sensor model lookup table
        precompute_sensor_model(node);
    }
    else
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to get map from map server");
    }
}

/**
 * @brief Non-blocking async map loading - AMCL style graceful initialization
 */
void try_load_map(ParticleFilter* node)
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

                // Precompute sensor model
                precompute_sensor_model(node);

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
// ================================================================================================
/**
 * @brief Precomputes sensor model lookup table for fast likelihood evaluation
 */
void precompute_sensor_model(ParticleFilter* node)
{
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(node->map_lock_);
        local_map = node->map_msg_;
    }

    if (!local_map || local_map->info.resolution <= 0.0)
    {
        RCLCPP_ERROR(node->get_logger(), "Invalid map resolution: %.6f",
                     local_map ? local_map->info.resolution : 0.0);
        return;
    }

    int table_width = node->MAX_RANGE_PX_ + 1;
    node->sensor_model_table_ = Eigen::MatrixXd::Zero(table_width, table_width);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Build lookup table
    for (int d = 0; d < table_width; ++d)  // d = expected range
    {
        double norm = 0.0;

        for (int r = 0; r < table_width; ++r)  // r = observed range
        {
            double prob = 0.0;
            double z = static_cast<double>(r - d);

            // Z_HIT: Gaussian around expected range
            prob += node->Z_HIT * std::exp(-(z * z) / (2.0 * node->SIGMA_HIT * node->SIGMA_HIT)) / (node->SIGMA_HIT * std::sqrt(2.0 * M_PI));

            // Z_SHORT: Exponential for early obstacles
            if (r < d)
            {
                prob += 2.0 * node->Z_SHORT * (d - r) / static_cast<double>(d);
            }

            // Z_MAX: Delta function at maximum range
            if (r == node->MAX_RANGE_PX_)
            {
                prob += node->Z_MAX;
            }

            // Z_RAND: Uniform distribution
            if (r < node->MAX_RANGE_PX_)
            {
                prob += node->Z_RAND * 1.0 / static_cast<double>(node->MAX_RANGE_PX_);
            }

            norm += prob;
            node->sensor_model_table_(r, d) = prob;
        }

        // Normalize
        if (norm > 0)
        {
            node->sensor_model_table_.col(d) /= norm;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_INFO(node->get_logger(), "Sensor model ready (%ld ms)", duration.count());
}

} // namespace map_manager
} // namespace particle_filter_cpp
