// ================================================================================================
// MAP MANAGER IMPLEMENTATION - Map Loading and Sensor Model Precomputation
// ================================================================================================
// Extracted from particle_filter.cpp for better code organization
// ================================================================================================

#include "mcl_pkg/map_manager.hpp"
#include "mcl_pkg/mcl.hpp"
#include "mcl_pkg/initialization.hpp"
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
        precompute_sensor_model(node);

        // Generate distance field and lookup table (likelihood field model)
        if (node->SENSOR_MODEL_TYPE == "likelihood_field") {
            precompute_distance_field(node);
            node->precompute_likelihood_lookup_table();
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
                precompute_sensor_model(node);

                // Precompute distance field and lookup table (likelihood field model)
                if (node->SENSOR_MODEL_TYPE == "likelihood_field") {
                    precompute_distance_field(node);
                    node->precompute_likelihood_lookup_table();
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
// ================================================================================================
/**
 * @brief Precomputes sensor model lookup table for fast likelihood evaluation
 */
void precompute_sensor_model(MCL* node)
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
    RCLCPP_INFO(node->get_logger(), "Beam sensor model ready (%ld ms)", duration.count());
}

/**
 * @brief Precomputes distance field using efficient two-pass distance transform
 * Based on chamfer distance transform - O(N) complexity instead of O(N*M)
 */
void precompute_distance_field(MCL* node)
{
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(node->map_lock_);
        local_map = node->map_msg_;
    }

    if (!local_map || local_map->info.resolution <= 0.0)
    {
        RCLCPP_ERROR(node->get_logger(), "Invalid map for distance field computation");
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int width = local_map->info.width;
    int height = local_map->info.height;
    double resolution = local_map->info.resolution;

    // Initialize distance field
    node->distance_field_.assign(width * height, 0.0f);
    node->distance_field_width_ = width;
    node->distance_field_height_ = height;
    node->distance_field_resolution_ = resolution;

    // Initialize with binary: 0 for obstacles, large value for free space
    const float INF = 9999.0f;
    int obstacle_count = 0;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            // Occupied cells (>50) or unknown cells (-1/255) are obstacles
            if (local_map->data[idx] > 50 || local_map->data[idx] < 0)
            {
                node->distance_field_[idx] = 0.0f;
                obstacle_count++;
            }
            else
            {
                node->distance_field_[idx] = INF;
            }
        }
    }

    // Forward pass: scan from top-left to bottom-right
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;

            if (node->distance_field_[idx] > 0.0f)
            {
                // Check left neighbor
                if (x > 0)
                {
                    int left_idx = y * width + (x - 1);
                    float dist = node->distance_field_[left_idx] + 1.0f;
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }

                // Check top neighbor
                if (y > 0)
                {
                    int top_idx = (y - 1) * width + x;
                    float dist = node->distance_field_[top_idx] + 1.0f;
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }

                // Check top-left diagonal
                if (x > 0 && y > 0)
                {
                    int diag_idx = (y - 1) * width + (x - 1);
                    float dist = node->distance_field_[diag_idx] + 1.414f;  // sqrt(2)
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }

                // Check top-right diagonal
                if (x < width - 1 && y > 0)
                {
                    int diag_idx = (y - 1) * width + (x + 1);
                    float dist = node->distance_field_[diag_idx] + 1.414f;
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }
            }
        }
    }

    // Backward pass: scan from bottom-right to top-left
    for (int y = height - 1; y >= 0; --y)
    {
        for (int x = width - 1; x >= 0; --x)
        {
            int idx = y * width + x;

            if (node->distance_field_[idx] > 0.0f)
            {
                // Check right neighbor
                if (x < width - 1)
                {
                    int right_idx = y * width + (x + 1);
                    float dist = node->distance_field_[right_idx] + 1.0f;
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }

                // Check bottom neighbor
                if (y < height - 1)
                {
                    int bottom_idx = (y + 1) * width + x;
                    float dist = node->distance_field_[bottom_idx] + 1.0f;
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }

                // Check bottom-right diagonal
                if (x < width - 1 && y < height - 1)
                {
                    int diag_idx = (y + 1) * width + (x + 1);
                    float dist = node->distance_field_[diag_idx] + 1.414f;
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }

                // Check bottom-left diagonal
                if (x > 0 && y < height - 1)
                {
                    int diag_idx = (y + 1) * width + (x - 1);
                    float dist = node->distance_field_[diag_idx] + 1.414f;
                    if (dist < node->distance_field_[idx])
                        node->distance_field_[idx] = dist;
                }
            }
        }
    }

    // Convert pixel distances to meters
    for (size_t i = 0; i < node->distance_field_.size(); ++i)
    {
        node->distance_field_[i] *= resolution;
    }

    node->distance_field_initialized_ = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_INFO(node->get_logger(),
                "Distance field computed (%ld ms) - %d obstacles, %dx%d cells",
                duration.count(), obstacle_count, width, height);
}

} // namespace map_manager
} // namespace mcl_pkg
