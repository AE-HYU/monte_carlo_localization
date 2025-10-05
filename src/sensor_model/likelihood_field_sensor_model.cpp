// ================================================================================================
// LIKELIHOOD FIELD SENSOR MODEL - Implementation
// ================================================================================================
// Distance field based sensor model - no ray casting needed
// Multi-component model: z_hit (Gaussian) + z_short (Exponential) + z_max + z_rand
// ================================================================================================

#include "mcl_pkg/sensor_model/likelihood_field_sensor_model.hpp"
#include "mcl_pkg/mcl.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

namespace mcl_pkg {
namespace sensor_model {

/**
 * @brief Precomputes distance field using efficient two-pass distance transform
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

/**
 * @brief Precompute Gaussian likelihood lookup table
 */
void precompute_likelihood_lookup_table(MCL* node)
{
    // Maximum distance to precompute (meters)
    const double MAX_DIST = 5.0;

    // Calculate table size based on resolution
    node->likelihood_table_size_ = static_cast<int>(MAX_DIST / node->likelihood_table_resolution_) + 1;
    node->likelihood_lookup_table_.resize(node->likelihood_table_size_);

    // Precompute Gaussian normalizer
    const double norm_factor = 1.0 / (node->LIKELIHOOD_SIGMA * std::sqrt(2.0 * M_PI));
    const double inv_two_sigma_sq = 1.0 / (2.0 * node->LIKELIHOOD_SIGMA * node->LIKELIHOOD_SIGMA);

    // Fill lookup table
    for (int i = 0; i < node->likelihood_table_size_; ++i)
    {
        double dist = i * node->likelihood_table_resolution_;
        node->likelihood_lookup_table_[i] = norm_factor * std::exp(-dist * dist * inv_two_sigma_sq);
    }

    RCLCPP_INFO(node->get_logger(),
                "Likelihood lookup table ready - %d entries, resolution: %.3fm",
                node->likelihood_table_size_, node->likelihood_table_resolution_);
}

/**
 * @brief Likelihood field sensor model - no ray casting needed
 */
void likelihood_field_sensor_update(MCL* node,
                                   const Eigen::MatrixXd& proposal_dist,
                                   const std::vector<float>& obs,
                                   std::vector<double>& weights)
{
    if (!node->distance_field_initialized_) {
        RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 1000,
                            "Distance field not initialized, skipping sensor update");
        std::fill(weights.begin(), weights.end(), 1.0);
        return;
    }

    const int num_rays = node->downsampled_angles_.size();

    // Thread-safe access to map info
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(node->map_lock_);
        local_map = node->map_msg_;
    }
    if (!local_map) return;

    double resolution = local_map->info.resolution;
    double origin_x = local_map->info.origin.position.x;
    double origin_y = local_map->info.origin.position.y;

    // Precompute uniform probability for z_max and z_rand
    const double prob_uniform = 1.0 / node->MAX_RANGE_METERS;

    // Evaluate each particle (parallelize with OpenMP)
    if (node->USE_PARALLEL_RAYCASTING) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < node->MAX_PARTICLES; ++i) {
            double weight = 1.0;
            const double px = proposal_dist(i, 0);
            const double py = proposal_dist(i, 1);
            const double ptheta = proposal_dist(i, 2);

            // Precompute particle rotation once per particle
            const double cos_theta = std::cos(ptheta);
            const double sin_theta = std::sin(ptheta);

            for (int j = 0; j < num_rays; ++j) {
                const float obs_range = obs[j];

                // === z_max component: max range measurements ===
                if (obs_range >= node->MAX_RANGE_METERS) {
                    weight *= (node->Z_MAX * prob_uniform + node->Z_RAND * prob_uniform);
                    continue;
                }

                // === Invalid measurements: z_rand only ===
                if (obs_range <= 0.0f) {
                    weight *= (node->Z_RAND * prob_uniform);
                    continue;
                }

                // Calculate endpoint of the beam in world coordinates using precomputed cos/sin
                const double local_x = obs_range * node->cos_table_[j];
                const double local_y = obs_range * node->sin_table_[j];
                const double endpoint_x = px + (local_x * cos_theta - local_y * sin_theta);
                const double endpoint_y = py + (local_x * sin_theta + local_y * cos_theta);

                // Convert to grid coordinates
                int grid_x = static_cast<int>((endpoint_x - origin_x) / resolution);
                int grid_y = static_cast<int>((endpoint_y - origin_y) / resolution);

                // Out of bounds: z_rand only
                if (grid_x < 0 || grid_x >= node->distance_field_width_ ||
                    grid_y < 0 || grid_y >= node->distance_field_height_) {
                    weight *= (node->Z_RAND * prob_uniform);
                    continue;
                }

                // Look up distance to nearest obstacle
                int idx = grid_y * node->distance_field_width_ + grid_x;
                float dist = node->distance_field_[idx];

                // === z_hit component: Gaussian likelihood ===
                int table_idx = std::min(static_cast<int>(dist / node->likelihood_table_resolution_),
                                         node->likelihood_table_size_ - 1);
                double prob_hit = node->likelihood_lookup_table_[table_idx];

                // === z_short component: Exponential decay for short readings ===
                // If endpoint is far from obstacles, there might be a dynamic obstacle blocking the ray
                double prob_short = 0.0;
                if (dist > obs_range * 0.3) {  // Endpoint is in free space
                    // Exponential model: shorter measurements more likely due to occlusion
                    prob_short = (2.0 / obs_range) * std::exp(-2.0 * obs_range);
                }

                // === Multi-component probability ===
                double prob_total =
                    node->Z_HIT * prob_hit +
                    node->Z_SHORT * prob_short +
                    node->Z_MAX * prob_uniform +
                    node->Z_RAND * prob_uniform;

                // Accumulate probability (add small epsilon for numerical stability)
                weight *= (prob_total + 1e-6);
            }

            // Apply squash factor
            if (weight > 0.0) {
                weights[i] = std::exp(node->INV_SQUASH_FACTOR * std::log(weight));
            } else {
                weights[i] = 0.0;
            }
        }
    } else {
        // Sequential version
        for (int i = 0; i < node->MAX_PARTICLES; ++i) {
            double weight = 1.0;
            const double px = proposal_dist(i, 0);
            const double py = proposal_dist(i, 1);
            const double ptheta = proposal_dist(i, 2);

            // Precompute particle rotation once per particle
            const double cos_theta = std::cos(ptheta);
            const double sin_theta = std::sin(ptheta);

            for (int j = 0; j < num_rays; ++j) {
                const float obs_range = obs[j];

                // === z_max component: max range measurements ===
                if (obs_range >= node->MAX_RANGE_METERS) {
                    weight *= (node->Z_MAX * prob_uniform + node->Z_RAND * prob_uniform);
                    continue;
                }

                // === Invalid measurements: z_rand only ===
                if (obs_range <= 0.0f) {
                    weight *= (node->Z_RAND * prob_uniform);
                    continue;
                }

                // Calculate endpoint of the beam in world coordinates using precomputed cos/sin
                const double local_x = obs_range * node->cos_table_[j];
                const double local_y = obs_range * node->sin_table_[j];
                const double endpoint_x = px + (local_x * cos_theta - local_y * sin_theta);
                const double endpoint_y = py + (local_x * sin_theta + local_y * cos_theta);

                // Convert to grid coordinates
                int grid_x = static_cast<int>((endpoint_x - origin_x) / resolution);
                int grid_y = static_cast<int>((endpoint_y - origin_y) / resolution);

                // Out of bounds: z_rand only
                if (grid_x < 0 || grid_x >= node->distance_field_width_ ||
                    grid_y < 0 || grid_y >= node->distance_field_height_) {
                    weight *= (node->Z_RAND * prob_uniform);
                    continue;
                }

                // Look up distance to nearest obstacle
                int idx = grid_y * node->distance_field_width_ + grid_x;
                float dist = node->distance_field_[idx];

                // === z_hit component: Gaussian likelihood ===
                int table_idx = std::min(static_cast<int>(dist / node->likelihood_table_resolution_),
                                         node->likelihood_table_size_ - 1);
                double prob_hit = node->likelihood_lookup_table_[table_idx];

                // === z_short component: Exponential decay for short readings ===
                // If endpoint is far from obstacles, there might be a dynamic obstacle blocking the ray
                double prob_short = 0.0;
                if (dist > obs_range * 0.3) {  // Endpoint is in free space
                    // Exponential model: shorter measurements more likely due to occlusion
                    prob_short = (2.0 / obs_range) * std::exp(-2.0 * obs_range);
                }

                // === Multi-component probability ===
                double prob_total =
                    node->Z_HIT * prob_hit +
                    node->Z_SHORT * prob_short +
                    node->Z_MAX * prob_uniform +
                    node->Z_RAND * prob_uniform;

                // Accumulate probability (add small epsilon for numerical stability)
                weight *= (prob_total + 1e-6);
            }

            // Apply squash factor
            if (weight > 0.0) {
                weights[i] = std::exp(node->INV_SQUASH_FACTOR * std::log(weight));
            } else {
                weights[i] = 0.0;
            }
        }
    }
}

} // namespace sensor_model
} // namespace mcl_pkg
