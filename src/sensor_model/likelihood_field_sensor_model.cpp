// ================================================================================================
// LIKELIHOOD FIELD SENSOR MODEL - Implementation
// ================================================================================================
// Distance field based sensor model - no ray casting needed
// Standard algorithm from Probabilistic Robotics Ch.6.4 with extended unknown space handling:
//
// Base 3-component model:
//   - z_hit: Gaussian around expected measurement
//   - z_rand: Uniform random noise
//   - z_max: Max range measurements (skipped, no information)
//
// Extended algorithm (3-category map classification):
//   - Occupied: obstacle cells (distance = 0)
//   - Free: known free space (distance > 0)
//   - Unknown: unexplored space (p(z|unknown) = 1/z_max, uniform)
// ================================================================================================

#include "mcl_pkg/sensor_model/likelihood_field_sensor_model.hpp"
#include "mcl_pkg/mcl.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

namespace mcl_pkg {
namespace sensor_model {

/**
 * @brief Helper function: Evaluate likelihood for a single particle
 * @note Implements Algorithm likelihood_field_range_finder_model from Probabilistic Robotics
 *       Chapter 6.4, using 3-component model: z_hit + z_rand + z_max
 */
static inline double evaluate_particle_likelihood(
    MCL* node,
    const Eigen::MatrixXd& proposal_dist,
    const std::vector<float>& obs,
    int particle_idx,
    int num_rays,
    double prob_uniform,
    double resolution,
    double origin_x,
    double origin_y)
{
    // Use log-space computation to avoid numerical underflow
    double log_weight = 0.0;
    const double px = proposal_dist(particle_idx, 0);
    const double py = proposal_dist(particle_idx, 1);
    const double ptheta = proposal_dist(particle_idx, 2);

    // Precompute particle rotation once per particle
    const double cos_theta = std::cos(ptheta);
    const double sin_theta = std::sin(ptheta);

    // Pre-allocate arrays for vectorized operations (minimize allocations in hot loop)
    static thread_local std::vector<double> local_x_batch;
    static thread_local std::vector<double> local_y_batch;
    static thread_local std::vector<double> endpoint_x_batch;
    static thread_local std::vector<double> endpoint_y_batch;

    if (local_x_batch.size() < static_cast<size_t>(num_rays)) {
        local_x_batch.resize(num_rays);
        local_y_batch.resize(num_rays);
        endpoint_x_batch.resize(num_rays);
        endpoint_y_batch.resize(num_rays);
    }

    // Vectorized endpoint calculation for all valid rays
    // This enables compiler auto-vectorization (SIMD)
    for (int j = 0; j < num_rays; ++j) {
        const float obs_range = obs[j];
        local_x_batch[j] = obs_range * node->cos_table_[j];
        local_y_batch[j] = obs_range * node->sin_table_[j];
        endpoint_x_batch[j] = px + (local_x_batch[j] * cos_theta - local_y_batch[j] * sin_theta);
        endpoint_y_batch[j] = py + (local_x_batch[j] * sin_theta + local_y_batch[j] * cos_theta);
    }

    // Process each ray
    for (int j = 0; j < num_rays; ++j) {
        const float obs_range = obs[j];

        // === Algorithm Line 4: if z_t^k ≠ z_max ===
        // Max range measurements are skipped (no information)
        if (obs_range >= node->MAX_RANGE_METERS) {
            continue;
        }

        // === Invalid measurements: skip ===
        if (obs_range <= 0.0f) {
            continue;
        }

        // === Algorithm Line 5-6: Calculate endpoint coordinates ===
        // Use pre-computed endpoint from vectorized calculation
        const double endpoint_x = endpoint_x_batch[j];
        const double endpoint_y = endpoint_y_batch[j];

        // Convert to grid coordinates
        int grid_x = static_cast<int>((endpoint_x - origin_x) / resolution);
        int grid_y = static_cast<int>((endpoint_y - origin_y) / resolution);

        // Out of bounds: treat as unknown space
        // Extended algorithm: p(z|unknown) = 1/z_max (uniform probability)
        if (grid_x < 0 || grid_x >= node->distance_field_width_ ||
            grid_y < 0 || grid_y >= node->distance_field_height_) {
            double prob_unknown = prob_uniform;  // 1/z_max
            log_weight += std::log(std::max(prob_unknown, 1e-10));
            continue;
        }

        // === Algorithm Line 7: dist = min distance to nearest obstacle ===
        int idx = grid_y * node->distance_field_width_ + grid_x;
        float dist = node->distance_field_[idx];

        // Extended algorithm: check if endpoint falls in unknown space
        if (dist < 0.0f) {
            // Unknown cell: p(z|unknown) = 1/z_max (uniform probability)
            double prob_unknown = prob_uniform;
            log_weight += std::log(std::max(prob_unknown, 1e-10));
            continue;
        }

        // === z_hit component: Gaussian likelihood ===
        int table_idx = std::min(static_cast<int>(dist / node->likelihood_table_resolution_),
                                 node->likelihood_table_size_ - 1);
        double prob_hit = node->likelihood_lookup_table_[table_idx];

        // === Algorithm Line 8: q = q · (z_hit · prob(dist, σ_hit) + z_random/z_max) ===
        // Two-component model: z_hit (Gaussian) + z_rand (uniform)
        double prob_total = node->Z_HIT * prob_hit + node->Z_RAND * prob_uniform;

        // Clamp probability for numerical stability
        prob_total = std::clamp(prob_total, 1e-10, 1.0);

        // Accumulate in log-space
        log_weight += std::log(prob_total);
    }

    // Apply squash factor and convert back from log-space
    return std::exp(node->INV_SQUASH_FACTOR * log_weight);
}


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

    // Initialize with 3-category classification (extended algorithm):
    // 0.0f: occupied (obstacle), INF: free space, -1.0f: unknown
    const float INF = 9999.0f;
    const float UNKNOWN_MARKER = -1.0f;
    int obstacle_count = 0;
    int unknown_count = 0;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            int8_t cell_value = local_map->data[idx];

            // Three categories: occupied, free, unknown
            if (cell_value > 50) {
                // Occupied cell (obstacle)
                node->distance_field_[idx] = 0.0f;
                obstacle_count++;
            }
            else if (cell_value < 0) {
                // Unknown cell (-1 or 255)
                node->distance_field_[idx] = UNKNOWN_MARKER;
                unknown_count++;
            }
            else {
                // Free space (0-50)
                node->distance_field_[idx] = INF;
            }
        }
    }

    // Forward pass: scan from top-left to bottom-right
    // Skip unknown cells (marked as -1.0f)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;

            // Only process free space cells (skip obstacles and unknown)
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
    // Skip unknown cells (marked as -1.0f)
    for (int y = height - 1; y >= 0; --y)
    {
        for (int x = width - 1; x >= 0; --x)
        {
            int idx = y * width + x;

            // Only process free space cells (skip obstacles and unknown)
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

    // Convert pixel distances to meters (but preserve unknown markers)
    for (size_t i = 0; i < node->distance_field_.size(); ++i)
    {
        if (node->distance_field_[i] >= 0.0f) {
            node->distance_field_[i] *= resolution;
        }
        // Unknown cells remain as -1.0f
    }

    node->distance_field_initialized_ = true;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_INFO(node->get_logger(),
                "Distance field computed (%ld ms) - %d obstacles, %d unknown, %dx%d cells",
                duration.count(), obstacle_count, unknown_count, width, height);
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
            weights[i] = evaluate_particle_likelihood(
                node, proposal_dist, obs, i, num_rays, prob_uniform,
                resolution, origin_x, origin_y);
        }
    } else {
        // Sequential version
        for (int i = 0; i < node->MAX_PARTICLES; ++i) {
            weights[i] = evaluate_particle_likelihood(
                node, proposal_dist, obs, i, num_rays, prob_uniform,
                resolution, origin_x, origin_y);
        }
    }
}

} // namespace sensor_model
} // namespace mcl_pkg
