// ================================================================================================
// BEAM SENSOR MODEL - Implementation
// ================================================================================================
// Beam-based range finder model with ray casting and multi-component noise model
// Components: z_hit (Gaussian), z_short (Exponential), z_max (Delta), z_rand (Uniform)
// ================================================================================================

#include "mcl_pkg/sensor_model/beam_sensor_model.hpp"
#include "mcl_pkg/mcl.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

namespace mcl_pkg {
namespace sensor_model {

/**
 * @brief Precomputes beam sensor model lookup table for fast likelihood evaluation
 */
void precompute_beam_sensor_model(MCL* node)
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
 * @brief Calculate particle weights using beam sensor model with ray casting
 */
void beam_sensor_update(MCL* node,
                       const Eigen::MatrixXd& proposal_dist,
                       const std::vector<float>& obs,
                       std::vector<double>& weights)
{
    const int num_rays = node->downsampled_angles_.size();
    const int total_queries = node->MAX_PARTICLES * num_rays;

    // Initialize arrays on first call
    initialize_beam_sensor_arrays(node, num_rays, total_queries);

    // Timing for performance monitoring
    auto query_prep_start = std::chrono::high_resolution_clock::now();

    // Generate ray casting queries
    generate_beam_ray_queries(node, proposal_dist, num_rays);

    node->timing_stats_.query_prep_time += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - query_prep_start).count();

    // Perform ray casting (if parallel raycasting enabled)
    if (node->USE_PARALLEL_RAYCASTING) {
        node->ranges_ = calc_range_many(node, node->queries_);

        // Calculate weights using lookup table
        auto sensor_eval_start = std::chrono::high_resolution_clock::now();
        calculate_beam_particle_weights(node, obs, num_rays, weights);
        node->timing_stats_.sensor_model_time += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - sensor_eval_start).count();
    }
}

/**
 * @brief Initialize sensor arrays for beam model computation
 */
void initialize_beam_sensor_arrays(MCL* node, int num_rays, int total_queries)
{
    if (node->first_sensor_update_) {
        node->queries_ = Eigen::MatrixXd::Zero(total_queries, 3);
        node->ranges_.resize(total_queries);

        // Pre-allocate sensor model vectors
        node->obs_px_.resize(num_rays);
        node->ranges_px_.resize(total_queries);

        // Pre-compute tiled angles for efficiency
        node->tiled_angles_.resize(total_queries);
        for (int i = 0; i < node->MAX_PARTICLES; ++i) {
            std::copy(node->downsampled_angles_.begin(), node->downsampled_angles_.end(),
                     node->tiled_angles_.begin() + i * num_rays);
        }
        node->first_sensor_update_ = false;
    }
}

/**
 * @brief Generate ray queries for batch ray casting
 */
void generate_beam_ray_queries(MCL* node,
                               const Eigen::MatrixXd& proposal_dist,
                               int num_rays)
{
    for (int i = 0; i < node->MAX_PARTICLES; ++i) {
        const int base_idx = i * num_rays;
        const double x = proposal_dist(i, 0);
        const double y = proposal_dist(i, 1);
        const double theta = proposal_dist(i, 2);

        for (int j = 0; j < num_rays; ++j) {
            const int idx = base_idx + j;
            node->queries_(idx, 0) = x;
            node->queries_(idx, 1) = y;
            node->queries_(idx, 2) = theta + node->downsampled_angles_[j];
        }
    }
}

/**
 * @brief Calculate particle weights using lookup table
 */
void calculate_beam_particle_weights(MCL* node,
                                     const std::vector<float>& obs,
                                     int num_rays,
                                     std::vector<double>& weights)
{
    // Thread-safe access to map resolution
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(node->map_lock_);
        local_map = node->map_msg_;
    }
    if (!local_map) return;
    double resolution = local_map->info.resolution;

    // Convert observations to pixel units (pre-allocated)
    for (size_t i = 0; i < obs.size(); ++i) {
        node->obs_px_[i] = std::min(static_cast<double>(node->MAX_RANGE_PX_), obs[i] / resolution);
    }

    // Convert expected ranges to pixel units (pre-allocated)
    for (size_t i = 0; i < node->ranges_.size(); ++i) {
        node->ranges_px_[i] = std::min(static_cast<double>(node->MAX_RANGE_PX_), node->ranges_[i] / resolution);
    }

    // Compute particle weights using pre-computed sensor model lookup table
    for (int i = 0; i < node->MAX_PARTICLES; ++i) {
        double weight = 1.0;
        const int base_idx = i * num_rays;

        for (int j = 0; j < num_rays; ++j) {
            const int obs_idx = std::max(0, std::min(static_cast<int>(std::round(node->obs_px_[j])), node->MAX_RANGE_PX_));
            const int range_idx = std::max(0, std::min(static_cast<int>(std::round(node->ranges_px_[base_idx + j])), node->MAX_RANGE_PX_));

            weight *= node->sensor_model_table_(obs_idx, range_idx);
        }

        // Apply squash factor AFTER computing the product - optimized
        if (weight > 0.0) {
            weights[i] = std::exp(node->INV_SQUASH_FACTOR * std::log(weight));
        } else {
            weights[i] = 0.0;
        }
    }
}

// ================================================================================================
// RAY CASTING FUNCTIONS
// ================================================================================================

/**
 * @brief Performs batch ray casting for multiple queries
 */
std::vector<float> calc_range_many(MCL* node, const Eigen::MatrixXd &queries)
{
    auto raycast_start = std::chrono::high_resolution_clock::now();

    std::vector<float> results(queries.rows());

    // Get map once for all ray casting operations
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(node->map_lock_);
        local_map = node->map_msg_;
    }

    if (!local_map || !node->map_initialized_) {
        std::fill(results.begin(), results.end(), node->MAX_RANGE_METERS);
        return results;
    }

    if (node->USE_PARALLEL_RAYCASTING) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
        }
    } else {
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
        }
    }

    auto raycast_end = std::chrono::high_resolution_clock::now();
    node->timing_stats_.ray_casting_time += std::chrono::duration<double, std::milli>(raycast_end - raycast_start).count();

    return results;
}

/**
 * @brief Casts single ray to find obstacle distance
 */
float cast_ray(double x, double y, double angle,
               const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
               int max_range_px)
{
    if (!local_map)
        return 10.0;  // Default MAX_RANGE_METERS

    double resolution = local_map->info.resolution;
    double origin_x = local_map->info.origin.position.x;
    double origin_y = local_map->info.origin.position.y;

    double dx = std::cos(angle) * resolution;
    double dy = std::sin(angle) * resolution;

    for (int step = 0; step < max_range_px; ++step)
    {
        x += dx;
        y += dy;

        // World to grid coordinate transformation
        int grid_x = static_cast<int>((x - origin_x) / resolution);
        int grid_y = static_cast<int>((y - origin_y) / resolution);

        // Map boundary collision
        if (grid_x < 0 || grid_x >= static_cast<int>(local_map->info.width) || grid_y < 0 ||
            grid_y >= static_cast<int>(local_map->info.height))
        {
            return step * resolution;
        }

        // Check for obstacles
        int map_idx = grid_y * local_map->info.width + grid_x;
        if (map_idx >= 0 && map_idx < static_cast<int>(local_map->data.size()))
        {
            if (local_map->data[map_idx] > 50)
            {
                return step * resolution;
            }
        }
    }

    return max_range_px * resolution;
}

} // namespace sensor_model
} // namespace mcl_pkg
