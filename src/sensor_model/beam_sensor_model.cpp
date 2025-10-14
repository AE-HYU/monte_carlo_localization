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

    double resolution = local_map->info.resolution;
    int table_width = node->MAX_RANGE_PX_ + 1;
    node->sensor_model_table_ = Eigen::MatrixXd::Zero(table_width, table_width);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert SIGMA_HIT from meters to pixels
    double sigma_hit_px = node->SIGMA_HIT / resolution;

    RCLCPP_INFO(node->get_logger(),
                "Building beam sensor model: sigma_hit=%.3fm (%.2fpx), resolution=%.3fm",
                node->SIGMA_HIT, sigma_hit_px, resolution);

    // Build lookup table
    for (int d = 0; d < table_width; ++d)  // d = expected range (pixels)
    {
        // First pass: calculate normalization factors for Z_HIT and Z_SHORT
        double sum_z_hit = 0.0;
        double sum_z_short = 0.0;

        for (int r = 0; r < table_width; ++r)
        {
            double z = static_cast<double>(r - d);  // Error in pixels

            // Z_HIT: Gaussian (needs normalization)
            double prob_hit = std::exp(-(z * z) / (2.0 * sigma_hit_px * sigma_hit_px))
                            / (sigma_hit_px * std::sqrt(2.0 * M_PI));
            sum_z_hit += prob_hit;

            // Z_SHORT: Exponential (needs normalization)
            if (r < d && d > 0)
            {
                double prob_short = 2.0 * (d - r) / static_cast<double>(d);
                sum_z_short += prob_short;
            }
        }

        // Normalization factors
        double norm_hit = (sum_z_hit > 0) ? (1.0 / sum_z_hit) : 0.0;
        double norm_short = (sum_z_short > 0) ? (1.0 / sum_z_short) : 0.0;

        // Second pass: build normalized sensor model
        for (int r = 0; r < table_width; ++r)  // r = observed range (pixels)
        {
            double z = static_cast<double>(r - d);

            // === Z_HIT: Normalized Gaussian ===
            double prob_hit = std::exp(-(z * z) / (2.0 * sigma_hit_px * sigma_hit_px))
                            / (sigma_hit_px * std::sqrt(2.0 * M_PI)) * norm_hit;

            // === Z_SHORT: Normalized Exponential ===
            double prob_short = 0.0;
            if (r < d && d > 0)
            {
                prob_short = 2.0 * (d - r) / static_cast<double>(d) * norm_short;
            }

            // === Z_MAX: Delta function (already normalized) ===
            double prob_max = (r == node->MAX_RANGE_PX_) ? 1.0 : 0.0;

            // === Z_RAND: Uniform (already normalized) ===
            double prob_rand = (r < node->MAX_RANGE_PX_) ? (1.0 / node->MAX_RANGE_PX_) : 0.0;

            // Combine with mixture weights (Z_HIT + Z_SHORT + Z_MAX + Z_RAND = 1.0)
            double prob = node->Z_HIT * prob_hit +
                          node->Z_SHORT * prob_short +
                          node->Z_MAX * prob_max +
                          node->Z_RAND * prob_rand;

            node->sensor_model_table_(r, d) = prob;
        }

        // Normalize column to ensure sum to 1.0 (numerical stability)
        double col_sum = node->sensor_model_table_.col(d).sum();
        if (col_sum > 1e-12)
            node->sensor_model_table_.col(d) /= col_sum;
        
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Verify normalization (column sum should be ≈ 1.0)
    double col_sum = node->sensor_model_table_.col(table_width / 2).sum();

    RCLCPP_INFO(node->get_logger(),
                "Beam sensor model ready (%ld ms) - %dx%d table, column sum check: %.6f",
                duration.count(), table_width, table_width, col_sum);
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

    auto query_prep_end = std::chrono::high_resolution_clock::now();
    double query_time = std::chrono::duration<double, std::milli>(query_prep_end - query_prep_start).count();
    node->timing_stats_.query_prep_time += query_time;
    node->timing_stats_.current_query_prep_time = query_time;

    // Perform ray casting (if parallel raycasting enabled)
    if (node->USE_PARALLEL_RAYCASTING) {
        node->ranges_ = calc_range_many(node, node->queries_);

        // Calculate weights using lookup table
        auto sensor_eval_start = std::chrono::high_resolution_clock::now();
        calculate_beam_particle_weights(node, obs, num_rays, weights);
        auto sensor_eval_end = std::chrono::high_resolution_clock::now();
        double sensor_time = std::chrono::duration<double, std::milli>(sensor_eval_end - sensor_eval_start).count();
        node->timing_stats_.sensor_model_time += sensor_time;
        node->timing_stats_.current_sensor_model_time = sensor_time;
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
        // Apply OpenMP scheduling based on parameters
        if (node->OMP_SCHEDULE_TYPE == "static") {
            if (node->OMP_CHUNK_SIZE > 0) {
                #pragma omp parallel for schedule(static, node->OMP_CHUNK_SIZE)
                for (int i = 0; i < queries.rows(); ++i) {
                    results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
                }
            } else {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < queries.rows(); ++i) {
                    results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
                }
            }
        } else if (node->OMP_SCHEDULE_TYPE == "dynamic") {
            if (node->OMP_CHUNK_SIZE > 0) {
                #pragma omp parallel for schedule(dynamic, node->OMP_CHUNK_SIZE)
                for (int i = 0; i < queries.rows(); ++i) {
                    results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
                }
            } else {
                #pragma omp parallel for schedule(dynamic)
                for (int i = 0; i < queries.rows(); ++i) {
                    results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
                }
            }
        } else if (node->OMP_SCHEDULE_TYPE == "guided") {
            if (node->OMP_CHUNK_SIZE > 0) {
                #pragma omp parallel for schedule(guided, node->OMP_CHUNK_SIZE)
                for (int i = 0; i < queries.rows(); ++i) {
                    results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
                }
            } else {
                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < queries.rows(); ++i) {
                    results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
                }
            }
        }
    } else {
        for (int i = 0; i < queries.rows(); ++i)
        {
            results[i] = cast_ray(queries(i, 0), queries(i, 1), queries(i, 2), local_map, node->MAX_RANGE_PX_);
        }
    }

    auto raycast_end = std::chrono::high_resolution_clock::now();
    double raycast_time = std::chrono::duration<double, std::milli>(raycast_end - raycast_start).count();
    node->timing_stats_.ray_casting_time += raycast_time;
    node->timing_stats_.current_ray_casting_time = raycast_time;

    return results;
}

/**
 * @brief Casts single ray to find obstacle distance using Bresenham line algorithm
 *
 * Bresenham's line algorithm provides:
 * - Faster execution (integer-only arithmetic, no floating point)
 * - More accurate pixel traversal (visits exactly the pixels the line crosses)
 * - Better cache locality (sequential grid access pattern)
 *
 * Optimizations:
 * - Uses integer distance calculation (avoids sqrt until final return)
 * - Accurate step count (max(dx,dy) not dx+dy)
 * - Early termination on obstacles
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
    int width = local_map->info.width;
    int height = local_map->info.height;

    // Convert start position to grid coordinates
    int x0 = static_cast<int>((x - origin_x) / resolution);
    int y0 = static_cast<int>((y - origin_y) / resolution);

    // Calculate end position in grid coordinates
    double x_end = x + std::cos(angle) * max_range_px * resolution;
    double y_end = y + std::sin(angle) * max_range_px * resolution;
    int x1 = static_cast<int>((x_end - origin_x) / resolution);
    int y1 = static_cast<int>((y_end - origin_y) / resolution);

    // Check if start position is out of bounds or occupied
    if (x0 < 0 || x0 >= width || y0 < 0 || y0 >= height) {
        return 0.0;
    }
    int start_idx = y0 * width + x0;
    if (local_map->data[start_idx] > 50) {
        return 0.0;
    }

    // Bresenham's line algorithm
    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    int x_curr = x0;
    int y_curr = y0;

    // Actual steps needed: max(dx, dy) for Bresenham
    // This is much more accurate than dx + dy
    int max_steps = std::max(dx, dy) + 1;

    while (max_steps-- > 0)
    {
        // Check boundary (first, before obstacle check for safety)
        if (x_curr < 0 || x_curr >= width || y_curr < 0 || y_curr >= height) {
            // Hit boundary - calculate distance using integer arithmetic
            int delta_x = x_curr - x0;
            int delta_y = y_curr - y0;
            return std::sqrt(delta_x * delta_x + delta_y * delta_y) * resolution;
        }

        // Check for obstacle
        int map_idx = y_curr * width + x_curr;
        if (local_map->data[map_idx] > 50) {
            // Hit obstacle - calculate distance using integer arithmetic
            int delta_x = x_curr - x0;
            int delta_y = y_curr - y0;
            return std::sqrt(delta_x * delta_x + delta_y * delta_y) * resolution;
        }

        // Check if reached end point (max range)
        if (x_curr == x1 && y_curr == y1) {
            break;
        }

        // Bresenham step
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x_curr += sx;
        }
        if (e2 < dx) {
            err += dx;
            y_curr += sy;
        }
    }

    // No obstacle found within max range
    return max_range_px * resolution;
}

} // namespace sensor_model
} // namespace mcl_pkg
