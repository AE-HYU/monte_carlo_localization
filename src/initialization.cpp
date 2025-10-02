// ================================================================================================
// INITIALIZATION - Particle initialization functions
// ================================================================================================

#include "mcl_pkg/initialization.hpp"
#include "mcl_pkg/mcl.hpp"
#include "mcl_pkg/utils.hpp"

#include <random>

namespace mcl_pkg
{
namespace initialization
{

void initialize_particles_pose(MCL* node, const Eigen::Vector3d &pose)
{
    RCLCPP_INFO(node->get_logger(), "Initializing particles at [%.3f, %.3f, %.3f]", 
                pose[0], pose[1], pose[2]);

    std::lock_guard<std::mutex> lock(node->state_lock_);
    std::fill(node->weights_.begin(), node->weights_.end(), 1.0 / node->MAX_PARTICLES);

    // Use tighter distribution for faster convergence after manual pose setting
    double pos_std = 0.1;   // ±10cm
    double angle_std = 0.1; // ±5.7°

    // Generate all noise values at once
    std::vector<double> noise_x_values(node->MAX_PARTICLES);
    std::vector<double> noise_y_values(node->MAX_PARTICLES);
    std::vector<double> noise_theta_values(node->MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(node->rng_lock_);
        for (int i = 0; i < node->MAX_PARTICLES; ++i)
        {
            noise_x_values[i] = node->normal_dist_(node->rng_) * pos_std;
            noise_y_values[i] = node->normal_dist_(node->rng_) * pos_std;
            noise_theta_values[i] = node->normal_dist_(node->rng_) * angle_std;
        }
    }

    // Apply noise to particles without lock
    for (int i = 0; i < node->MAX_PARTICLES; ++i)
    {
        node->particles_(i, 0) = pose[0] + noise_x_values[i];
        node->particles_(i, 1) = pose[1] + noise_y_values[i];
        node->particles_(i, 2) = pose[2] + noise_theta_values[i];

        // Normalize angle
        node->particles_(i, 2) = utils::geometry::normalize_angle(node->particles_(i, 2));
    }
}

void initialize_global(MCL* node)
{
    if (!node->map_initialized_)
        return;

    RCLCPP_INFO(node->get_logger(), "Global initialization started");

    // 1. Copy map data once with minimal lock time
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
    {
        std::lock_guard<std::mutex> lock(node->map_lock_);
        local_map = node->map_msg_;
    }

    if (!local_map) return;

    // 2. Extract free space cells without lock (read-only operation)
    int height = local_map->info.height;
    int width = local_map->info.width;
    double resolution = local_map->info.resolution;
    double origin_x = local_map->info.origin.position.x;
    double origin_y = local_map->info.origin.position.y;

    std::vector<std::pair<int, int>> permissible_positions;
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            int idx = i * width + j;
            if (idx < static_cast<int>(local_map->data.size()) && local_map->data[idx] == 0)
            {
                permissible_positions.emplace_back(i, j);
            }
        }
    }

    if (permissible_positions.empty())
    {
        RCLCPP_ERROR(node->get_logger(), "No free space found in map!");
        return;
    }

    // 3. Generate particles without lock (using local data)
    std::uniform_int_distribution<int> pos_dist(0, permissible_positions.size() - 1);
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);

    // 4. Generate random values at once
    std::vector<int> indices(node->MAX_PARTICLES);
    std::vector<double> angles(node->MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(node->rng_lock_);
        for (int i = 0; i < node->MAX_PARTICLES; ++i)
        {
            indices[i] = pos_dist(node->rng_);
            angles[i] = angle_dist(node->rng_);
        }
    }

    // 5. Update particle state with minimal lock time
    {
        std::lock_guard<std::mutex> lock(node->state_lock_);

        for (int i = 0; i < node->MAX_PARTICLES; ++i)
        {
            auto pos = permissible_positions[indices[i]];
            node->particles_(i, 0) = pos.second * resolution + origin_x;
            node->particles_(i, 1) = pos.first * resolution + origin_y;
            node->particles_(i, 2) = angles[i];
        }

        std::fill(node->weights_.begin(), node->weights_.end(), 1.0 / node->MAX_PARTICLES);

        // Calculate expected pose from initialized particles
        Eigen::Vector3d initial_pose = node->expected_pose();

        RCLCPP_INFO(node->get_logger(),
                    "Global initialization complete - %d particles at pose: [%.3f, %.3f, %.3f]",
                    node->MAX_PARTICLES, initial_pose[0], initial_pose[1], initial_pose[2]);
    }
}

} // namespace initialization
} // namespace mcl_pkg
