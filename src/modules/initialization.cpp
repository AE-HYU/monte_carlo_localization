#include "particle_filter_cpp/modules/initialization.hpp"
#include <cmath>
#include <algorithm>

namespace particle_filter_cpp
{
namespace modules
{

ParticleInitializer::ParticleInitializer(std::mt19937& rng)
    : rng_(rng), uniform_(0.0, 1.0), normal_(0.0, 1.0)
{
}

void ParticleInitializer::initialize_global(ParticleSet& particles, WeightVector& weights,
                                          const MapInfo& map_info)
{
    if (particles.empty()) return;
    
    // Get free space points from map
    auto free_points = get_free_space_points(map_info);
    
    if (free_points.empty()) {
        // Fallback: initialize all particles at origin
        for (auto& particle : particles) {
            particle.x = 0.0;
            particle.y = 0.0;
            particle.theta = uniform_(rng_) * 2.0 * M_PI - M_PI;
            particle.weight = 1.0 / particles.size();
        }
    } else {
        // Randomly sample from free space
        std::uniform_int_distribution<size_t> point_dist(0, free_points.size() - 1);
        
        for (auto& particle : particles) {
            auto point = free_points[point_dist(rng_)];
            particle.x = point.x();
            particle.y = point.y();
            particle.theta = uniform_(rng_) * 2.0 * M_PI - M_PI;
            particle.weight = 1.0 / particles.size();
        }
    }
    
    // Initialize weights uniformly
    double uniform_weight = 1.0 / particles.size();
    std::fill(weights.begin(), weights.end(), uniform_weight);
}

void ParticleInitializer::initialize_pose(ParticleSet& particles, WeightVector& weights,
                                        const Eigen::Vector3d& pose,
                                        const Eigen::Vector3d& std_dev)
{
    if (particles.empty()) return;
    
    // Create normal distributions for each dimension
    std::normal_distribution<double> x_dist(pose[0], std_dev[0]);
    std::normal_distribution<double> y_dist(pose[1], std_dev[1]);
    std::normal_distribution<double> theta_dist(pose[2], std_dev[2]);
    
    for (auto& particle : particles) {
        particle.x = x_dist(rng_);
        particle.y = y_dist(rng_);
        particle.theta = theta_dist(rng_);
        particle.weight = 1.0 / particles.size();
        
        // Normalize angle
        while (particle.theta > M_PI) particle.theta -= 2.0 * M_PI;
        while (particle.theta < -M_PI) particle.theta += 2.0 * M_PI;
    }
    
    // Initialize weights uniformly
    double uniform_weight = 1.0 / particles.size();
    std::fill(weights.begin(), weights.end(), uniform_weight);
}

std::vector<Eigen::Vector2d> ParticleInitializer::get_free_space_points(const MapInfo& map_info)
{
    std::vector<Eigen::Vector2d> free_points;
    
    for (int y = 0; y < map_info.height; ++y) {
        for (int x = 0; x < map_info.width; ++x) {
            if (is_free_space(x, y, map_info)) {
                // Convert map coordinates to world coordinates
                Eigen::Vector3d map_coord(x, y, 0.0);
                Eigen::Vector3d world_coord = map_to_world(map_coord, map_info);
                free_points.emplace_back(world_coord.x(), world_coord.y());
            }
        }
    }
    
    return free_points;
}

bool ParticleInitializer::is_free_space(int x, int y, const MapInfo& map_info)
{
    if (x < 0 || x >= map_info.width || y < 0 || y >= map_info.height) {
        return false;
    }
    
    int index = y * map_info.width + x;
    return map_info.data[index] == 0; // 0 = free space in occupancy grid
}

Eigen::Vector3d ParticleInitializer::map_to_world(const Eigen::Vector3d& map_coord, const MapInfo& map_info)
{
    double scale = map_info.resolution;
    double angle = map_info.origin[2]; // theta component of origin
    
    // Apply rotation
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);
    double x_rot = cos_a * map_coord.x() - sin_a * map_coord.y();
    double y_rot = sin_a * map_coord.x() + cos_a * map_coord.y();
    
    // Apply scale and translation
    double x_world = x_rot * scale + map_info.origin.x();
    double y_world = y_rot * scale + map_info.origin.y();
    double theta_world = map_coord.z() + angle;
    
    // Normalize angle
    while (theta_world > M_PI) theta_world -= 2.0 * M_PI;
    while (theta_world < -M_PI) theta_world += 2.0 * M_PI;
    
    return Eigen::Vector3d(x_world, y_world, theta_world);
}

} // namespace modules
} // namespace particle_filter_cpp