#ifndef PARTICLE_FILTER_CPP__MODULES__INITIALIZATION_HPP_
#define PARTICLE_FILTER_CPP__MODULES__INITIALIZATION_HPP_

#include "particle.hpp"
#include <eigen3/Eigen/Dense>
#include <random>

namespace particle_filter_cpp
{
namespace modules
{

struct MapInfo
{
    int width;
    int height;
    double resolution;
    Eigen::Vector3d origin;  // x, y, theta
    std::vector<int8_t> data;  // occupancy data
};

class ParticleInitializer
{
public:
    explicit ParticleInitializer(std::mt19937& rng);
    
    // Initialize particles uniformly across free space
    void initialize_global(ParticleSet& particles, WeightVector& weights, 
                          const MapInfo& map_info);
    
    // Initialize particles around a given pose with Gaussian distribution
    void initialize_pose(ParticleSet& particles, WeightVector& weights,
                        const Eigen::Vector3d& pose, 
                        const Eigen::Vector3d& std_dev);
    
    // Get free space points from map
    std::vector<Eigen::Vector2d> get_free_space_points(const MapInfo& map_info);
    
private:
    std::mt19937& rng_;
    std::uniform_real_distribution<double> uniform_;
    std::normal_distribution<double> normal_;
    
    bool is_free_space(int x, int y, const MapInfo& map_info);
    Eigen::Vector3d map_to_world(const Eigen::Vector3d& map_coord, const MapInfo& map_info);
};

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__INITIALIZATION_HPP_