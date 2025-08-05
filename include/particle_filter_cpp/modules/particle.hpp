#ifndef PARTICLE_FILTER_CPP__MODULES__PARTICLE_HPP_
#define PARTICLE_FILTER_CPP__MODULES__PARTICLE_HPP_

#include <vector>
#include <eigen3/Eigen/Dense>

namespace particle_filter_cpp
{
namespace modules
{

struct Particle
{
    double x;
    double y;
    double theta;
    double weight;
    
    Particle() : x(0.0), y(0.0), theta(0.0), weight(0.0) {}
    Particle(double x_, double y_, double theta_, double weight_ = 1.0) 
        : x(x_), y(y_), theta(theta_), weight(weight_) {}
    
    Eigen::Vector3d to_vector() const {
        return Eigen::Vector3d(x, y, theta);
    }
    
    void from_vector(const Eigen::Vector3d& vec) {
        x = vec[0];
        y = vec[1];
        theta = vec[2];
    }
};

using ParticleSet = std::vector<Particle>;
using WeightVector = std::vector<double>;

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__PARTICLE_HPP_