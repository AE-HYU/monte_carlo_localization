#ifndef PARTICLE_FILTER_CPP__MODULES__COMMON_TYPES_HPP_
#define PARTICLE_FILTER_CPP__MODULES__COMMON_TYPES_HPP_

#include <eigen3/Eigen/Dense>
#include <vector>
#include <chrono>

namespace particle_filter_cpp
{
namespace modules
{

// Map information structure
struct MapInfo
{
    int width;
    int height;
    double resolution;
    Eigen::Vector3d origin;  // x, y, theta
    std::vector<int8_t> data;
    
    MapInfo() : width(0), height(0), resolution(0.05), origin(0, 0, 0) {}
    MapInfo(int w, int h, double res, const Eigen::Vector3d& orig, const std::vector<int8_t>& d)
        : width(w), height(h), resolution(res), origin(orig), data(d) {}
};

// Odometry data structure
struct OdometryData
{
    Eigen::Vector3d pose;      // x, y, theta
    Eigen::Vector3d velocity;  // vx, vy, omega
    double timestamp;
    
    OdometryData() : pose(0, 0, 0), velocity(0, 0, 0), timestamp(0.0) {}
    OdometryData(const Eigen::Vector3d& p, const Eigen::Vector3d& v, double t)
        : pose(p), velocity(v), timestamp(t) {}
};

// Motion model parameters
struct MotionModelParams
{
    double dispersion_x;
    double dispersion_y;
    double dispersion_theta;
    
    MotionModelParams() : dispersion_x(0.05), dispersion_y(0.025), dispersion_theta(0.25) {}
    MotionModelParams(double dx, double dy, double dt)
        : dispersion_x(dx), dispersion_y(dy), dispersion_theta(dt) {}
};

// Resampling parameters
struct ResamplingParams
{
    enum class Method
    {
        SYSTEMATIC,
        LOW_VARIANCE,
        MULTINOMIAL
    };
    
    Method method;
    double ess_threshold_ratio;
    bool adaptive;
    
    ResamplingParams() : method(Method::SYSTEMATIC), ess_threshold_ratio(0.5), adaptive(true) {}
};

// Pose estimation parameters
struct PoseEstimationParams
{
    enum class Method
    {
        WEIGHTED_AVERAGE,
        MAX_LIKELIHOOD,
        CLUSTERED_AVERAGE
    };
    
    Method method;
    double convergence_threshold;
    int max_clusters;
    
    PoseEstimationParams() : method(Method::WEIGHTED_AVERAGE), convergence_threshold(0.1), max_clusters(5) {}
};

// Extended MCL statistics with covariance
struct MCLStatistics
{
    double effective_sample_size;
    int resample_count;
    double pose_uncertainty;
    double computation_time_ms;
    bool converged;
    int iteration_count;
    Eigen::Matrix3d covariance;  // 3x3 covariance matrix for pose (x, y, theta)
    
    MCLStatistics() : effective_sample_size(0.0), resample_count(0), pose_uncertainty(0.0),
                      computation_time_ms(0.0), converged(false), iteration_count(0),
                      covariance(Eigen::Matrix3d::Identity()) {}
};

} // namespace modules
} // namespace particle_filter_cpp

#endif // PARTICLE_FILTER_CPP__MODULES__COMMON_TYPES_HPP_ 