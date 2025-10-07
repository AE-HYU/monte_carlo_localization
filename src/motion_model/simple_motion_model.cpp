// ================================================================================================
// SIMPLE MOTION MODEL - Implementation
// ================================================================================================
// Simplified motion model with fixed Gaussian noise
// Directly transforms global pose changes to robot's local frame without RTR decomposition
// ================================================================================================

#include "mcl_pkg/motion_model/simple_motion_model.hpp"
#include "mcl_pkg/mcl.hpp"
#include <vector>

namespace mcl_pkg {
namespace motion_model {

/**
 * @brief Applies simple motion model to particles with fixed Gaussian noise
 */
void simple_motion_update(MCL* node,
                          Eigen::MatrixXd& proposal_dist,
                          const Eigen::Vector3d& action)
{
    // Vectorized motion model implementation
    // Transform the action into the coordinate space of each particle

    // Pre-compute trigonometric values for all particles (vectorized approach)
    Eigen::VectorXd cos_thetas = proposal_dist.col(2).array().cos();
    Eigen::VectorXd sin_thetas = proposal_dist.col(2).array().sin();

    // Apply motion transformation: local → global coordinates (vectorized)
    Eigen::VectorXd global_dx = cos_thetas * action[0] - sin_thetas * action[1];
    Eigen::VectorXd global_dy = sin_thetas * action[0] + cos_thetas * action[1];

    // Apply displacement
    proposal_dist.col(0) += global_dx;
    proposal_dist.col(1) += global_dy;
    proposal_dist.col(2).array() += action[2];

    // Add Gaussian noise - generate all noise values at once
    std::vector<double> noise_x_values(node->MAX_PARTICLES);
    std::vector<double> noise_y_values(node->MAX_PARTICLES);
    std::vector<double> noise_theta_values(node->MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(node->rng_lock_);
        for (int i = 0; i < node->MAX_PARTICLES; ++i)
        {
            noise_x_values[i] = node->normal_dist_(node->rng_) * node->MOTION_DISPERSION_X;
            noise_y_values[i] = node->normal_dist_(node->rng_) * node->MOTION_DISPERSION_Y;
            noise_theta_values[i] = node->normal_dist_(node->rng_) * node->MOTION_DISPERSION_THETA;
        }
    }

    // Apply noise without lock
    for (int i = 0; i < node->MAX_PARTICLES; ++i)
    {
        proposal_dist(i, 0) += noise_x_values[i];
        proposal_dist(i, 1) += noise_y_values[i];
        proposal_dist(i, 2) += noise_theta_values[i];
    }
}

} // namespace motion_model
} // namespace mcl_pkg
