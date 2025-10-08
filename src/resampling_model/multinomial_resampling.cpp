// ================================================================================================
// MULTINOMIAL RESAMPLING IMPLEMENTATION
// ================================================================================================

#include "mcl_pkg/resampling_model/multinomial_resampling.hpp"
#include "mcl_pkg/mcl.hpp"

namespace mcl_pkg
{
namespace resampling_model
{

void multinomial_resample(
    MCL* node,
    const Eigen::MatrixXd& proposal_dist,
    const std::vector<double>& weights,
    Eigen::MatrixXd& resampled_particles)
{
    int num_particles = proposal_dist.rows();

    // Create discrete distribution from weights
    std::discrete_distribution<int> particle_dist(weights.begin(), weights.end());

    // Generate resample indices
    std::vector<int> resample_indices(num_particles);

    {
        std::lock_guard<std::mutex> lock(node->rng_lock_);
        for (int i = 0; i < num_particles; ++i)
        {
            resample_indices[i] = particle_dist(node->rng_);
        }
    }

    // Copy resampled particles
    for (int i = 0; i < num_particles; ++i)
    {
        resampled_particles.row(i) = proposal_dist.row(resample_indices[i]);
    }
}

} // namespace resampling_model
} // namespace mcl_pkg
