// ================================================================================================
// LOW VARIANCE SAMPLER IMPLEMENTATION
// ================================================================================================

#include "mcl_pkg/resampling_model/low_variance_resampling.hpp"
#include "mcl_pkg/mcl.hpp"

namespace mcl_pkg
{
namespace resampling_model
{

void low_variance_resample(
    MCL* node,
    const Eigen::MatrixXd& proposal_dist,
    const std::vector<double>& weights,
    Eigen::MatrixXd& resampled_particles)
{
    int num_particles = proposal_dist.rows();

    // Generate random offset r ∈ [0, 1/M)
    std::uniform_real_distribution<double> uniform(0.0, 1.0 / num_particles);
    double r = 0.0;
    {
        std::lock_guard<std::mutex> lock(node->rng_lock_);
        r = uniform(node->rng_);
    }

    // Initialize cumulative weight and index
    double c = weights[0];
    int i = 0;

    // Systematic sampling with uniform spacing
    for (int m = 0; m < num_particles; ++m)
    {
        // Calculate position for m-th sample
        double u = r + static_cast<double>(m) / num_particles;

        // Find particle index where cumulative weight exceeds u
        while (u > c && i < num_particles - 1)
        {
            i++;
            c += weights[i];
        }

        // Select particle i for position m
        resampled_particles.row(m) = proposal_dist.row(i);
    }
}

} // namespace resampling_model
} // namespace mcl_pkg
