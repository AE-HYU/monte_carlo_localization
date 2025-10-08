// ================================================================================================
// MULTINOMIAL RESAMPLING - Standard particle filter resampling method
// ================================================================================================
// Independent sampling with replacement based on particle weights.
// Simple but high variance - particles are selected independently.
// ================================================================================================

#ifndef MCL_PKG__RESAMPLING_MODEL__MULTINOMIAL_RESAMPLING_HPP_
#define MCL_PKG__RESAMPLING_MODEL__MULTINOMIAL_RESAMPLING_HPP_

#include <Eigen/Dense>
#include <random>
#include <vector>

namespace mcl_pkg
{

// Forward declaration
class MCL;

namespace resampling_model
{

/**
 * @brief Multinomial resampling (standard particle filter resampling)
 *
 * Independently samples M particles from the proposal distribution
 * according to their weights. Each particle is selected with probability
 * proportional to its weight.
 *
 * Complexity: O(M log N) where M = number of samples, N = number of particles
 *
 * Pros:
 * - Simple implementation
 * - Easy to understand
 * - Unbiased
 *
 * Cons:
 * - High variance (sample impoverishment)
 * - Can miss important particles due to randomness
 *
 * @param node Pointer to MCL node
 * @param proposal_dist Proposal distribution (after motion + sensor update)
 * @param weights Normalized particle weights
 * @param resampled_particles Output resampled particles
 */
void multinomial_resample(
    MCL* node,
    const Eigen::MatrixXd& proposal_dist,
    const std::vector<double>& weights,
    Eigen::MatrixXd& resampled_particles);

} // namespace resampling_model
} // namespace mcl_pkg

#endif // MCL_PKG__RESAMPLING_MODEL__MULTINOMIAL_RESAMPLING_HPP_
