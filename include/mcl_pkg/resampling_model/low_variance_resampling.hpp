// ================================================================================================
// LOW VARIANCE SAMPLER - Systematic resampling with minimal variance
// ================================================================================================
// Deterministic systematic sampling with single random offset.
// Much lower variance than multinomial - better particle diversity.
// ================================================================================================

#ifndef MCL_PKG__RESAMPLING_MODEL__LOW_VARIANCE_RESAMPLING_HPP_
#define MCL_PKG__RESAMPLING_MODEL__LOW_VARIANCE_RESAMPLING_HPP_

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
 * @brief Low Variance Sampler (systematic resampling)
 *
 * Systematic resampling using a single random offset and uniform spacing.
 * Guarantees that particles are selected proportionally to their weights
 * with minimal variance.
 *
 * Algorithm:
 * 1. Generate random offset r ∈ [0, 1/M)
 * 2. For i = 0 to M-1:
 *    - u_i = r + i/M
 *    - Select particle j where cumulative_weight[j] > u_i
 *
 * Complexity: O(M) where M = number of samples (faster than multinomial!)
 *
 * Pros:
 * - Minimal variance (near-deterministic selection)
 * - Faster than multinomial O(M) vs O(M log N)
 * - Better particle diversity
 * - Reduces sample impoverishment
 *
 * Cons:
 * - Slightly more complex implementation
 * - Less randomness (can be a pro or con)
 *
 * Reference:
 * - Probabilistic Robotics (Thrun, Burgard, Fox), Table 4.4
 * - Doucet et al. (2001) "Sequential Monte Carlo Methods in Practice"
 *
 * @param node Pointer to MCL node
 * @param proposal_dist Proposal distribution (after motion + sensor update)
 * @param weights Normalized particle weights
 * @param resampled_particles Output resampled particles
 */
void low_variance_resample(
    MCL* node,
    const Eigen::MatrixXd& proposal_dist,
    const std::vector<double>& weights,
    Eigen::MatrixXd& resampled_particles);

} // namespace resampling_model
} // namespace mcl_pkg

#endif // MCL_PKG__RESAMPLING_MODEL__LOW_VARIANCE_RESAMPLING_HPP_
