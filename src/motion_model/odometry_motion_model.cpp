// ================================================================================================
// ODOMETRY MOTION MODEL - Implementation
// ================================================================================================
// RTR (Rotation-Translation-Rotation) based motion model from Probabilistic Robotics
// Chapter 5.4, Table 5.6: sample_motion_model_odometry
// Reference: Thrun, Burgard, Fox - "Probabilistic Robotics" (2005), pp. 134-136
// ================================================================================================

#include "mcl_pkg/motion_model/odometry_motion_model.hpp"
#include "mcl_pkg/mcl.hpp"
#include <cmath>
#include <vector>

namespace mcl_pkg {
namespace motion_model {

/**
 * @brief Applies RTR-based odometry motion model to particles
 *
 * Implements sample_motion_model_odometry from Probabilistic Robotics.
 * Decomposes motion into rotation1 → translation → rotation2 sequence
 * with motion-proportional Gaussian noise.
 */
void odometry_motion_update(MCL* node,
                            Eigen::MatrixXd& proposal_dist,
                            const Eigen::Vector3d& action)
{
    // Extract motion in local frame: [forward, lateral, rotation]
    double delta_trans_local = std::sqrt(action[0] * action[0] + action[1] * action[1]);
    double delta_rot_total = action[2];

    // If motion is too small, apply minimal noise to prevent discontinuity
    if (delta_trans_local < node->SMALL_TRANS_THRESHOLD && std::abs(delta_rot_total) < node->SMALL_ROT_THRESHOLD) {
        // Apply small random noise to maintain particle diversity
        std::lock_guard<std::mutex> lock(node->rng_lock_);
        for (int i = 0; i < node->MAX_PARTICLES; ++i) {
            double noise_x = node->normal_dist_(node->rng_) * 0.01;  // 1cm std (increased from 1mm)
            double noise_y = node->normal_dist_(node->rng_) * 0.01;  // 1cm std
            double noise_theta = node->normal_dist_(node->rng_) * 0.01;  // ~0.57 deg std (increased)

            proposal_dist(i, 0) += noise_x;
            proposal_dist(i, 1) += noise_y;
            proposal_dist(i, 2) += noise_theta;
        }
        return;
    }

    // ============================================================================
    // RTR Decomposition
    // ============================================================================

    // Normalize angles to [-π, π]
    auto normalize_angle = [](double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    };

    // rot1: Initial rotation to face the direction of translation
    double delta_rot1 = 0.0;
    if (delta_trans_local > 1e-4) {
        delta_rot1 = std::atan2(action[1], action[0]);
    }

    // trans: Translation distance (sign indicates forward/backward)
    double delta_trans = delta_trans_local;

    // rot2: Final rotation to achieve target heading
    double delta_rot2 = normalize_angle(delta_rot_total - delta_rot1);

    // ============================================================================
    // Reverse motion handling (similar to AMCL)
    // ============================================================================
    // If delta_rot1 is closer to ±π than to 0, the robot is likely reversing.
    // In this case, we flip the direction and reduce rotation uncertainty.

    bool is_reverse = std::abs(delta_rot1) > M_PI / 2.0;

    if (is_reverse) {
        // Flip the translation direction (robot moving backward)
        delta_trans = -delta_trans;

        // Adjust rot1 by ±π to point in the opposite direction
        if (delta_rot1 > 0) {
            delta_rot1 -= M_PI;
        } else {
            delta_rot1 += M_PI;
        }

        // Recalculate rot2 with adjusted rot1
        delta_rot2 = normalize_angle(delta_rot_total - delta_rot1);
    }

    // Normalize rot1 after potential reverse adjustment
    delta_rot1 = normalize_angle(delta_rot1);

    // ============================================================================
    // Generate noise samples for all particles (with RNG lock)
    // ============================================================================

    std::vector<double> noise_rot1_values(node->MAX_PARTICLES);
    std::vector<double> noise_trans_values(node->MAX_PARTICLES);
    std::vector<double> noise_rot2_values(node->MAX_PARTICLES);

    {
        std::lock_guard<std::mutex> lock(node->rng_lock_);

        for (int i = 0; i < node->MAX_PARTICLES; ++i) {
            // Alpha parameters: motion-proportional noise standard deviations
            // α1: rotation → rotation noise
            // α2: translation → rotation noise
            // α3: translation → translation noise
            // α4: rotation → translation noise

            // Linear approximation for standard deviation (as in Probabilistic Robotics)
            // This avoids the sqrt(α*x²) = |α*x| computation
            double std_rot1 = node->ALPHA1 * std::abs(delta_rot1) +
                             node->ALPHA2 * std::abs(delta_trans);
            double std_trans = node->ALPHA3 * std::abs(delta_trans) +
                              node->ALPHA4 * (std::abs(delta_rot1) + std::abs(delta_rot2));
            double std_rot2 = node->ALPHA1 * std::abs(delta_rot2) +
                             node->ALPHA2 * std::abs(delta_trans);

            // Sample from Gaussian with linear approximation of std dev
            noise_rot1_values[i] = node->normal_dist_(node->rng_) * std_rot1;
            noise_trans_values[i] = node->normal_dist_(node->rng_) * std_trans;
            noise_rot2_values[i] = node->normal_dist_(node->rng_) * std_rot2;
        }
    }

    // ============================================================================
    // Apply noisy motion to each particle
    // ============================================================================

    for (int i = 0; i < node->MAX_PARTICLES; ++i) {
        // Apply noise to motion parameters
        double delta_rot1_hat = delta_rot1 - noise_rot1_values[i];
        double delta_trans_hat = delta_trans - noise_trans_values[i];
        double delta_rot2_hat = delta_rot2 - noise_rot2_values[i];

        // Get current particle pose
        double x = proposal_dist(i, 0);
        double y = proposal_dist(i, 1);
        double theta = proposal_dist(i, 2);

        // Apply RTR motion
        // 1. Rotate to face translation direction
        theta += delta_rot1_hat;

        // 2. Translate (forward if positive, backward if negative)
        x += delta_trans_hat * std::cos(theta);
        y += delta_trans_hat * std::sin(theta);

        // 3. Rotate to final heading
        theta += delta_rot2_hat;

        // Normalize final angle
        theta = normalize_angle(theta);

        // Update particle
        proposal_dist(i, 0) = x;
        proposal_dist(i, 1) = y;
        proposal_dist(i, 2) = theta;
    }
}

} // namespace motion_model
} // namespace mcl_pkg
