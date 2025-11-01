// ================================================================================================
// PARAMETER MANAGER - Parameter initialization, validation, and dynamic reconfiguration
// ================================================================================================

#include "mcl_pkg/parameter_manager.hpp"
#include "mcl_pkg/mcl.hpp"

#include <cmath>

namespace mcl_pkg
{
namespace parameter_manager
{

void initParameters(MCL* node)
{
    // === PARAMETER DECLARATIONS ===
    // Core algorithm parameters
    node->declare_parameter("angle_step", 18);
    node->declare_parameter("max_particles", 4000);
    node->declare_parameter("min_particles", 500);
    node->declare_parameter("max_viz_particles", 60);
    node->declare_parameter("squash_factor", 2.2);
    node->declare_parameter("max_range", 12.0);
    node->declare_parameter("max_pose_range", 10000.0);

    // Sensor model parameters
    node->declare_parameter("sensor_model_type", "beam");  // "beam" or "likelihood_field"

    // Beam model parameters (4-component: z_hit + z_short + z_max + z_rand = 1.0)
    node->declare_parameter("beam_z_hit", 0.85);
    node->declare_parameter("beam_z_short", 0.06);
    node->declare_parameter("beam_z_max", 0.06);
    node->declare_parameter("beam_z_rand", 0.03);
    node->declare_parameter("beam_sigma_hit", 0.2);

    // Likelihood field model parameters (2-component: z_hit + z_rand = 1.0)
    node->declare_parameter("likelihood_z_hit", 0.90);
    node->declare_parameter("likelihood_z_rand", 0.10);
    node->declare_parameter("likelihood_sigma", 0.1);

    // Motion model parameters
    node->declare_parameter("motion_model_type", "simple");  // "simple" or "odometry" (RTR)

    // Simple motion model (fixed Gaussian noise)
    node->declare_parameter("motion_dispersion_x", 0.02);
    node->declare_parameter("motion_dispersion_y", 0.01);
    node->declare_parameter("motion_dispersion_theta", 0.05);

    // Odometry motion model (RTR with alpha parameters)
    node->declare_parameter("alpha1", 0.5);  // rotation → rotation
    node->declare_parameter("alpha2", 0.02);  // translation → rotation
    node->declare_parameter("alpha3", 1.0);  // translation → translation
    node->declare_parameter("alpha4", 0.1);  // rotation → translation
    node->declare_parameter("small_trans_threshold", 0.001);  // Small motion threshold (m)
    node->declare_parameter("small_rot_threshold", 0.001);    // Small rotation threshold (rad)

    // Robot geometry
    node->declare_parameter("wheelbase", 0.324);

    // Velocity estimation filter
    node->declare_parameter("velocity_filter_alpha", 0.3);  // Low-pass filter coefficient (0-1)

    // Resampling parameters
    node->declare_parameter("use_adaptive_resampling", true);   // Enable adaptive resampling
    node->declare_parameter("ess_threshold", 0.5);              // ESS threshold (0.0-1.0)
    node->declare_parameter("resampling_type", "low_variance"); // "multinomial" or "low_variance"

    // ROS interface
    node->declare_parameter("scan_topic", "/scan");
    node->declare_parameter("odom_topic", "/odom");
    node->declare_parameter("publish_odom", true);
    node->declare_parameter("viz", true);
    node->declare_parameter("timer_frequency", 35.0);

    // Performance
    node->declare_parameter("use_parallel_raycasting", true);
    node->declare_parameter("num_threads", 0); // 0 = auto-detect
    node->declare_parameter("omp_schedule_type", "static"); // "static", "dynamic", "guided"
    node->declare_parameter("omp_chunk_size", 0); // 0 = default (no chunk size specified)

    // TF frames
    node->declare_parameter("map_frame", "map");
    node->declare_parameter("odom_frame", "odom");
    node->declare_parameter("base_frame", "base_link");
    node->declare_parameter("laser_frame", "laser");

    // TF publishing control
    node->declare_parameter("publish_map_odom_tf", true);

    // === PARAMETER RETRIEVAL ===
    // Core algorithm parameters
    node->ANGLE_STEP = node->get_parameter("angle_step").as_int();
    node->MAX_PARTICLES = node->get_parameter("max_particles").as_int();
    node->MIN_PARTICLES = node->get_parameter("min_particles").as_int();
    node->MAX_VIZ_PARTICLES = node->get_parameter("max_viz_particles").as_int();
    node->INV_SQUASH_FACTOR = 1.0 / node->get_parameter("squash_factor").as_double();
    node->MAX_RANGE_METERS = node->get_parameter("max_range").as_double();
    node->MAX_POSE_RANGE = node->get_parameter("max_pose_range").as_double();

    // Sensor model parameters
    node->SENSOR_MODEL_TYPE = node->get_parameter("sensor_model_type").as_string();

    // Load beam model parameters
    node->BEAM_Z_HIT = node->get_parameter("beam_z_hit").as_double();
    node->BEAM_Z_SHORT = node->get_parameter("beam_z_short").as_double();
    node->BEAM_Z_MAX = node->get_parameter("beam_z_max").as_double();
    node->BEAM_Z_RAND = node->get_parameter("beam_z_rand").as_double();
    node->BEAM_SIGMA_HIT = node->get_parameter("beam_sigma_hit").as_double();

    // Load likelihood field model parameters
    node->LIKELIHOOD_Z_HIT = node->get_parameter("likelihood_z_hit").as_double();
    node->LIKELIHOOD_Z_RAND = node->get_parameter("likelihood_z_rand").as_double();
    node->LIKELIHOOD_SIGMA = node->get_parameter("likelihood_sigma").as_double();

    // Motion model parameters
    node->MOTION_MODEL_TYPE = node->get_parameter("motion_model_type").as_string();

    // Simple motion model
    node->MOTION_DISPERSION_X = node->get_parameter("motion_dispersion_x").as_double();
    node->MOTION_DISPERSION_Y = node->get_parameter("motion_dispersion_y").as_double();
    node->MOTION_DISPERSION_THETA = node->get_parameter("motion_dispersion_theta").as_double();

    // Odometry motion model (RTR)
    node->ALPHA1 = node->get_parameter("alpha1").as_double();
    node->ALPHA2 = node->get_parameter("alpha2").as_double();
    node->ALPHA3 = node->get_parameter("alpha3").as_double();
    node->ALPHA4 = node->get_parameter("alpha4").as_double();
    node->SMALL_TRANS_THRESHOLD = node->get_parameter("small_trans_threshold").as_double();
    node->SMALL_ROT_THRESHOLD = node->get_parameter("small_rot_threshold").as_double();

    // Robot geometry
    node->WHEELBASE = node->get_parameter("wheelbase").as_double();

    // Velocity estimation filter
    node->velocity_filter_alpha_ = node->get_parameter("velocity_filter_alpha").as_double();

    // Resampling parameters
    node->USE_ADAPTIVE_RESAMPLING = node->get_parameter("use_adaptive_resampling").as_bool();
    node->ESS_THRESHOLD = node->get_parameter("ess_threshold").as_double();
    node->RESAMPLING_TYPE = node->get_parameter("resampling_type").as_string();

    // ROS interface
    node->PUBLISH_ODOM = node->get_parameter("publish_odom").as_bool();
    node->DO_VIZ = node->get_parameter("viz").as_bool();
    node->TIMER_FREQUENCY = node->get_parameter("timer_frequency").as_double();

    // Performance
    node->USE_PARALLEL_RAYCASTING = node->get_parameter("use_parallel_raycasting").as_bool();
    node->NUM_THREADS = node->get_parameter("num_threads").as_int();
    node->OMP_SCHEDULE_TYPE = node->get_parameter("omp_schedule_type").as_string();
    node->OMP_CHUNK_SIZE = node->get_parameter("omp_chunk_size").as_int();

    // TF frames
    node->MAP_FRAME = node->get_parameter("map_frame").as_string();
    node->ODOM_FRAME = node->get_parameter("odom_frame").as_string();
    node->BASE_FRAME = node->get_parameter("base_frame").as_string();
    node->LASER_FRAME = node->get_parameter("laser_frame").as_string();

    // TF publishing control
    node->PUBLISH_MAP_ODOM_TF = node->get_parameter("publish_map_odom_tf").as_bool();

    // === SEMANTIC VALIDATION ===

    // Validate sensor model type
    if (node->SENSOR_MODEL_TYPE != "beam" && node->SENSOR_MODEL_TYPE != "likelihood_field") {
        RCLCPP_WARN(node->get_logger(),
            "Invalid sensor_model_type '%s', using default 'beam'", node->SENSOR_MODEL_TYPE.c_str());
        node->SENSOR_MODEL_TYPE = "beam";
    }

    // Validate and normalize beam model weights (4-component)
    double beam_sum = node->BEAM_Z_HIT + node->BEAM_Z_SHORT + node->BEAM_Z_MAX + node->BEAM_Z_RAND;
    if (std::abs(beam_sum - 1.0) > 0.01) {
        RCLCPP_WARN(node->get_logger(),
            "Beam model weights sum to %.3f (expected 1.0), normalizing...", beam_sum);
        node->BEAM_Z_HIT /= beam_sum;
        node->BEAM_Z_SHORT /= beam_sum;
        node->BEAM_Z_MAX /= beam_sum;
        node->BEAM_Z_RAND /= beam_sum;
        RCLCPP_INFO(node->get_logger(),
            "Normalized beam: z_hit=%.3f, z_short=%.3f, z_max=%.3f, z_rand=%.3f",
            node->BEAM_Z_HIT, node->BEAM_Z_SHORT, node->BEAM_Z_MAX, node->BEAM_Z_RAND);
    }

    // Validate and normalize likelihood model weights (2-component)
    double likelihood_sum = node->LIKELIHOOD_Z_HIT + node->LIKELIHOOD_Z_RAND;
    if (std::abs(likelihood_sum - 1.0) > 0.01) {
        RCLCPP_WARN(node->get_logger(),
            "Likelihood model weights sum to %.3f (expected 1.0), normalizing...", likelihood_sum);
        node->LIKELIHOOD_Z_HIT /= likelihood_sum;
        node->LIKELIHOOD_Z_RAND /= likelihood_sum;
        RCLCPP_INFO(node->get_logger(),
            "Normalized likelihood: z_hit=%.3f, z_rand=%.3f",
            node->LIKELIHOOD_Z_HIT, node->LIKELIHOOD_Z_RAND);
    }

    // Set legacy parameters based on active sensor model
    if (node->SENSOR_MODEL_TYPE == "beam") {
        node->Z_HIT = node->BEAM_Z_HIT;
        node->Z_SHORT = node->BEAM_Z_SHORT;
        node->Z_MAX = node->BEAM_Z_MAX;
        node->Z_RAND = node->BEAM_Z_RAND;
        node->SIGMA_HIT = node->BEAM_SIGMA_HIT;
        RCLCPP_INFO(node->get_logger(),
            "Using beam model parameters: z_hit=%.3f, z_short=%.3f, z_max=%.3f, z_rand=%.3f, sigma=%.3f",
            node->Z_HIT, node->Z_SHORT, node->Z_MAX, node->Z_RAND, node->SIGMA_HIT);
    } else {
        node->Z_HIT = node->LIKELIHOOD_Z_HIT;
        node->Z_RAND = node->LIKELIHOOD_Z_RAND;
        node->Z_SHORT = 0.0;  // Not used in likelihood field model
        node->Z_MAX = 0.0;    // Not used in likelihood field model
        node->SIGMA_HIT = 0.0; // Not used in likelihood field model
        RCLCPP_INFO(node->get_logger(),
            "Using likelihood field model parameters: z_hit=%.3f, z_rand=%.3f, sigma=%.3f",
            node->Z_HIT, node->Z_RAND, node->LIKELIHOOD_SIGMA);
    }

    // Validate beam sigma_hit
    if (node->BEAM_SIGMA_HIT <= 0.0) {
        RCLCPP_WARN(node->get_logger(),
            "beam_sigma_hit must be positive, got %.3f. Using default 0.2", node->BEAM_SIGMA_HIT);
        node->BEAM_SIGMA_HIT = 0.2;
    }

    // Validate likelihood field sigma
    if (node->LIKELIHOOD_SIGMA <= 0.0) {
        RCLCPP_WARN(node->get_logger(),
            "likelihood_sigma must be positive, got %.3f. Using default 0.1", node->LIKELIHOOD_SIGMA);
        node->LIKELIHOOD_SIGMA = 0.1;
    }

    // Validate motion model type
    if (node->MOTION_MODEL_TYPE != "simple" && node->MOTION_MODEL_TYPE != "odometry") {
        RCLCPP_WARN(node->get_logger(),
            "Invalid motion_model_type '%s', using default 'simple'", node->MOTION_MODEL_TYPE.c_str());
        node->MOTION_MODEL_TYPE = "simple";
    }

    // Validate particle counts
    if (node->MAX_PARTICLES < 0) {
        RCLCPP_WARN(node->get_logger(),
            "max_particles is negative (%d), using default 2000", node->MAX_PARTICLES);
        node->MAX_PARTICLES = 2000;
    }

    if (node->MIN_PARTICLES < 0) {
        RCLCPP_WARN(node->get_logger(),
            "min_particles is negative (%d), using default 500", node->MIN_PARTICLES);
        node->MIN_PARTICLES = 500;
    }

    if (node->MIN_PARTICLES > node->MAX_PARTICLES) {
        RCLCPP_WARN(node->get_logger(),
            "min_particles (%d) > max_particles (%d), setting max = min",
            node->MIN_PARTICLES, node->MAX_PARTICLES);
        node->MAX_PARTICLES = node->MIN_PARTICLES;
    }

    // Validate angle step
    if (node->ANGLE_STEP <= 0) {
        RCLCPP_ERROR(node->get_logger(),
            "angle_step must be positive, got %d. Using default 18", node->ANGLE_STEP);
        node->ANGLE_STEP = 18;
    }

    // Validate ESS threshold
    if (node->ESS_THRESHOLD < 0.0 || node->ESS_THRESHOLD > 1.0) {
        RCLCPP_WARN(node->get_logger(),
            "ess_threshold must be between 0.0 and 1.0, got %.3f. Using default 0.5", node->ESS_THRESHOLD);
        node->ESS_THRESHOLD = 0.5;
    }

    // Validate resampling type
    if (node->RESAMPLING_TYPE != "multinomial" && node->RESAMPLING_TYPE != "low_variance") {
        RCLCPP_WARN(node->get_logger(),
            "Invalid resampling_type '%s', using default 'low_variance'", node->RESAMPLING_TYPE.c_str());
        node->RESAMPLING_TYPE = "low_variance";
    }


    // Validate simple motion model parameters
    if (node->MOTION_DISPERSION_X < 0.0) {
        RCLCPP_WARN(node->get_logger(),
            "motion_dispersion_x negative, using absolute value");
        node->MOTION_DISPERSION_X = std::abs(node->MOTION_DISPERSION_X);
    }
    if (node->MOTION_DISPERSION_Y < 0.0) {
        RCLCPP_WARN(node->get_logger(),
            "motion_dispersion_y negative, using absolute value");
        node->MOTION_DISPERSION_Y = std::abs(node->MOTION_DISPERSION_Y);
    }
    if (node->MOTION_DISPERSION_THETA < 0.0) {
        RCLCPP_WARN(node->get_logger(),
            "motion_dispersion_theta negative, using absolute value");
        node->MOTION_DISPERSION_THETA = std::abs(node->MOTION_DISPERSION_THETA);
    }

    // Validate odometry motion model parameters (alpha)
    if (node->ALPHA1 < 0.0) {
        RCLCPP_WARN(node->get_logger(), "alpha1 negative, using absolute value");
        node->ALPHA1 = std::abs(node->ALPHA1);
    }
    if (node->ALPHA2 < 0.0) {
        RCLCPP_WARN(node->get_logger(), "alpha2 negative, using absolute value");
        node->ALPHA2 = std::abs(node->ALPHA2);
    }
    if (node->ALPHA3 < 0.0) {
        RCLCPP_WARN(node->get_logger(), "alpha3 negative, using absolute value");
        node->ALPHA3 = std::abs(node->ALPHA3);
    }
    if (node->ALPHA4 < 0.0) {
        RCLCPP_WARN(node->get_logger(), "alpha4 negative, using absolute value");
        node->ALPHA4 = std::abs(node->ALPHA4);
    }

    // Validate max range
    if (node->MAX_RANGE_METERS <= 0.0) {
        RCLCPP_WARN(node->get_logger(),
            "max_range must be positive, got %.3f. Using default 12.0", node->MAX_RANGE_METERS);
        node->MAX_RANGE_METERS = 12.0;
    }

    // Validate timer frequency
    if (node->TIMER_FREQUENCY <= 0.0) {
        RCLCPP_WARN(node->get_logger(),
            "timer_frequency must be positive, got %.3f. Using default 35.0", node->TIMER_FREQUENCY);
        node->TIMER_FREQUENCY = 35.0;
    }

    // Validate OpenMP schedule type
    if (node->OMP_SCHEDULE_TYPE != "static" &&
        node->OMP_SCHEDULE_TYPE != "dynamic" &&
        node->OMP_SCHEDULE_TYPE != "guided") {
        RCLCPP_WARN(node->get_logger(),
            "Invalid omp_schedule_type '%s', using default 'static'", node->OMP_SCHEDULE_TYPE.c_str());
        node->OMP_SCHEDULE_TYPE = "static";
    }

    // Validate OpenMP chunk size
    if (node->OMP_CHUNK_SIZE < 0) {
        RCLCPP_WARN(node->get_logger(),
            "omp_chunk_size cannot be negative (%d), using default 0", node->OMP_CHUNK_SIZE);
        node->OMP_CHUNK_SIZE = 0;
    }

    // Validate velocity filter alpha
    if (node->velocity_filter_alpha_ < 0.0 || node->velocity_filter_alpha_ > 1.0) {
        RCLCPP_WARN(node->get_logger(),
            "velocity_filter_alpha must be between 0.0 and 1.0, got %.3f. Using default 0.3",
            node->velocity_filter_alpha_);
        node->velocity_filter_alpha_ = 0.3;
    }

    RCLCPP_INFO(node->get_logger(),
        "Parameters validated - Particles: [%d-%d], AngleStep: %d, Frequency: %.1f Hz, Sensor: %s, OMP: %s(%d)",
        node->MIN_PARTICLES, node->MAX_PARTICLES, node->ANGLE_STEP, node->TIMER_FREQUENCY,
        node->SENSOR_MODEL_TYPE.c_str(), node->OMP_SCHEDULE_TYPE.c_str(), node->OMP_CHUNK_SIZE);
}

rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(
    MCL* node,
    const std::vector<rclcpp::Parameter> &parameters)
{
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";

    for (const auto &param : parameters) {
        const std::string &name = param.get_name();

        // Skip nested parameters (e.g., "qos_overrides.xxx")
        if (name.find('.') != std::string::npos) {
            continue;
        }

        RCLCPP_INFO(node->get_logger(), "Parameter change request: %s", name.c_str());

        // Motion model parameters
        if (name == "motion_dispersion_x") {
            node->MOTION_DISPERSION_X = std::abs(param.as_double());
            RCLCPP_INFO(node->get_logger(), "Updated motion_dispersion_x: %.3f", node->MOTION_DISPERSION_X);
        }
        else if (name == "motion_dispersion_y") {
            node->MOTION_DISPERSION_Y = std::abs(param.as_double());
            RCLCPP_INFO(node->get_logger(), "Updated motion_dispersion_y: %.3f", node->MOTION_DISPERSION_Y);
        }
        else if (name == "motion_dispersion_theta") {
            node->MOTION_DISPERSION_THETA = std::abs(param.as_double());
            RCLCPP_INFO(node->get_logger(), "Updated motion_dispersion_theta: %.3f", node->MOTION_DISPERSION_THETA);
        }

        // Sensor model parameters
        else if (name == "z_hit" || name == "z_short" || name == "z_max" || name == "z_rand") {
            // Update sensor model weights
            if (name == "z_hit") node->Z_HIT = param.as_double();
            else if (name == "z_short") node->Z_SHORT = param.as_double();
            else if (name == "z_max") node->Z_MAX = param.as_double();
            else if (name == "z_rand") node->Z_RAND = param.as_double();

            // Validate and normalize
            double sum = node->Z_HIT + node->Z_SHORT + node->Z_MAX + node->Z_RAND;
            if (std::abs(sum - 1.0) > 0.01) {
                node->Z_HIT /= sum;
                node->Z_SHORT /= sum;
                node->Z_MAX /= sum;
                node->Z_RAND /= sum;
                RCLCPP_WARN(node->get_logger(),
                    "Sensor weights normalized: z_hit=%.3f, z_short=%.3f, z_max=%.3f, z_rand=%.3f",
                    node->Z_HIT, node->Z_SHORT, node->Z_MAX, node->Z_RAND);
            }

            // Recompute sensor model lookup table - requires access to map_initialized_ and precompute_sensor_model()
            // Note: This will be called via friend/public access
        }
        else if (name == "sigma_hit") {
            if (param.as_double() > 0) {
                node->SIGMA_HIT = param.as_double();
                RCLCPP_INFO(node->get_logger(), "Updated sigma_hit: %.3f (sensor model recompute needed)", node->SIGMA_HIT);
            } else {
                result.successful = false;
                result.reason = "sigma_hit must be positive";
            }
        }

        // Max range
        else if (name == "max_range") {
            if (param.as_double() > 0) {
                node->MAX_RANGE_METERS = param.as_double();
                RCLCPP_INFO(node->get_logger(), "Updated max_range: %.3f (sensor model recompute needed)", node->MAX_RANGE_METERS);
            } else {
                result.successful = false;
                result.reason = "max_range must be positive";
            }
        }

        // Visualization
        else if (name == "viz") {
            node->DO_VIZ = param.as_bool();
            RCLCPP_INFO(node->get_logger(), "Updated viz: %s", node->DO_VIZ ? "true" : "false");
        }

        // Max viz particles
        else if (name == "max_viz_particles") {
            node->MAX_VIZ_PARTICLES = param.as_int();
            RCLCPP_INFO(node->get_logger(), "Updated max_viz_particles: %d", node->MAX_VIZ_PARTICLES);
        }
    }

    return result;
}

} // namespace parameter_manager
} // namespace mcl_pkg
