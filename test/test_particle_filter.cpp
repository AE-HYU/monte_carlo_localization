#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include "particle_filter_cpp/particle_filter.hpp"
#include "particle_filter_cpp/utils.hpp"

class ParticleFilterTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        rclcpp::init(0, nullptr);
        node_ = std::make_shared<particle_filter_cpp::ParticleFilter>();
    }

    void TearDown() override
    {
        rclcpp::shutdown();
    }

    std::shared_ptr<particle_filter_cpp::ParticleFilter> node_;
};

TEST_F(ParticleFilterTest, NodeInitialization)
{
    ASSERT_TRUE(node_ != nullptr);
    EXPECT_EQ(node_->get_name(), std::string("particle_filter"));
}

TEST_F(ParticleFilterTest, ParameterLoading)
{
    // Test that parameters are loaded correctly
    int num_particles = node_->get_parameter("num_particles").as_int();
    EXPECT_GT(num_particles, 0);
    EXPECT_LE(num_particles, 10000); // Reasonable upper bound
}

class UtilsTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(UtilsTest, AngleNormalization)
{
    // Test angle normalization
    EXPECT_NEAR(particle_filter_cpp::utils::normalize_angle(0.0), 0.0, 1e-6);
    EXPECT_NEAR(particle_filter_cpp::utils::normalize_angle(M_PI), M_PI, 1e-6);
    EXPECT_NEAR(particle_filter_cpp::utils::normalize_angle(-M_PI), -M_PI, 1e-6);
    EXPECT_NEAR(particle_filter_cpp::utils::normalize_angle(2 * M_PI), 0.0, 1e-6);
    EXPECT_NEAR(particle_filter_cpp::utils::normalize_angle(-2 * M_PI), 0.0, 1e-6);
    EXPECT_NEAR(particle_filter_cpp::utils::normalize_angle(3 * M_PI), -M_PI, 1e-6);
}

TEST_F(UtilsTest, QuaternionConversion)
{
    // Test quaternion to yaw conversion
    auto quat = particle_filter_cpp::utils::yaw_to_quaternion(M_PI / 2);
    double yaw = particle_filter_cpp::utils::quaternion_to_yaw(quat);
    EXPECT_NEAR(yaw, M_PI / 2, 1e-6);
    
    // Test round trip
    double original_yaw = -M_PI / 4;
    auto q = particle_filter_cpp::utils::yaw_to_quaternion(original_yaw);
    double converted_yaw = particle_filter_cpp::utils::quaternion_to_yaw(q);
    EXPECT_NEAR(converted_yaw, original_yaw, 1e-6);
}

TEST_F(UtilsTest, WeightNormalization)
{
    std::vector<double> weights = {1.0, 2.0, 3.0, 4.0};
    auto normalized = particle_filter_cpp::utils::normalize_weights(weights);
    
    // Check that weights sum to 1
    double sum = 0.0;
    for (double w : normalized) {
        sum += w;
    }
    EXPECT_NEAR(sum, 1.0, 1e-6);
    
    // Check relative proportions
    EXPECT_NEAR(normalized[0], 0.1, 1e-6);  // 1/10
    EXPECT_NEAR(normalized[1], 0.2, 1e-6);  // 2/10
    EXPECT_NEAR(normalized[2], 0.3, 1e-6);  // 3/10
    EXPECT_NEAR(normalized[3], 0.4, 1e-6);  // 4/10
}

TEST_F(UtilsTest, EffectiveSampleSize)
{
    // Uniform weights should give ESS = N
    std::vector<double> uniform_weights = {0.25, 0.25, 0.25, 0.25};
    double ess = particle_filter_cpp::utils::compute_effective_sample_size(uniform_weights);
    EXPECT_NEAR(ess, 4.0, 1e-6);
    
    // One dominant weight should give ESS = 1
    std::vector<double> dominant_weights = {0.97, 0.01, 0.01, 0.01};
    ess = particle_filter_cpp::utils::compute_effective_sample_size(dominant_weights);
    EXPECT_LT(ess, 2.0);
}

TEST_F(UtilsTest, SystematicResampling)
{
    std::vector<double> weights = {0.1, 0.2, 0.3, 0.4};
    auto indices = particle_filter_cpp::utils::systematic_resampling(weights, 100);
    
    EXPECT_EQ(indices.size(), 100);
    
    // Count occurrences of each index
    std::vector<int> counts(4, 0);
    for (int idx : indices) {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, 4);
        counts[idx]++;
    }
    
    // Check that sampling is proportional to weights (approximately)
    EXPECT_LT(counts[0], counts[1]); // 0.1 < 0.2
    EXPECT_LT(counts[1], counts[2]); // 0.2 < 0.3
    EXPECT_LT(counts[2], counts[3]); // 0.3 < 0.4
}

TEST_F(UtilsTest, ParticleConversion)
{
    // Create test particles
    std::vector<particle_filter_cpp::Particle> particles;
    particles.emplace_back(1.0, 2.0, M_PI / 4, 0.5);
    particles.emplace_back(-1.0, -2.0, -M_PI / 4, 0.3);
    
    // Convert to pose array and back
    auto pose_array = particle_filter_cpp::utils::particles_to_pose_array(particles);
    auto converted_particles = particle_filter_cpp::utils::pose_array_to_particles(pose_array);
    
    EXPECT_EQ(converted_particles.size(), particles.size());
    
    for (size_t i = 0; i < particles.size(); ++i) {
        EXPECT_NEAR(converted_particles[i].x, particles[i].x, 1e-6);
        EXPECT_NEAR(converted_particles[i].y, particles[i].y, 1e-6);
        EXPECT_NEAR(converted_particles[i].theta, particles[i].theta, 1e-6);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}