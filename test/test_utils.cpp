#include <gtest/gtest.h>
#include "particle_filter_cpp/utils.hpp"
#include "particle_filter_cpp/particle_filter.hpp"
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <random>
#include <thread>

class MapUtilsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a simple test map
        test_map_.info.width = 10;
        test_map_.info.height = 10;
        test_map_.info.resolution = 0.1;
        test_map_.info.origin.position.x = -0.5;
        test_map_.info.origin.position.y = -0.5;
        test_map_.info.origin.position.z = 0.0;
        test_map_.info.origin.orientation.w = 1.0;
        test_map_.info.origin.orientation.x = 0.0;
        test_map_.info.origin.orientation.y = 0.0;
        test_map_.info.origin.orientation.z = 0.0;
        
        // Create a map with free space in the center and obstacles around
        test_map_.data.resize(100);
        for (int i = 0; i < 100; ++i) {
            int x = i % 10;
            int y = i / 10;
            
            // Free space in center 6x6 area
            if (x >= 2 && x < 8 && y >= 2 && y < 8) {
                test_map_.data[i] = 0; // Free
            } else {
                test_map_.data[i] = 100; // Occupied
            }
        }
    }

    nav_msgs::msg::OccupancyGrid test_map_;
};

TEST_F(MapUtilsTest, CoordinateTransformation)
{
    // Test map to world transformation
    Eigen::Vector3d map_coord(5, 5, 0); // Center of map in map coordinates
    Eigen::Vector3d world_coord = particle_filter_cpp::utils::map_to_world(map_coord, test_map_.info);
    
    // Should be at origin in world coordinates
    EXPECT_NEAR(world_coord.x(), 0.0, 1e-6);
    EXPECT_NEAR(world_coord.y(), 0.0, 1e-6);
    
    // Test round trip
    Eigen::Vector3d back_to_map = particle_filter_cpp::utils::world_to_map(world_coord, test_map_.info);
    EXPECT_NEAR(back_to_map.x(), map_coord.x(), 1e-6);
    EXPECT_NEAR(back_to_map.y(), map_coord.y(), 1e-6);
}

TEST_F(MapUtilsTest, ValidPointCheck)
{
    // Test free space point (should be valid)
    Eigen::Vector2d free_point(5, 5);
    EXPECT_TRUE(particle_filter_cpp::utils::is_valid_point(free_point, test_map_));
    
    // Test occupied space point (should be invalid)
    Eigen::Vector2d occupied_point(0, 0);
    EXPECT_FALSE(particle_filter_cpp::utils::is_valid_point(occupied_point, test_map_));
    
    // Test out of bounds point (should be invalid)
    Eigen::Vector2d out_of_bounds(-1, -1);
    EXPECT_FALSE(particle_filter_cpp::utils::is_valid_point(out_of_bounds, test_map_));
}

TEST_F(MapUtilsTest, FreeSpaceExtraction)
{
    auto free_points = particle_filter_cpp::utils::get_free_space_points(test_map_);
    
    // Should find 6x6 = 36 free points
    EXPECT_EQ(free_points.size(), 36);
    
    // All points should be in world coordinates
    for (const auto& point : free_points) {
        // Convert back to map coordinates to verify
        Eigen::Vector3d world_coord(point.x(), point.y(), 0.0);
        Eigen::Vector3d map_coord = particle_filter_cpp::utils::world_to_map(world_coord, test_map_.info);
        
        int x = static_cast<int>(map_coord.x());
        int y = static_cast<int>(map_coord.y());
        
        EXPECT_GE(x, 2);
        EXPECT_LT(x, 8);
        EXPECT_GE(y, 2);
        EXPECT_LT(y, 8);
    }
}

TEST_F(MapUtilsTest, RayCasting)
{
    // Test ray casting from center of free space
    Eigen::Vector2d start(0.0, 0.0); // World coordinates
    double angle = 0.0; // Pointing in +x direction
    double max_range = 1.0;
    
    double range = particle_filter_cpp::utils::cast_ray(start, angle, max_range, test_map_);
    
    // Should hit obstacle at the edge of free space
    EXPECT_LT(range, max_range);
    EXPECT_GT(range, 0.0);
}

class TimerTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TimerTest, BasicTiming)
{
    particle_filter_cpp::utils::Timer timer(5);
    
    // Initially should have 0 fps
    EXPECT_EQ(timer.fps(), 0.0);
    
    // Simulate some ticks
    timer.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.tick();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.tick();
    
    // Should have non-zero fps
    EXPECT_GT(timer.fps(), 0.0);
    EXPECT_LT(timer.fps(), 1000.0); // Reasonable upper bound
}

class CircularArrayTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CircularArrayTest, BasicOperations)
{
    particle_filter_cpp::utils::CircularArray<double> arr(3);
    
    EXPECT_TRUE(arr.empty());
    EXPECT_EQ(arr.size(), 0);
    
    arr.push(1.0);
    arr.push(2.0);
    arr.push(3.0);
    
    EXPECT_FALSE(arr.empty());
    EXPECT_EQ(arr.size(), 3);
    EXPECT_NEAR(arr.mean(), 2.0, 1e-6);
    EXPECT_NEAR(arr.median(), 2.0, 1e-6);
    
    // Test overflow
    arr.push(4.0); // Should replace oldest (1.0)
    EXPECT_EQ(arr.size(), 3);
    EXPECT_NEAR(arr.mean(), 3.0, 1e-6); // (2+3+4)/3 = 3
}

class RandomUtilsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        gen_.seed(42); // Fixed seed for reproducible tests
    }
    
    std::mt19937 gen_;
};

TEST_F(RandomUtilsTest, GaussianNoise)
{
    auto noise = particle_filter_cpp::utils::generate_gaussian_noise(1000, 0.0, 1.0, gen_);
    
    EXPECT_EQ(noise.size(), 1000);
    
    // Calculate sample mean and std dev
    double mean = 0.0;
    for (double n : noise) {
        mean += n;
    }
    mean /= noise.size();
    
    double variance = 0.0;
    for (double n : noise) {
        variance += (n - mean) * (n - mean);
    }
    variance /= (noise.size() - 1);
    double std_dev = std::sqrt(variance);
    
    // Should be approximately normal(0, 1)
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std_dev, 1.0, 0.1);
}

TEST_F(RandomUtilsTest, UniformNoise)
{
    auto noise = particle_filter_cpp::utils::generate_uniform_noise(1000, -1.0, 1.0, gen_);
    
    EXPECT_EQ(noise.size(), 1000);
    
    // All values should be in range [-1, 1]
    for (double n : noise) {
        EXPECT_GE(n, -1.0);
        EXPECT_LE(n, 1.0);
    }
    
    // Calculate sample mean
    double mean = 0.0;
    for (double n : noise) {
        mean += n;
    }
    mean /= noise.size();
    
    // Should be approximately 0 for uniform [-1, 1]
    EXPECT_NEAR(mean, 0.0, 0.1);
}