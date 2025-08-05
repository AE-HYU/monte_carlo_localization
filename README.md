# Particle Filter C++ 

High-performance Monte Carlo Localization (MCL) for robot navigation. Direct C++ port of the working Python implementation with optimizations.

## Quick Start

```bash
# Build
colcon build --packages-select particle_filter_cpp

# Run
ros2 launch particle_filter_cpp localize_launch.py
```

## Configuration

Edit `config/localize.yaml` to customize:
- `num_particles`: 4000 (default)
- `max_range`: 10.0 meters  
- `motion_dispersion_*`: Noise parameters
- `z_hit/short/max/rand`: Sensor model weights

## Launch Options

```bash
# Custom map
ros2 launch particle_filter_cpp localize_launch.py map_file:=your_map.yaml

# Without RViz  
ros2 launch particle_filter_cpp localize_launch.py use_rviz:=false

# Custom topics
ros2 launch particle_filter_cpp localize_launch.py \
    scan_topic:=/scan odom_topic:=/ego_racecar/odom
```

## Key Topics

**Subscribes:**
- `/scan` - Laser scan data
- `/ego_racecar/odom` - Odometry data  
- `/initialpose` - Initial pose from RViz

**Publishes:**
- `/pf/viz/particles` - Particle cloud
- `/pf/viz/inferred_pose` - Estimated pose
- `/tf` - Map to laser transform

## How MCL Works

Monte Carlo Localization uses particles to estimate robot pose:

1. **Prediction**: Move particles based on odometry + noise
2. **Update**: Weight particles by laser scan likelihood  
3. **Resampling**: Keep particles with higher weights
4. **Estimation**: Compute weighted average as robot pose

```
Particles → Motion Model → Sensor Model → Resampling → Pose Estimate
   ↑                                                        ↓
   └─────────────── Repeat every update ──────────────────┘
```

## Code Architecture

```cpp
class ParticleFilter {
    // Core MCL algorithm (direct from Python)
    void MCL(action, observation);
    void motion_model(particles, action);     // Add noise to particle motion
    void sensor_model(particles, scan);       // Weight by scan likelihood
    Eigen::Vector3d expected_pose();          // Weighted average
    
    // ROS interface
    void odomCB(odom_msg);                   // Triggers MCL update
    void lidarCB(scan_msg);                  // Stores scan data
    void publish_tf(pose);                   // Publishes results
    
    // State
    Eigen::MatrixXd particles_;              // [N x 3] particle poses
    std::vector<double> weights_;            // Particle weights
    Eigen::MatrixXd sensor_model_table_;     // Pre-computed lookup table
};
```

**Key Data Flow:**
1. Odometry → `odomCB()` → `MCL()` → Pose estimate
2. Laser → `lidarCB()` → Store for next MCL update
3. MCL → `publish_tf()` → ROS topics

## Implementation Notes

- **Direct Python Port**: Preserves exact algorithm from working Python version
- **Vectorized Operations**: Uses Eigen for fast matrix operations  
- **Pre-computed Sensor Model**: Lookup table for fast likelihood computation
- **Simple Ray Casting**: Basic implementation (RangeLibc optional for speed)
- **Memory Optimized**: Pre-allocated arrays, minimal runtime allocation

## Performance

Expect ~10x faster execution compared to Python version due to:
- Compiled C++ vs interpreted Python
- Optimized Eigen matrix operations  
- Reduced memory allocation overhead
- Vectorized particle operations

Built for F1TENTH racing simulation and real-time robotic navigation.