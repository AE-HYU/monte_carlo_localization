# Monte Carlo Localization (MCL) for F1TENTH

High-performance Monte Carlo Localization implementation with unified configuration for both real hardware and simulation environments.

## Features
- **Unified Configuration**: Single config/launch file for both real and simulation modes
- **Enhanced Odometry Tracking**: Works with both RViz initialization and global initialization
- **Configurable Validation**: Adjustable pose range limits for different map sizes
- **Optimized Performance**: Parallel ray casting, lookup tables, vectorized operations

## Quick Start

### Real Hardware
```bash
# Build
colcon build --packages-select particle_filter_cpp --symlink-install

# Launch with default settings
ros2 launch particle_filter_cpp mcl_launch.py
```

### Simulation Mode
```bash
# Launch with simulation settings
ros2 launch particle_filter_cpp mcl_launch.py sim_mode:=true
```

## Configuration

### Unified Config File: `config/mcl_config.yaml`

**Core Parameters:**
- `max_particles: 4000` - Number of particles
- `max_pose_range: 10000.0` - Maximum valid pose coordinate range (meters)
- `sim_mode: false` - Simulation mode flag (overridden by launch argument)

**Automatic Mode Selection:**
- **Real Hardware Mode** (`sim_mode:=false`):
  - Topics: `/scan`, `/odom`
  - LiDAR offset: `0.288m`
  - Wheelbase: `0.325m`
  - Timer frequency: `100Hz`

- **Simulation Mode** (`sim_mode:=true`):
  - Topics: `/scan`, `/ego_racecar/odom`
  - LiDAR offset: `0.25m`  
  - Wheelbase: `0.324m`
  - Timer frequency: `200Hz`

## Launch Options

### Basic Usage
```bash
# Real hardware with default map
ros2 launch particle_filter_cpp mcl_launch.py

# Simulation mode
ros2 launch particle_filter_cpp mcl_launch.py sim_mode:=true

# Custom map
ros2 launch particle_filter_cpp mcl_launch.py map_name:=Spielberg_map

# Without RViz
ros2 launch particle_filter_cpp mcl_launch.py use_rviz:=false
```

### Combined Options
```bash
# Simulation with custom map
ros2 launch particle_filter_cpp mcl_launch.py sim_mode:=true map_name:=levine

# Real hardware with specific map, no RViz
ros2 launch particle_filter_cpp mcl_launch.py map_name:=map_1753950572 use_rviz:=false
```

## Odometry Tracking

The system now supports odometry tracking in two scenarios:

### 1. RViz Initialization
- Use "2D Pose Estimate" tool in RViz
- Immediate odometry tracking activation
- Log: `"MCL correction [RViz]: [x, y, theta]"`

### 2. Global Initialization (New!)
- Automatic activation after system startup
- Triggered when MCL converges to stable pose estimate
- Log: `"MCL correction [Global]: [x, y, theta]"`

## Available Maps

Place map files in `maps/` directory:
- `sibal1` (default)
- `Spielberg_map`
- `levine`
- `map_1753950572`
- `redbull_1`
- And more...

## Key Topics

**Subscribed:**
- `/scan` or `/ego_racecar/odom` - Laser scan data
- `/odom` or `/ego_racecar/odom` - Odometry data
- `/initialpose` - Initial pose from RViz
- `/clicked_point` - Global reinitialization

**Published:**
- `/pf/viz/particles` - Particle cloud visualization
- `/pf/viz/inferred_pose` - Estimated pose
- `/pf/pose/odom` - Pose as odometry message
- `/tf` - Transform: map → base_link
- `/map` - Map data (persistent)

## Algorithm Overview

Monte Carlo Localization estimates robot pose using particle filtering:

```
1. PREDICTION:   Move particles based on odometry + noise
2. UPDATE:       Weight particles by laser scan likelihood
3. RESAMPLING:   Retain high-weight particles
4. ESTIMATION:   Compute weighted average as pose estimate
5. CORRECTION:   Apply MCL corrections to odometry tracking
```

### Dual-Rate Architecture
- **High-frequency odometry tracking** (100-200 Hz): Smooth motion interpolation
- **Low-frequency MCL corrections** (~6 Hz): Drift correction from sensor data

## Performance Features

**Optimized Ray Casting:**
- Parallel processing with OpenMP
- Configurable thread count
- Distance transform methods

**Sensor Model:**
- Pre-computed lookup tables
- 4-component beam model (z_hit, z_short, z_max, z_rand)
- Vectorized likelihood computation

**Memory Management:**
- Pre-allocated matrices
- Minimal runtime allocation
- Eigen-based vectorization

## Troubleshooting

**Large Maps:**
Increase `max_pose_range` in config for maps larger than 10km:
```yaml
particle_filter:
  ros__parameters:
    max_pose_range: 50000.0  # 50km range
```

**Poor Localization:**
- Check laser scan data quality
- Verify map-reality correspondence
- Adjust motion noise parameters
- Increase particle count for complex environments

**Performance Issues:**
- Reduce `max_particles` for real-time constraints
- Disable `viz: false` for headless operation
- Adjust `angle_step` for fewer laser rays

## Development

**File Structure:**
```
├── config/
│   └── mcl_config.yaml          # Unified configuration
├── launch/
│   └── mcl_launch.py           # Unified launch file
├── src/
│   └── particle_filter.cpp     # Main implementation
├── include/particle_filter_cpp/
│   └── particle_filter.hpp     # Header file
└── maps/                       # Map files (.yaml + .png/.pgm)
```

**Key Classes:**
```cpp
class ParticleFilter {
    // Core MCL algorithm
    void MCL(action, observation);
    void motion_model(particles, action);
    void sensor_model(particles, scan);
    
    // Odometry tracking
    void initialize_odom_tracking(pose, from_rviz);
    void update_odom_pose(odom_msg);
    
    // Utilities
    bool is_pose_valid(pose);
    Eigen::Vector3d get_current_pose();
};
```

Built for high-performance F1TENTH racing and real-time robotic navigation.