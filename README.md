# Monte Carlo Localization (MCL)

High-performance particle filter localization for F1TENTH with unified real/simulation configuration.

## Quick Start

```bash
# Build
colcon build --packages-select particle_filter_cpp
source install/setup.bash

# Real hardware
ros2 launch particle_filter_cpp mcl_launch.py

# Simulation
ros2 launch particle_filter_cpp mcl_launch.py sim_mode:=true
```

## Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sim_mode` | `false` | Use simulation topics if true |
| `map_name` | `sibal1` | Map file to load |
| `use_rviz` | `true` | Launch RViz visualization |

## Topics

### Real Hardware Mode
- **Input**: `/scan`, `/odom` - LiDAR and odometry data
- **Output**: `/pf/pose/odom` - Localized pose for planner/controller
- **Transforms**: `map → base_link`

### Simulation Mode
- **Input**: `/scan`, `/ego_racecar/odom` - Simulation sensor data
- **Output**: `/pf/pose/odom` - Localized pose
- **Transforms**: `map → base_link`

### Visualization
- `/pf/viz/particles` - Particle cloud
- `/pf/viz/inferred_pose` - Estimated pose marker
- `/map` - Map display

## Key Configuration

Edit `config/mcl_config.yaml`:

```yaml
# Core MCL
max_particles: 4000           # Number of particles
update_rate: 100              # Real (Hz) / 200 (sim)
max_pose_range: 10000.0       # Map coordinate limits (m)

# Vehicle parameters (auto-set by sim_mode)
# Real hardware: wheelbase=0.325, lidar_offset=0.288
# Simulation: wheelbase=0.324, lidar_offset=0.25
```

## Initialization

### RViz Method
1. Launch MCL with RViz (`use_rviz:=true`)
2. Use "2D Pose Estimate" tool to set initial pose
3. Odometry tracking starts immediately

### Global Method
- Automatic initialization when MCL converges
- No manual intervention required
- Activates when pose estimate stabilizes

## Available Maps

Place map files in `maps/` directory:
- `sibal1` (default racing circuit)
- `Spielberg_map` (F1 Austria GP)
- `levine` (multi-floor building)
- `map_1753950572` (real sensor data)

## Algorithm

Monte Carlo Localization with dual-rate architecture:
- **High-frequency odometry tracking** (100-200 Hz): Smooth interpolation
- **Low-frequency MCL corrections** (~6 Hz): Drift correction from sensors

## Prerequisites

- Map file loaded in `maps/` directory
- LiDAR and odometry data available
- Sufficient computational resources for particle filtering