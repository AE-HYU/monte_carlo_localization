# Monte Carlo Localization (MCL)

High-performance particle filter localization for F1TENTH with unified real/simulation configuration.

## Architecture

**Modular Design** - Core components separated into specialized modules:

- **Motion Model** (`motion_model/`) - Odometry-based particle propagation with noise
- **Sensor Models** (`sensor_model/`) - Pluggable observation models:
  - `beam` - Ray casting with multi-component mixture (hit/short/max/rand)
  - `likelihood_field` - Precomputed distance field for faster updates
- **Core Modules** (`modules/`) - Map management, visualization, initialization, utilities
- **Main Node** - Particle filter loop with adaptive resampling

**Sensor Model Selection** - Configure in `mcl_config.yaml`:
```yaml
sensor_model_type: "beam"              # or "likelihood_field"
z_hit: 0.85  z_short: 0.01  z_max: 0.07  z_rand: 0.07  # Mixture weights
```

## Quick Start

```bash
# Build
colcon build --packages-select mcl_pkg
source install/setup.bash

# Real car
ros2 launch mcl_pkg mcl_launch.py mod:=real

# Simulation
ros2 launch mcl_pkg mcl_launch.py mod:=sim

# Bag playback
ros2 launch mcl_pkg mcl_launch.py mod:=bag

# Change map via launch arg or config file
ros2 launch mcl_pkg mcl_launch.py mod:=sim map_name:=fixmap_toolbox
```

## Launch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mod` | `real` | Launch mode: `real`, `sim`, or `bag` |
| `map_name` | from config | Map file to load (overrides `config/mcl_config.yaml`) |
| `use_rviz` | `true` | Launch RViz visualization |

## Topics

### Real Mode (`mod:=real`)
- **Input**: `/scan`, `/odom` - LiDAR and odometry data
- **Output**:
  - `/pf/pose/odom` - Localized pose (for visualization/comparison)
  - **TF**: `map → odom` (50 Hz) - Primary output for planners/controllers
- **Usage**: Downstream nodes should use TF lookup `map → base_link` (composition of `map→odom` ⊗ `odom→base_link`)
- **Timing**: Real time

### Simulation Mode (`mod:=sim`)
- **Input**: `/scan`, `/ego_racecar/odom` - Simulation sensor data
- **Output**: `/pf/pose/odom`, **TF**: `mcl_map → map` (for comparison with simulator)
- **Timing**: Simulation time

### Bag Playback Mode (`mod:=bag`)
- **Input**: `/scan`, `/odom` - Recorded LiDAR and odometry data
- **Output**: `/pf/pose/odom`, **TF**: `map → odom`
- **Timing**: Simulation time

### Visualization
- `/pf/viz/particles` - Particle cloud
- `/pf/viz/inferred_pose` - Estimated pose marker
- `/map` - Map display

**Note**: Planners/controllers should manually compose TF transforms (lookup `map→odom` and `odom→base_link` separately, then compose via rotation matrix). Use velocity from `/odom` topic. Topics `/pf/pose/odom`, `/pf/viz/inferred_pose` are for visualization only.

## Key Configuration

Edit `config/mcl_config.yaml`:

```yaml
# Sensor Model Selection
sensor_model_type: "beam"                  # "beam" or "likelihood_field"
z_hit: 0.85  z_short: 0.01  z_max: 0.07  z_rand: 0.07  # Mixture weights

# Core MCL
max_particles: 4000                        # Number of particles
timer_frequency: 50.0                      # TF update rate (Hz)
max_range: 8.5                            # Laser max range (m)

# Map Configuration
map_server:
  ros__parameters:
    map: 'fixmap_toolbox'                 # Default map (no .yaml extension)
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

**Particle Filter Pipeline**:
1. **Motion Update** - Propagate particles using odometry with noise model
2. **Sensor Update** - Compute likelihood using selected sensor model (beam/likelihood_field)
3. **Resampling** - Adaptive low-variance resampling when effective particles drop
4. **Pose Estimation** - Weighted mean of particle cloud

**Dual-Rate Architecture**:
- **High-frequency TF publishing** (50 Hz): Smooth transform broadcast
- **Event-driven MCL updates** (~40 Hz): Triggered by LiDAR scans

## Prerequisites

- Map file loaded in `maps/` directory
- LiDAR and odometry data available
- Sufficient computational resources for particle filtering