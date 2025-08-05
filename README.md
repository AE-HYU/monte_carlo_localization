# Particle Filter C++ Implementation

A high-performance C++ implementation of Monte Carlo Localization (MCL) for robotic navigation, with RangeLibc integration for accelerated ray casting.

## Features

- **Monte Carlo Localization**: Robust probabilistic localization using particle filters
- **RangeLibc Integration**: Ultra-fast ray casting for real-time performance
- **Multiple Ray Casting Methods**: CDDT, GPU acceleration, lookup tables, and more
- **ROS 2 Integration**: Full integration with ROS 2 navigation stack
- **Configurable Parameters**: Extensive parameter tuning via YAML files
- **Comprehensive Testing**: Unit and integration tests included

## Dependencies

### Required Dependencies
```bash
# ROS 2 packages
sudo apt install ros-$ROS_DISTRO-rclcpp ros-$ROS_DISTRO-sensor-msgs ros-$ROS_DISTRO-nav-msgs \
                 ros-$ROS_DISTRO-geometry-msgs ros-$ROS_DISTRO-visualization-msgs \
                 ros-$ROS_DISTRO-tf2-ros ros-$ROS_DISTRO-nav2-map-server

# Other dependencies
sudo apt install libeigen3-dev libpcl-dev
```

### RangeLibc Installation (ROS 2 Compatible)

This implementation uses the ROS 2 compatible version of RangeLibc for optimal performance:

```bash
# Navigate to your workspace src directory
cd ~/your_workspace/src

# Clone the ROS 2 compatible RangeLibc
git clone https://github.com/2025-AILAB-Internship-F1TheBeast/range_libc.git

# Build RangeLibc
cd range_libc
mkdir build && cd build
cmake ..
make -j$(nproc)

# The particle filter will automatically find RangeLibc in the workspace
```

#### GPU Support (Optional)
For NVIDIA GPU acceleration with CUDA:
```bash
# Install CUDA development tools
sudo apt install nvidia-cuda-toolkit

# Build with CUDA support
cd range_libc/build
cmake -DWITH_CUDA=ON ..
make -j$(nproc)
```

## Building the Package

```bash
# Navigate to your ROS 2 workspace
cd ~/your_workspace

# Build RangeLibc first
colcon build --packages-select range_libc

# Build the particle filter package
colcon build --packages-select particle_filter_cpp

# Source the workspace
source install/setup.bash
```

## Usage

### Basic Launch
```bash
# Launch particle filter with default parameters
ros2 launch particle_filter_cpp localize_launch.py

# Launch with custom map
ros2 launch particle_filter_cpp localize_launch.py map_file:=your_map.yaml

# Launch without RViz
ros2 launch particle_filter_cpp localize_launch.py use_rviz:=false
```

### Parameter Configuration

Edit `config/localize.yaml` to customize parameters:

```yaml
particle_filter:
  ros__parameters:
    # Core parameters
    num_particles: 4000
    max_range: 10.0
    
    # RangeLibc method selection
    range_method: "cddt"  # Options: bl, rm, rmgpu, cddt, pcddt, glt
    theta_discretization: 112
    
    # Motion model parameters
    motion_dispersion_x: 0.05
    motion_dispersion_y: 0.025
    motion_dispersion_theta: 0.25
    
    # Sensor model parameters
    z_hit: 0.75
    z_short: 0.01
    z_max: 0.07
    z_rand: 0.12
    sigma_hit: 8.0
```

### RangeLibc Methods

| Method | Description | Performance | Use Case |
|--------|-------------|------------|----------|
| `bl` | Bresenham's Line | Slow | Debugging/testing |
| `rm` | Ray Marching | Medium | CPU fallback |
| `rmgpu` | Ray Marching GPU | Very Fast | NVIDIA GPUs |
| `cddt` | Compressed DDT | Fast | **Recommended default** |
| `pcddt` | Pruned CDDT | Fast | Memory constrained |
| `glt` | Giant Lookup Table | Very Fast | High-end CPUs |

### Performance Tuning

For optimal performance:
1. **With GPU**: Use `rmgpu` method
2. **High-end CPU**: Use `glt` method  
3. **General use**: Use `cddt` method (default)
4. **Memory limited**: Use `pcddt` method

## Testing

```bash
# Run all tests
colcon test --packages-select particle_filter_cpp

# View test results
colcon test-result --verbose
```

## Topics and Services

### Subscribed Topics
- `/scan` (sensor_msgs/LaserScan): Laser scan data
- `/odom` (nav_msgs/Odometry): Odometry data
- `/initialpose` (geometry_msgs/PoseWithCovarianceStamped): Initial pose estimate

### Published Topics
- `/pf/viz/particles` (geometry_msgs/PoseArray): Particle visualization
- `/pf/viz/inferred_pose` (geometry_msgs/PoseStamped): Estimated robot pose
- `/pf/pose/odom` (nav_msgs/Odometry): Localization odometry
- `/tf` (tf2_msgs/TFMessage): Transform from map to base_link

### Services Used
- `/map_server/map` (nav_msgs/GetMap): Retrieve occupancy grid map

## Integration with F1TENTH

This implementation is compatible with F1TENTH simulation environments:

```bash
# Launch with F1TENTH simulator
ros2 launch particle_filter_cpp localize_launch.py \
    map_file:=levine.yaml \
    scan_topic:=/scan \
    odom_topic:=/ego_racecar/odom
```

## Troubleshooting

### RangeLibc Not Found
If you see warnings about RangeLibc not being found:
1. Ensure RangeLibc is built: `colcon build --packages-select range_libc`
2. Source your workspace: `source install/setup.bash`
3. The system will fallback to slower ray casting if RangeLibc is unavailable

### Performance Issues
- Use `cddt` or `rmgpu` methods for better performance
- Reduce `num_particles` if real-time performance is critical
- Ensure map resolution is appropriate (0.05m recommended)

### Memory Issues
- Use `pcddt` method to reduce memory usage
- Reduce `theta_discretization` parameter
- Decrease `num_particles`

## File Structure

```
particle_filter_cpp/
├── src/
│   ├── particle_filter_node.cpp  # Main executable
│   ├── particle_filter.cpp       # Core MCL implementation
│   └── utils.cpp                 # Utility functions
├── include/particle_filter_cpp/
│   ├── particle_filter.hpp       # Main class definition
│   └── utils.hpp                 # Utility declarations
├── config/
│   └── localize.yaml            # Configuration parameters
├── launch/
│   └── localize_launch.py       # ROS 2 launch file
├── rviz/
│   └── particle_filter.rviz     # RViz configuration
└── test/
    ├── test_particle_filter.cpp # Unit tests
    └── test_utils.cpp           # Utility tests
```

## License

MIT License - see LICENSE file for details.

## References

- [ROS 2 Compatible RangeLibc](https://github.com/2025-AILAB-Internship-F1TheBeast/range_libc.git)
- [Original RangeLibc](https://github.com/kctess5/range_libc)
- [MIT Racecar Particle Filter](https://github.com/mit-racecar/particle_filter)
- [Probabilistic Robotics](http://www.probabilisticrobotics.org/) by Thrun, Burgard, and Fox