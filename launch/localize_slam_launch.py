#!/usr/bin/env python3
"""
Particle Filter Launch for Real F1TENTH Car with SLAM Maps
- Auto-detects map from localize_slam.yaml
- Optimized for real-time performance
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    # Get package share directory for resource paths
    pkg_share = FindPackageShare('particle_filter_cpp')
    
    # Read SLAM config file to auto-detect map name
    # Try source directory first (development), then install directory
    src_config_file = os.path.join(os.getcwd(), 'src', 'particle_filter_cpp', 'config', 'localize_slam.yaml')
    config_file = src_config_file if os.path.exists(src_config_file) else os.path.join(
        get_package_share_directory('particle_filter_cpp'),
        'config',
        'localize_slam.yaml'
    )
    # Extract default SLAM-generated map name from config
    config_dict = yaml.safe_load(open(config_file, 'r'))
    map_name = config_dict['map_server']['ros__parameters']['map']
    
    # Declare launch arguments for real F1TENTH car with SLAM maps
    map_name_arg = DeclareLaunchArgument(
        'map_name',
        default_value=map_name,
        description='SLAM-generated map name (default: map_1753950572 from SLAM)'
    )
    
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan',
        description='Real F1TENTH LiDAR scan topic'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='Real F1TENTH wheel odometry topic'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz for real-time localization monitoring'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (false for real hardware)'
    )
    
    # Configure file paths for SLAM-based localization
    # SLAM config file: optimized for real-time performance with SLAM maps
    src_config_path = os.path.join(os.getcwd(), 'src', 'particle_filter_cpp', 'config', 'localize_slam.yaml')
    config_file_path = src_config_path if os.path.exists(src_config_path) else PathJoinSubstitution([pkg_share, 'config', 'localize_slam.yaml'])
    
    # SLAM-generated map file: resolve path dynamically
    # If map_name contains '/', it's already a full path, otherwise resolve from maps directory
    if '/' in map_name:
        map_file = map_name + '.yaml'
    else:
        # Try source directory first (for development)
        src_maps_path = os.path.join('src', 'particle_filter_cpp', 'maps', map_name + '.yaml')
        if os.path.exists(src_maps_path):
            map_file = os.path.abspath(src_maps_path)
        else:
            # Fallback to package share directory
            map_file = os.path.join(get_package_share_directory('particle_filter_cpp'), 'maps', map_name + '.yaml')
    # RViz config for real-time localization visualization
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'particle_filter.rviz'])
    
    # Common ROS parameters for real hardware operation
    common_params = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    
    # Map server node - serves SLAM-generated map for localization
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='particle_filter_map_server',
        output='screen',
        parameters=[
            common_params,
            {'yaml_filename': map_file}  # SLAM-generated map file
        ]
    )
    
    # Lifecycle manager - manages SLAM map server for real-time operation
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_particle_filter',
        output='screen',
        parameters=[
            common_params,
            {
                'autostart': True,          # Auto-start for seamless operation
                'node_names': ['particle_filter_map_server'] # Manage SLAM map server
            }
        ]
    )
    
    # Particle filter node - real-time localization with SLAM maps
    # Uses optimized particle count (1000) for real-time performance
    particle_filter_node = TimerAction(
        period=2.0,  # Allow SLAM map server to initialize
        actions=[
            Node(
                package='particle_filter_cpp',
                executable='particle_filter_node',
                name='particle_filter',
                output='screen',
                parameters=[config_file_path, common_params],  # Uses localize_slam.yaml
                remappings=[
                    ('/scan', LaunchConfiguration('scan_topic')),  # Real F1TENTH LiDAR
                    ('/odom', LaunchConfiguration('odom_topic')),   # Real F1TENTH odometry
                    ('/map_server/map', '/particle_filter_map_server/map')  # Map service remapping
                ]
            )
        ]
    )
    
    # Static transform publisher - real F1TENTH LiDAR sensor position
    # Calibrated offset from base_link to laser frame (288mm forward)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='particle_filter_static_tf_publisher',
        arguments=['0.288', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'laser'],
        output='screen',
        parameters=[common_params]
    )
    
    # RViz visualization - critical for monitoring real-time localization
    # Shows particle convergence and localization accuracy on SLAM map
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],  # Real-time monitoring configuration
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
        parameters=[common_params]
    )
    
    return LaunchDescription([
        # Launch arguments
        map_name_arg,
        scan_topic_arg,
        odom_topic_arg,
        use_rviz_arg,
        use_sim_time_arg,
        
        # Nodes (simple sequential launch)
        map_server_node,
        lifecycle_manager_node,
        static_tf_node,
        particle_filter_node,
        rviz_node,
    ])
