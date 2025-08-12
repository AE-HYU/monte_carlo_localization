#!/usr/bin/env python3

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
    
    # Read config file to auto-detect map name (runtime resolution)
    # Try source directory first (development), then install directory
    src_config_file = os.path.join(os.getcwd(), 'src', 'particle_filter_cpp', 'config', 'localize.yaml')
    config_file = src_config_file if os.path.exists(src_config_file) else os.path.join(
        get_package_share_directory('particle_filter_cpp'),
        'config',
        'localize.yaml'
    )
    # Extract default map name from config file
    config_dict = yaml.safe_load(open(config_file, 'r'))
    map_name = config_dict['map_server']['ros__parameters']['map']
    
    # Declare launch arguments for runtime configuration
    map_name_arg = DeclareLaunchArgument(
        'map_name',
        default_value=map_name,
        description='Map name (without .yaml extension, e.g., levine, map_1753950572, Spielberg_map)'
    )
    
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan',
        description='Laser scan topic name'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='Odometry topic name (real car: /odom, sim: /ego_racecar/odom)'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz visualization'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time (set to true for Gazebo/simulation)'
    )
    
    # Configure file paths for resources
    # Config file: prefer source directory for development
    src_config_path = os.path.join(os.getcwd(), 'src', 'particle_filter_cpp', 'config', 'localize.yaml')
    config_file_path = src_config_path if os.path.exists(src_config_path) else PathJoinSubstitution([pkg_share, 'config', 'localize.yaml'])
    
    # Map file: resolve path dynamically like f1tenth_gym_ros
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
    # RViz configuration file
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'particle_filter.rviz'])
    
    # Common ROS parameters for all nodes
    common_params = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    
    # Map server node - loads and serves map data
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            common_params,
            {'yaml_filename': map_file}  # Map file auto-resolved from config
        ]
    )
    
    # Lifecycle manager - manages map server lifecycle (configure/activate)
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[
            common_params,
            {
                'autostart': True,          # Automatically start managed nodes
                'node_names': ['map_server'] # Nodes under lifecycle management
            }
        ]
    )
    
    # Particle filter node - main localization algorithm
    # Delayed start to ensure map server is ready
    particle_filter_node = TimerAction(
        period=2.0,  # Wait 2 seconds for map server initialization
        actions=[
            Node(
                package='particle_filter_cpp',
                executable='particle_filter_node',
                name='particle_filter',
                output='screen',
                parameters=[config_file_path, common_params],
                remappings=[
                    ('/scan', LaunchConfiguration('scan_topic')),  # Laser scan input
                    ('/odom', LaunchConfiguration('odom_topic'))   # Odometry input
                ]
            )
        ]
    )
    
    # Static transform publisher - defines laser sensor position relative to robot base
    # Args: x y z roll pitch yaw parent_frame child_frame
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_publisher',
        arguments=['0.288', '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'laser'],
        output='screen',
        parameters=[common_params]
    )
    
    # RViz visualization - launches only if use_rviz is true
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],  # Load custom RViz configuration
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
