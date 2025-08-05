#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    # Package directories
    pkg_share = FindPackageShare('particle_filter_cpp')
    
    # Read config file to get map file
    config_path = os.path.join(get_package_share_directory('particle_filter_cpp'), 'config', 'localize.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get map file from config, fallback to default
    map_file_from_config = config.get('particle_filter', {}).get('ros__parameters', {}).get('map_file', 'levine.yaml')
    
    # Launch arguments
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value=map_file_from_config,
        description='Map file to load (without path)'
    )
    
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan',
        description='Laser scan topic name'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/ego_racecar/odom',
        description='Odometry topic name'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    # Paths
    config_file_path = PathJoinSubstitution([
        pkg_share, 'config', 'localize.yaml'
    ])
    
    map_file_path = PathJoinSubstitution([
        pkg_share, 'maps', LaunchConfiguration('map_file')
    ])
    
    rviz_config_path = PathJoinSubstitution([
        pkg_share, 'rviz', 'particle_filter.rviz'
    ])
    
    # Nodes
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'yaml_filename': map_file_path,
            'topic_name': 'map',
            'frame_id': 'map'
        }]
    )
    
    map_server_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_mapper',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['map_server']
        }]
    )
    
    # Use the particle filter with config file
    particle_filter_node = Node(
        package='particle_filter_cpp',
        executable='particle_filter_node',
        name='particle_filter',
        output='screen',
        parameters=[config_file_path],
        remappings=[
            ('/scan', LaunchConfiguration('scan_topic')),
            ('/odom', LaunchConfiguration('odom_topic'))
        ]
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen'
    )
    
    return LaunchDescription([
        # Arguments
        map_file_arg,
        scan_topic_arg,
        odom_topic_arg,
        use_rviz_arg,
        
        # Nodes
        map_server_node,
        map_server_lifecycle,
        particle_filter_node,
        rviz_node
    ])
