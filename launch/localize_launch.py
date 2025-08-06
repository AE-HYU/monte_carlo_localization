#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package directory
    pkg_share = FindPackageShare('particle_filter_cpp')
    
    # Launch arguments
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='levine.yaml',
        description='Map file to load (without path)'
    )
    
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan',
        description='Laser scan topic name'
    )
    
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='Odometry topic name'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    # File paths
    config_file = PathJoinSubstitution([pkg_share, 'config', 'localize.yaml'])
    map_file = PathJoinSubstitution([pkg_share, 'maps', LaunchConfiguration('map_file')])
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'particle_filter.rviz'])
    
    # Common parameters
    common_params = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    
    # Map server node
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            common_params,
            {'yaml_filename': map_file}
        ]
    )
    
    # Lifecycle manager for map server
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[
            common_params,
            {
                'autostart': True,
                'node_names': ['map_server']
            }
        ]
    )
    
    # Particle filter node
    particle_filter_node = Node(
        package='particle_filter_cpp',
        executable='particle_filter_node',
        name='particle_filter',
        output='screen',
        parameters=[config_file, common_params],
        remappings=[
            ('/scan', LaunchConfiguration('scan_topic')),
            ('/odom', LaunchConfiguration('odom_topic'))
        ]
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
        parameters=[common_params]
    )
    
    return LaunchDescription([
        # Launch arguments
        map_file_arg,
        scan_topic_arg,
        odom_topic_arg,
        use_rviz_arg,
        use_sim_time_arg,
        
        # Nodes (simple sequential launch)
        map_server_node,
        lifecycle_manager_node,
        particle_filter_node,
        rviz_node,
    ])