#!/usr/bin/env python3
"""
Unified MCL Launch File
Supports real hardware, simulation, and bag playback modes

Usage:
  Real car:      ros2 launch particle_filter_cpp mcl_launch.py mod:=real
  Simulation:    ros2 launch particle_filter_cpp mcl_launch.py mod:=sim  
  Bag playback:  ros2 launch particle_filter_cpp mcl_launch.py mod:=bag
  
  # To change map, launch with map_name:='your_map'
  Example:       ros2 launch particle_filter_cpp mcl_launch.py mod:=real map_name:='my_custom_map'
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get package share directory
    pkg_share = FindPackageShare('particle_filter_cpp')
    
    # === LAUNCH ARGUMENTS ===
    mode_arg = DeclareLaunchArgument(
        'mod',
        default_value='real',
        description='Launch mode: real (real car using /odom), sim (simulation, use sim time, /ego_racecar/odom), bag (bag file play, use sim time, /odom)'
    )
    
    map_name_arg = DeclareLaunchArgument(
        'map_name',
        default_value='sibal1',
        description='Map name (without .yaml extension)'
    )
    
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz visualization'
    )
    
    # === CONFIGURATION ===
    config_file = PathJoinSubstitution([pkg_share, 'config', 'mcl_config.yaml'])
    map_file_path = PathJoinSubstitution([pkg_share, 'maps', [LaunchConfiguration('map_name'), '.yaml']])
    
    # === DYNAMIC PARAMETERS BASED ON MODE ===
    dynamic_params = {
        'sim_mode': PythonExpression([
            "'true' if '", LaunchConfiguration('mod'), "' == 'sim' else 'false'"
        ]),
        'scan_topic': '/scan',  # All modes use /scan
        'odom_topic': PythonExpression([
            "'/ego_racecar/odom' if '", LaunchConfiguration('mod'), "' == 'sim' else '/odom'"
        ]),
        'base_frame': PythonExpression([
            "'ego_racecar/base_link' if '", LaunchConfiguration('mod'), "' == 'sim' else 'base_link'"
        ]),
        'laser_frame': PythonExpression([
            "'ego_racecar/laser' if '", LaunchConfiguration('mod'), "' == 'sim' else 'laser'"
        ])
        # Removed hardcoded parameters - use config file values instead
    }
    
    # === COMMON PARAMETERS ===
    common_params = {
        'use_sim_time': PythonExpression([
            "'true' if '", LaunchConfiguration('mod'), "' in ['sim', 'bag'] else 'false'"
        ])
    }
    
    # === MAP SERVER NODE ===
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='particle_filter_map_server',
        output='screen',
        parameters=[
            common_params,
            {'yaml_filename': map_file_path}
        ]
    )
    
    # === LIFECYCLE MANAGER ===
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_particle_filter',
        output='screen',
        parameters=[
            common_params,
            {
                'autostart': True,
                'node_names': ['particle_filter_map_server']
            }
        ]
    )
    
    # === PARTICLE FILTER NODE ===
    particle_filter_node = TimerAction(
        period=2.0,  # Allow map server to initialize
        actions=[
            Node(
                package='particle_filter_cpp',
                executable='particle_filter_node',
                name='particle_filter',
                output='screen',
                parameters=[
                    config_file,
                    common_params,
                    dynamic_params
                ],
                remappings=[
                    ('/map_server/map', '/particle_filter_map_server/map')
                ]
            )
        ]
    )
    
    # === STATIC TRANSFORM PUBLISHERS ===
    # Note: TF values should match lidar_offset_x in config file
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='particle_filter_static_tf_publisher',
        arguments=[
            '0.288',  # Use config file value: lidar_offset_x
            '0.0', '0.0', '0.0', '0.0', '0.0', 
            PythonExpression([
                "'ego_racecar/base_link' if '", LaunchConfiguration('mod'), "' == 'sim' else 'base_link'"
            ]),
            PythonExpression([
                "'ego_racecar/laser' if '", LaunchConfiguration('mod'), "' == 'sim' else 'laser'"
            ])
        ],
        output='screen',
        parameters=[common_params]
    )

    # MCL will publish both map->odom and odom->base_link transforms
    # No static transform needed
    
    # === RVIZ NODE ===
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'particle_filter.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
        parameters=[
            common_params,
            {
                'transform_timeout': 300.0,
                'message_filter_queue_size': 100,
                'tf_buffer_cache_time_s': 300.0,
                'tf_tolerance': 300.0
            }
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        mode_arg,
        map_name_arg,
        use_rviz_arg,
        
        # Nodes
        map_server_node,
        lifecycle_manager_node,
        static_tf_node,
        particle_filter_node,
        rviz_node,
    ])