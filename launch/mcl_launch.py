#!/usr/bin/env python3
"""
Unified MCL Launch File
Supports both real hardware and simulation modes via sim_mode argument

Usage:
  Real hardware: ros2 launch particle_filter_cpp mcl_launch.py
  Simulation:    ros2 launch particle_filter_cpp mcl_launch.py sim_mode:=true
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
    sim_mode_arg = DeclareLaunchArgument(
        'sim_mode',
        default_value='false',
        description='Enable simulation mode (true/false)'
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
    
    # === DYNAMIC PARAMETERS BASED ON SIM_MODE ===
    dynamic_params = {
        'sim_mode': LaunchConfiguration('sim_mode'),
        'scan_topic': PythonExpression([
            "'/scan' if '", LaunchConfiguration('sim_mode'), "' == 'false' else '/scan'"
        ]),
        'odom_topic': PythonExpression([
            "'/odom' if '", LaunchConfiguration('sim_mode'), "' == 'false' else '/ego_racecar/odom'"
        ]),
        'lidar_offset_x': PythonExpression([
            "0.288 if '", LaunchConfiguration('sim_mode'), "' == 'false' else 0.25"
        ]),
        'wheelbase': PythonExpression([
            "0.325 if '", LaunchConfiguration('sim_mode'), "' == 'false' else 0.324"
        ]),
        'timer_frequency': PythonExpression([
            "100.0 if '", LaunchConfiguration('sim_mode'), "' == 'false' else 200.0"
        ])
    }
    
    # === COMMON PARAMETERS ===
    common_params = {'use_sim_time': LaunchConfiguration('sim_mode')}
    
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
    
    # === STATIC TRANSFORM PUBLISHER ===
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='particle_filter_static_tf_publisher',
        arguments=[
            PythonExpression(["'0.288' if '", LaunchConfiguration('sim_mode'), "' == 'false' else '0.25'"]),
            '0.0', '0.0', '0.0', '0.0', '0.0', 'base_link', 'laser'
        ],
        output='screen',
        parameters=[common_params]
    )
    
    # === RVIZ NODE ===
    rviz_config = PathJoinSubstitution([pkg_share, 'rviz', 'particle_filter.rviz'])
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
        sim_mode_arg,
        map_name_arg,
        use_rviz_arg,
        
        # Nodes
        map_server_node,
        lifecycle_manager_node,
        static_tf_node,
        particle_filter_node,
        rviz_node,
    ])