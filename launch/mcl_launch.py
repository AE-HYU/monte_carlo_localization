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

  # To change MCL update trigger (default: odom)
  Update modes:  ros2 launch particle_filter_cpp mcl_launch.py update_from:=odom   # Update on odometry
                 ros2 launch particle_filter_cpp mcl_launch.py update_from:=lidar  # Update on lidar
                 ros2 launch particle_filter_cpp mcl_launch.py update_from:=timer  # Update on timer
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Get package share directory  
    from ament_index_python.packages import get_package_share_directory
    pkg_share_dir = get_package_share_directory('particle_filter_cpp')
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

    update_from_arg = DeclareLaunchArgument(
        'update_from',
        default_value='odom',
        description='MCL update trigger: odom, lidar, or timer'
    )
    
    # === CONFIGURATION ===
    # Try to find source config first, fallback to install config
    install_config_file = os.path.join(
        get_package_share_directory('particle_filter_cpp'),
        'config',
        'mcl_config.yaml'
    )
    
    # Look for source config relative to install directory
    install_dir = get_package_share_directory('particle_filter_cpp')
    potential_source_config = os.path.join(install_dir, '..', '..', '..', '..', 'src', 'perception_ws', 'monte_carlo_localization', 'config', 'mcl_config.yaml')
    potential_source_config = os.path.abspath(potential_source_config)
    
    # Use source config if it exists, otherwise use install config
    if os.path.exists(potential_source_config):
        default_config_file = potential_source_config
        print(f"[MCL Launch] Using SOURCE config: {default_config_file}")
    else:
        default_config_file = install_config_file
        print(f"[MCL Launch] Using INSTALL config: {default_config_file}")
    
    config_arg = DeclareLaunchArgument(
        'config_file', 
        default_value=default_config_file,
        description='Path to MCL configuration file'
    )
    map_file_path = PathJoinSubstitution([pkg_share, 'maps', [LaunchConfiguration('map_name'), '.yaml']])
    
    # === DYNAMIC PARAMETERS BASED ON MODE ===
    dynamic_params = {
        # Topic names
        'scan_topic': '/scan',  # All modes use /scan
        'odom_topic': PythonExpression([
            "'/ego_racecar/odom' if '", LaunchConfiguration('mod'), "' == 'sim' else '/odom'"
        ]),
        
        # TF frame names
        'odom_frame': PythonExpression([
            "'ego_racecar/odom' if '", LaunchConfiguration('mod'), "' == 'sim' else 'odom'"
        ]),
        'base_frame': PythonExpression([
            "'ego_racecar/base_link' if '", LaunchConfiguration('mod'), "' == 'sim' else 'base_link'"
        ]),
        'laser_frame': PythonExpression([
            "'ego_racecar/laser' if '", LaunchConfiguration('mod'), "' == 'sim' else 'laser'"
        ]),
        
        # TF publishing control
        'publish_map_odom_tf': PythonExpression([
            "'false' if '", LaunchConfiguration('mod'), "' == 'sim' else 'true'"
        ]),
        'publish_odom_base_tf': PythonExpression([
            "'true' if '", LaunchConfiguration('mod'), "' == 'sim' else 'false'"
        ]),

        # MCL update trigger
        'update_from': LaunchConfiguration('update_from')
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
        period=3.0,  # Allow map server and simulator to initialize
        actions=[
            Node(
                package='particle_filter_cpp',
                executable='particle_filter_node',
                name='particle_filter',
                output='screen',
                parameters=[
                    LaunchConfiguration('config_file'),
                    common_params,
                    dynamic_params
                ],
                remappings=[
                    ('/map_server/map', '/particle_filter_map_server/map')
                ]
            )
        ]
    )
    
    # === TF TRANSFORMS RESPONSIBILITY ===
    # Real mode: F1Tenth stack provides base_link->laser, MCL provides map->odom->base_link
    # Sim mode:  Simulator provides map->base_link->laser, MCL only does localization
    
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
        update_from_arg,
        config_arg,
        
        # Nodes
        map_server_node,
        lifecycle_manager_node,
        particle_filter_node,
        rviz_node,
    ])