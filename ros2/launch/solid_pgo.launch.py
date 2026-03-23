"""
solid_pgo.launch.py  (ROS2)
Runs solid_pgo_node alongside FAST_LIO ROS2.
Topic remapping maps FAST_LIO ROS2 output to solid_pgo input.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('solid_pgo'),
        'config', 'params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=config,
            description='Full path to the ROS2 parameters file'
        ),

        Node(
            package='solid_pgo',
            executable='solid_pgo_node',
            name='solid_pgo_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')],
            remappings=[
                # FAST_LIO ROS2 output → solid_pgo input
                ('/aft_mapped_to_init',             '/Odometry'),
                ('/velodyne_cloud_registered_local', '/cloud_registered_body'),
            ]
        ),
    ])
