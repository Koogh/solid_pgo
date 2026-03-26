from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('solid_pgo')

    config_file = PathJoinSubstitution([pkg_share, 'config', 'params.yaml'])

    # Override topics via launch args if needed
    cloud_topic_arg = DeclareLaunchArgument(
        'cloud_topic',
        default_value='rko_lio/frame',
        description='LiDAR point cloud topic (sensor_msgs/PointCloud2)',
    )
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='rko_lio/odometry',
        description='Odometry topic from LIO (nav_msgs/Odometry)',
    )
    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description='Fixed frame for published map and path',
    )

    solid_pgo_node = Node(
        package='solid_pgo',
        executable='solid_pgo_node',
        name='solid_pgo_node',
        output='screen',
        parameters=[
            config_file,
            {
                'cloud_topic': LaunchConfiguration('cloud_topic'),
                'odom_topic':  LaunchConfiguration('odom_topic'),
                'map_frame':   LaunchConfiguration('map_frame'),
            },
        ],
    )

    return LaunchDescription([
        cloud_topic_arg,
        odom_topic_arg,
        map_frame_arg,
        solid_pgo_node,
    ])
