"""AlpaSim ROS2 Bridge launch ファイル。

使用例:
  ros2 launch alpasim_ros2_bridge bridge.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time from /clock",
        ),
        Node(
            package="alpasim_ros2_bridge",
            executable="bridge_node",
            name="alpasim_bridge",
            parameters=[{
                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }],
            output="screen",
        ),
    ])
