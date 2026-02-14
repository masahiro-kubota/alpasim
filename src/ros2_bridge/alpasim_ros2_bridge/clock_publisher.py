"""/clock を publish するモジュール。

シミュレーション時刻を ROS2 の /clock トピックに配信する。
use_sim_time:=true の ROS2 ノードが参照する。
"""

from __future__ import annotations

from rclpy.node import Node
from rosgraph_msgs.msg import Clock

from alpasim_ros2_bridge.conversions import timestamp_us_to_sec_nanosec


class ClockPublisher:
    """AlpaSim のシミュレーション時刻を /clock に publish する。"""

    def __init__(self, node: Node) -> None:
        self._pub = node.create_publisher(Clock, "/clock", 10)

    def publish(self, timestamp_us: int) -> None:
        """timestamp_us を /clock に publish する。"""
        sec, nanosec = timestamp_us_to_sec_nanosec(timestamp_us)
        msg = Clock()
        msg.clock.sec = sec
        msg.clock.nanosec = nanosec
        self._pub.publish(msg)
