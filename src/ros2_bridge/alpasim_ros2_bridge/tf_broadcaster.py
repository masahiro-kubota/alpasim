"""/tf を broadcast するモジュール。

ego 車両と周辺車両の pose を TF フレームとして配信する。

フレーム構成:
  - map → base_link (ego)
  - map → actor_{id} (周辺車両)
"""

from __future__ import annotations

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from alpasim_grpc.v0 import common_pb2
from alpasim_ros2_bridge.conversions import (
    alpasim_quat_to_ros,
    timestamp_us_to_sec_nanosec,
)


class TFBroadcaster:
    """AlpaSim の Pose を ROS2 TF として broadcast する。"""

    def __init__(self, node: Node) -> None:
        self._broadcaster = TransformBroadcaster(node)

    def _make_transform(
        self,
        pose: common_pb2.Pose,
        timestamp_us: int,
        parent_frame: str,
        child_frame: str,
    ) -> TransformStamped:
        sec, nanosec = timestamp_us_to_sec_nanosec(timestamp_us)
        t = TransformStamped()
        t.header.stamp.sec = sec
        t.header.stamp.nanosec = nanosec
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = float(pose.vec.x)
        t.transform.translation.y = float(pose.vec.y)
        t.transform.translation.z = float(pose.vec.z)

        rx, ry, rz, rw = alpasim_quat_to_ros(
            pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z
        )
        t.transform.rotation.x = rx
        t.transform.rotation.y = ry
        t.transform.rotation.z = rz
        t.transform.rotation.w = rw

        return t

    def send_ego_transform(self, pose: common_pb2.Pose, timestamp_us: int) -> None:
        """ego 車両の map → base_link を broadcast する。"""
        t = self._make_transform(pose, timestamp_us, "map", "base_link")
        self._broadcaster.sendTransform(t)

    def send_actor_transform(
        self, actor_id: str, pose: common_pb2.Pose, timestamp_us: int
    ) -> None:
        """周辺車両の map → actor_{id} を broadcast する。"""
        t = self._make_transform(pose, timestamp_us, "map", f"actor_{actor_id}")
        self._broadcaster.sendTransform(t)
