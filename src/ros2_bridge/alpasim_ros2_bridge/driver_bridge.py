"""ROS2 プランナーからの trajectory を AlpaSim に渡すブリッジ。

/planning/trajectory (autoware_planning_msgs/Trajectory) を subscribe し、
最新の trajectory を AlpaSim 形式 (common_pb2.Trajectory) に変換して保持する。
drive() メソッドで最新の trajectory を返す。
"""

from __future__ import annotations

from typing import Optional

from rclpy.node import Node

from alpasim_grpc.v0 import common_pb2
from alpasim_ros2_bridge.conversions import (
    autoware_trajectory_to_alpasim,
    ros_quat_to_alpasim,
)

# autoware_planning_msgs は ROS2 環境でのみ利用可能
from autoware_planning_msgs.msg import Trajectory as AwTrajectory


class DriverBridge:
    """ROS2 プランナーからの最新 trajectory を AlpaSim に渡す。"""

    def __init__(self, node: Node) -> None:
        self.latest_trajectory: Optional[common_pb2.Trajectory] = None
        self._sub = node.create_subscription(
            AwTrajectory, "/planning/trajectory", self._on_trajectory, 10
        )

    def _on_trajectory(self, msg: AwTrajectory) -> None:
        """autoware Trajectory を受信して AlpaSim 形式に変換・保持する。"""
        points = []
        for pt in msg.points:
            points.append({
                "pose": {
                    "position": {
                        "x": pt.pose.position.x,
                        "y": pt.pose.position.y,
                        "z": pt.pose.position.z,
                    },
                    "orientation": {
                        "x": pt.pose.orientation.x,
                        "y": pt.pose.orientation.y,
                        "z": pt.pose.orientation.z,
                        "w": pt.pose.orientation.w,
                    },
                },
                "time_from_start": {
                    "sec": pt.time_from_start.sec,
                    "nanosec": pt.time_from_start.nanosec,
                },
                "longitudinal_velocity_mps": pt.longitudinal_velocity_mps,
                "heading_rate_rps": pt.heading_rate_rps,
            })

        self.latest_trajectory = autoware_trajectory_to_alpasim(
            points=points,
            header_stamp_sec=msg.header.stamp.sec,
            header_stamp_nanosec=msg.header.stamp.nanosec,
        )

    async def drive(
        self,
        time_now_us: int,
        time_query_us: int,
    ) -> Optional[common_pb2.Trajectory]:
        """最新の trajectory を返す。未受信時は None（GT fallback）。"""
        if self.latest_trajectory is None:
            return None
        return self.latest_trajectory
