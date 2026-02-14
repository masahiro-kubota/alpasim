"""周辺車両情報と ego 速度を ROS2 トピックに publish するモジュール。

AlpaSim の traffic データを autoware_perception_msgs/TrackedObjects に変換し、
ego 車両の速度を autoware_vehicle_msgs/VelocityReport として配信する。
"""

from __future__ import annotations

import hashlib
from typing import Any

from rclpy.node import Node

from autoware_perception_msgs.msg import (
    ObjectClassification,
    Shape,
    TrackedObject,
    TrackedObjectKinematics,
    TrackedObjects,
)
from autoware_vehicle_msgs.msg import VelocityReport
from geometry_msgs.msg import Vector3
from unique_identifier_msgs.msg import UUID

from alpasim_grpc.v0 import common_pb2
from alpasim_ros2_bridge.conversions import (
    alpasim_quat_to_ros,
    timestamp_us_to_sec_nanosec,
)


def _object_id_to_uuid(object_id: str) -> UUID:
    """文字列の object_id を UUID メッセージに変換する。"""
    digest = hashlib.md5(object_id.encode()).digest()  # noqa: S324
    msg = UUID()
    msg.uuid = list(digest)
    return msg


class TrafficPublisher:
    """周辺車両と ego 速度を ROS2 トピックに publish する。"""

    def __init__(self, node: Node) -> None:
        self._objects_pub = node.create_publisher(
            TrackedObjects, "/perception/objects", 10
        )
        self._velocity_pub = node.create_publisher(
            VelocityReport, "/vehicle/status/velocity", 10
        )

    def publish_objects(
        self,
        traffic_objects: list[dict[str, Any]],
        timestamp_us: int,
    ) -> None:
        """周辺車両を TrackedObjects として publish する。

        traffic_objects は mock_data.make_traffic_objects() と同じ dict 形式:
            {"object_id", "pose" (common_pb2.Pose), "aabb" (common_pb2.AABB), "is_static"}
        """
        sec, nanosec = timestamp_us_to_sec_nanosec(timestamp_us)

        msg = TrackedObjects()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = "map"

        for obj_data in traffic_objects:
            obj = TrackedObject()
            obj.object_id = _object_id_to_uuid(obj_data["object_id"])
            obj.existence_probability = 1.0

            # Classification: デフォルトは CAR
            classification = ObjectClassification()
            classification.label = ObjectClassification.CAR
            classification.probability = 1.0
            obj.classification.append(classification)

            # Kinematics
            pose = obj_data["pose"]
            obj.kinematics.pose_with_covariance.pose.position.x = float(pose.vec.x)
            obj.kinematics.pose_with_covariance.pose.position.y = float(pose.vec.y)
            obj.kinematics.pose_with_covariance.pose.position.z = float(pose.vec.z)

            rx, ry, rz, rw = alpasim_quat_to_ros(
                pose.quat.w, pose.quat.x, pose.quat.y, pose.quat.z
            )
            obj.kinematics.pose_with_covariance.pose.orientation.x = rx
            obj.kinematics.pose_with_covariance.pose.orientation.y = ry
            obj.kinematics.pose_with_covariance.pose.orientation.z = rz
            obj.kinematics.pose_with_covariance.pose.orientation.w = rw

            obj.kinematics.is_stationary = obj_data.get("is_static", False)

            # Shape: BOUNDING_BOX + AABB サイズ
            aabb = obj_data["aabb"]
            obj.shape.type = Shape.BOUNDING_BOX
            obj.shape.dimensions = Vector3(
                x=float(aabb.size_x),
                y=float(aabb.size_y),
                z=float(aabb.size_z),
            )

            msg.objects.append(obj)

        self._objects_pub.publish(msg)

    def publish_velocity_report(
        self,
        dynamic_state: common_pb2.DynamicState,
        timestamp_us: int,
    ) -> None:
        """ego 車両の速度を VelocityReport として publish する。"""
        sec, nanosec = timestamp_us_to_sec_nanosec(timestamp_us)

        msg = VelocityReport()
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = "base_link"
        msg.longitudinal_velocity = float(dynamic_state.linear_velocity.x)
        msg.lateral_velocity = float(dynamic_state.linear_velocity.y)
        msg.heading_rate = float(dynamic_state.angular_velocity.z)

        self._velocity_pub.publish(msg)
