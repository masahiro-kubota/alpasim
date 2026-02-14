"""AlpaSim ROS2 Bridge 統合ノード。

各 publisher/subscriber を束ね、AlpaSim の 1 ステップ分のデータを
ROS2 トピックに配信する。
"""

from __future__ import annotations

from typing import Any, Optional

from rclpy.node import Node

from alpasim_grpc.v0 import common_pb2
from alpasim_ros2_bridge.clock_publisher import ClockPublisher
from alpasim_ros2_bridge.driver_bridge import DriverBridge
from alpasim_ros2_bridge.sensor_publisher import SensorPublisher
from alpasim_ros2_bridge.tf_broadcaster import TFBroadcaster
from alpasim_ros2_bridge.traffic_publisher import TrafficPublisher


class BridgeNode:
    """AlpaSim ↔ ROS2 のブリッジ統合クラス。

    AlpaSim の loop.py から呼ばれ、1 ステップ分のデータを
    ROS2 トピックに publish する。
    """

    def __init__(self, node: Node, camera_names: list[str]) -> None:
        self.clock_publisher = ClockPublisher(node)
        self.tf_broadcaster = TFBroadcaster(node)
        self.sensor_publisher = SensorPublisher(node, camera_names)
        self.traffic_publisher = TrafficPublisher(node)
        self.driver_bridge = DriverBridge(node)

    def step(
        self,
        timestamp_us: int,
        ego_pose: common_pb2.Pose,
        images: dict[str, bytes],
        image_width: int,
        image_height: int,
        traffic_objects: list[dict[str, Any]],
        ego_dynamic_state: Optional[common_pb2.DynamicState] = None,
    ) -> None:
        """1 ステップ分のデータを ROS2 に publish する。

        Args:
            timestamp_us: シミュレーション時刻 (マイクロ秒)
            ego_pose: ego 車両の Pose (common_pb2.Pose)
            images: カメラ名 → RGB 画像バイト列の dict
            image_width: 画像幅
            image_height: 画像高さ
            traffic_objects: 周辺車両データのリスト (mock_data 形式)
            ego_dynamic_state: ego の速度・加速度 (任意)
        """
        # 1. /clock
        self.clock_publisher.publish(timestamp_us)

        # 2. /tf (ego)
        self.tf_broadcaster.send_ego_transform(ego_pose, timestamp_us)

        # 3. カメラ画像
        for name, image_bytes in images.items():
            self.sensor_publisher.publish_image(
                name, image_bytes, image_width, image_height, timestamp_us
            )

        # 4. 周辺車両
        if traffic_objects:
            self.traffic_publisher.publish_objects(traffic_objects, timestamp_us)

        # 5. ego 速度
        if ego_dynamic_state is not None:
            self.traffic_publisher.publish_velocity_report(
                ego_dynamic_state, timestamp_us
            )
