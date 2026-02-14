"""センサーデータを ROS2 トピックに publish するモジュール。

AlpaSim sensorsim のレンダリング結果を sensor_msgs に変換して配信する。
"""

from __future__ import annotations

from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from alpasim_ros2_bridge.conversions import timestamp_us_to_sec_nanosec


class SensorPublisher:
    """カメラ画像を ROS2 トピックに publish する。"""

    def __init__(self, node: Node, camera_names: list[str]) -> None:
        self._image_pubs: dict[str, any] = {}
        self._info_pubs: dict[str, any] = {}

        for name in camera_names:
            self._image_pubs[name] = node.create_publisher(
                Image, f"/camera/{name}/image_raw", 10
            )
            self._info_pubs[name] = node.create_publisher(
                CameraInfo, f"/camera/{name}/camera_info", 10
            )

    def publish_image(
        self,
        name: str,
        image_bytes: bytes,
        width: int,
        height: int,
        timestamp_us: int,
    ) -> None:
        """RGB 画像を /camera/{name}/image_raw に publish する。"""
        sec, nanosec = timestamp_us_to_sec_nanosec(timestamp_us)

        # Image メッセージ
        img_msg = Image()
        img_msg.header.stamp.sec = sec
        img_msg.header.stamp.nanosec = nanosec
        img_msg.header.frame_id = f"camera_{name}"
        img_msg.height = height
        img_msg.width = width
        img_msg.encoding = "rgb8"
        img_msg.is_bigendian = False
        img_msg.step = width * 3
        img_msg.data = list(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else image_bytes
        self._image_pubs[name].publish(img_msg)

        # CameraInfo メッセージ
        info_msg = CameraInfo()
        info_msg.header.stamp.sec = sec
        info_msg.header.stamp.nanosec = nanosec
        info_msg.header.frame_id = f"camera_{name}"
        info_msg.width = width
        info_msg.height = height
        self._info_pubs[name].publish(info_msg)
