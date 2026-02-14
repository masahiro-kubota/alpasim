"""SensorPublisher のテスト（Layer 2: rclpy 必要）。

mock 画像が正しい ROS2 メッセージになるか検証。
"""

pytest = __import__("pytest")
rclpy = pytest.importorskip("rclpy")

from sensor_msgs.msg import Image, CameraInfo

from alpasim_ros2_bridge.sensor_publisher import SensorPublisher
from tests.mock_data import make_rgb_image


class TestSensorPublisher:

    def test_publish_rgb_image(self, ros2_node):
        """RGB 画像が /camera/front/image_raw に publish されること"""
        received = []
        ros2_node.create_subscription(
            Image, "/camera/front/image_raw",
            lambda msg: received.append(msg), 10,
        )

        pub = SensorPublisher(ros2_node, camera_names=["front"])
        image_bytes = make_rgb_image(width=4, height=3)
        pub.publish_image("front", image_bytes, width=4, height=3,
                          timestamp_us=1_000_000)

        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(received) == 1
        assert received[0].width == 4
        assert received[0].height == 3
        assert received[0].encoding == "rgb8"
        assert received[0].header.stamp.sec == 1

    def test_header_frame_id(self, ros2_node):
        """frame_id がカメラ名と一致すること"""
        received = []
        ros2_node.create_subscription(
            Image, "/camera/front/image_raw",
            lambda msg: received.append(msg), 10,
        )

        pub = SensorPublisher(ros2_node, camera_names=["front"])
        pub.publish_image("front", make_rgb_image(2, 2), 2, 2, 0)
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert received[0].header.frame_id == "camera_front"

    def test_camera_info_published(self, ros2_node):
        """CameraInfo が画像と同時に publish されること"""
        received = []
        ros2_node.create_subscription(
            CameraInfo, "/camera/front/camera_info",
            lambda msg: received.append(msg), 10,
        )

        pub = SensorPublisher(ros2_node, camera_names=["front"])
        pub.publish_image("front", make_rgb_image(4, 3), 4, 3, 1_000_000)
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(received) == 1
        assert received[0].width == 4
        assert received[0].height == 3
