"""TrafficPublisher のテスト（Layer 2: rclpy 必要）。

AlpaSim traffic → autoware_perception_msgs/TrackedObjects の変換を検証。
"""

import pytest

rclpy = pytest.importorskip("rclpy")
aw_perception = pytest.importorskip("autoware_perception_msgs")
aw_vehicle = pytest.importorskip("autoware_vehicle_msgs")

from autoware_perception_msgs.msg import (
    ObjectClassification,
    Shape,
    TrackedObjects,
)
from autoware_vehicle_msgs.msg import VelocityReport

from alpasim_ros2_bridge.traffic_publisher import TrafficPublisher
from tests.mock_data import make_dynamic_state, make_traffic_objects


class TestTrafficPublisher:

    def test_publish_tracked_objects(self, ros2_node):
        """traffic オブジェクトが TrackedObjects として publish されること"""
        received = []
        ros2_node.create_subscription(
            TrackedObjects, "/perception/objects",
            lambda msg: received.append(msg), 10,
        )

        pub = TrafficPublisher(ros2_node)
        pub.publish_objects(
            make_traffic_objects(n_actors=3, timestamp_us=1_000_000),
            timestamp_us=1_000_000,
        )
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(received) == 1
        assert len(received[0].objects) == 3

    def test_object_shape_is_bounding_box(self, ros2_node):
        """Shape が BOUNDING_BOX で AABB サイズが正しいこと"""
        received = []
        ros2_node.create_subscription(
            TrackedObjects, "/perception/objects",
            lambda msg: received.append(msg), 10,
        )

        pub = TrafficPublisher(ros2_node)
        pub.publish_objects(
            make_traffic_objects(n_actors=1, timestamp_us=0),
            timestamp_us=0,
        )
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        obj = received[0].objects[0]
        assert obj.shape.type == Shape.BOUNDING_BOX
        assert obj.shape.dimensions.x == pytest.approx(4.5)
        assert obj.shape.dimensions.y == pytest.approx(2.0)
        assert obj.shape.dimensions.z == pytest.approx(1.5)

    def test_object_classification_default_car(self, ros2_node):
        """デフォルトで ObjectClassification.CAR が設定されること"""
        received = []
        ros2_node.create_subscription(
            TrackedObjects, "/perception/objects",
            lambda msg: received.append(msg), 10,
        )

        pub = TrafficPublisher(ros2_node)
        pub.publish_objects(
            make_traffic_objects(n_actors=1, timestamp_us=0),
            timestamp_us=0,
        )
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        obj = received[0].objects[0]
        assert len(obj.classification) >= 1
        assert obj.classification[0].label == ObjectClassification.CAR

    def test_header_timestamp(self, ros2_node):
        """TrackedObjects の header.stamp がシミュレーション時刻と一致すること"""
        received = []
        ros2_node.create_subscription(
            TrackedObjects, "/perception/objects",
            lambda msg: received.append(msg), 10,
        )

        pub = TrafficPublisher(ros2_node)
        pub.publish_objects(
            make_traffic_objects(n_actors=1, timestamp_us=2_500_000),
            timestamp_us=2_500_000,
        )
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert received[0].header.stamp.sec == 2
        assert received[0].header.stamp.nanosec == 500_000_000


class TestVelocityPublisher:

    def test_publish_velocity_report(self, ros2_node):
        """ego の速度が VelocityReport として publish されること"""
        received = []
        ros2_node.create_subscription(
            VelocityReport, "/vehicle/status/velocity",
            lambda msg: received.append(msg), 10,
        )

        pub = TrafficPublisher(ros2_node)
        ds = make_dynamic_state(vx=15.0, vy=0.5)
        pub.publish_velocity_report(ds, timestamp_us=1_000_000)
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(received) == 1
        assert received[0].longitudinal_velocity == pytest.approx(15.0)
        assert received[0].lateral_velocity == pytest.approx(0.5)
        assert received[0].header.stamp.sec == 1
