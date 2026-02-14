"""統合テスト（Layer 3: rclpy + autoware_msgs 必要）。

1 ステップ分のデータフローを mock データで再現し、
全トピックが正しく publish されることを確認する。
AlpaSim 本体は不要。
"""

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("autoware_planning_msgs")

from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from autoware_planning_msgs.msg import Trajectory as AwTrajectory
from autoware_planning_msgs.msg import TrajectoryPoint
from builtin_interfaces.msg import Duration

from alpasim_ros2_bridge.bridge_node import BridgeNode
from tests.mock_data import make_pose, make_rgb_image, make_traffic_objects


def make_autoware_trajectory(n_points=5, start_sec=1, velocity_mps=10.0):
    """テスト用 autoware_planning_msgs/Trajectory を生成"""
    traj = AwTrajectory()
    traj.header.stamp.sec = start_sec
    traj.header.frame_id = "map"
    for i in range(n_points):
        pt = TrajectoryPoint()
        pt.time_from_start = Duration(sec=0, nanosec=i * 200_000_000)
        pt.pose.position.x = float(i) * 2.0
        pt.pose.orientation.w = 1.0
        pt.longitudinal_velocity_mps = velocity_mps
        traj.points.append(pt)
    return traj


@pytest.mark.integration
class TestSingleStepFlow:
    """1 ステップの全トピック publish を検証"""

    @pytest.fixture
    def subscribers(self, ros2_node):
        """全トピックを subscribe して受信データを収集"""
        received = {
            "clock": [],
            "image": [],
        }
        ros2_node.create_subscription(
            Clock, "/clock",
            lambda m: received["clock"].append(m), 10,
        )
        ros2_node.create_subscription(
            Image, "/camera/front/image_raw",
            lambda m: received["image"].append(m), 10,
        )
        return received

    def test_step_publishes_all_topics(self, ros2_node, subscribers):
        """bridge.step() で clock, image が全て publish されること"""
        bridge = BridgeNode(ros2_node, camera_names=["front"])

        bridge.step(
            timestamp_us=1_000_000,
            ego_pose=make_pose(x=10.0, y=5.0),
            images={"front": make_rgb_image(4, 3)},
            image_width=4,
            image_height=3,
            traffic_objects=make_traffic_objects(n_actors=2),
        )

        for _ in range(10):
            rclpy.spin_once(ros2_node, timeout_sec=0.01)

        assert len(subscribers["clock"]) >= 1
        assert len(subscribers["image"]) >= 1

    def test_timestamps_consistent(self, ros2_node, subscribers):
        """clock, image のタイムスタンプが一致すること"""
        bridge = BridgeNode(ros2_node, camera_names=["front"])
        target_us = 2_000_000

        bridge.step(
            timestamp_us=target_us,
            ego_pose=make_pose(x=0.0),
            images={"front": make_rgb_image(2, 2)},
            image_width=2,
            image_height=2,
            traffic_objects=[],
        )

        for _ in range(10):
            rclpy.spin_once(ros2_node, timeout_sec=0.01)

        clock_sec = subscribers["clock"][0].clock.sec
        image_sec = subscribers["image"][0].header.stamp.sec
        assert clock_sec == 2
        assert image_sec == 2


@pytest.mark.integration
class TestMultiStepFlow:
    """複数ステップのデータフロー検証"""

    def test_three_steps(self, ros2_node):
        """3 ステップ実行して clock が 3 回 publish されること"""
        received_clocks = []
        ros2_node.create_subscription(
            Clock, "/clock",
            lambda m: received_clocks.append(m), 10,
        )

        bridge = BridgeNode(ros2_node, camera_names=["front"])

        for i in range(3):
            t_us = (i + 1) * 100_000
            bridge.step(
                timestamp_us=t_us,
                ego_pose=make_pose(x=float(i)),
                images={"front": make_rgb_image(2, 2)},
                image_width=2,
                image_height=2,
                traffic_objects=[],
            )
            rclpy.spin_once(ros2_node, timeout_sec=0.05)

        assert len(received_clocks) == 3

    def test_driver_bridge_receives_during_steps(self, ros2_node):
        """ステップ実行中に ROS2 プランナーからの trajectory を受信できること"""
        bridge = BridgeNode(ros2_node, camera_names=["front"])

        planner_pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10,
        )

        # Step 1: trajectory なし → drive() は None
        bridge.step(
            timestamp_us=100_000,
            ego_pose=make_pose(),
            images={"front": make_rgb_image(2, 2)},
            image_width=2, image_height=2,
            traffic_objects=[],
        )
        rclpy.spin_once(ros2_node, timeout_sec=0.05)
        assert bridge.driver_bridge.latest_trajectory is None

        # プランナーが autoware Trajectory を publish
        planner_pub.publish(make_autoware_trajectory(n_points=5))
        rclpy.spin_once(ros2_node, timeout_sec=0.05)

        # Step 2: trajectory あり → drive() が返す
        bridge.step(
            timestamp_us=200_000,
            ego_pose=make_pose(x=1.0),
            images={"front": make_rgb_image(2, 2)},
            image_width=2, image_height=2,
            traffic_objects=[],
        )
        rclpy.spin_once(ros2_node, timeout_sec=0.05)
        assert bridge.driver_bridge.latest_trajectory is not None
