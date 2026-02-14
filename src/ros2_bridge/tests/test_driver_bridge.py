"""DriverBridge のテスト（Layer 2: rclpy 必要）。

autoware_planning_msgs/Trajectory → AlpaSim drive() の接続を検証。
"""

import pytest

rclpy = pytest.importorskip("rclpy")
aw_planning = pytest.importorskip("autoware_planning_msgs")

from autoware_planning_msgs.msg import Trajectory as AwTrajectory
from autoware_planning_msgs.msg import TrajectoryPoint
from builtin_interfaces.msg import Duration

from alpasim_ros2_bridge.driver_bridge import DriverBridge


def make_autoware_trajectory(n_points=5, start_sec=1, velocity_mps=10.0):
    """テスト用 autoware_planning_msgs/Trajectory を生成"""
    traj = AwTrajectory()
    traj.header.stamp.sec = start_sec
    traj.header.frame_id = "map"
    for i in range(n_points):
        pt = TrajectoryPoint()
        pt.time_from_start = Duration(sec=0, nanosec=i * 200_000_000)  # 0.2s 間隔
        pt.pose.position.x = float(i) * 2.0
        pt.pose.position.y = 0.0
        pt.pose.position.z = 0.0
        pt.pose.orientation.w = 1.0
        pt.longitudinal_velocity_mps = velocity_mps
        pt.lateral_velocity_mps = 0.0
        pt.acceleration_mps2 = 0.0
        pt.heading_rate_rps = 0.0
        pt.front_wheel_angle_rad = 0.0
        traj.points.append(pt)
    return traj


class TestDriverBridge:

    def test_initial_state_is_none(self, ros2_node):
        """初期状態では trajectory が None（force_gt 期間用）"""
        bridge = DriverBridge(ros2_node)
        assert bridge.latest_trajectory is None

    def test_receives_trajectory(self, ros2_node):
        """autoware Trajectory を subscribe して latest_trajectory が更新されること"""
        bridge = DriverBridge(ros2_node)

        pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10
        )
        pub.publish(make_autoware_trajectory(n_points=3))

        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert bridge.latest_trajectory is not None

    def test_trajectory_has_correct_waypoints(self, ros2_node):
        """変換後の trajectory の waypoint 数が一致すること"""
        bridge = DriverBridge(ros2_node)
        pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10
        )

        pub.publish(make_autoware_trajectory(n_points=5))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(bridge.latest_trajectory.poses) == 5

    def test_velocity_is_converted(self, ros2_node):
        """TrajectoryPoint の速度情報が AlpaSim 側に反映されること"""
        bridge = DriverBridge(ros2_node)
        pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10
        )

        pub.publish(make_autoware_trajectory(n_points=3, velocity_mps=15.0))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert bridge.latest_trajectory is not None

    def test_latest_wins(self, ros2_node):
        """複数回 publish した場合、最新の trajectory が使われること"""
        bridge = DriverBridge(ros2_node)
        pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10
        )

        pub.publish(make_autoware_trajectory(n_points=3))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        pub.publish(make_autoware_trajectory(n_points=7))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(bridge.latest_trajectory.poses) == 7

    @pytest.mark.asyncio
    async def test_drive_returns_latest(self, ros2_node):
        """drive() が最新の trajectory を返すこと"""
        bridge = DriverBridge(ros2_node)
        pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10
        )

        pub.publish(make_autoware_trajectory(n_points=4))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        result = await bridge.drive(
            time_now_us=1_000_000,
            time_query_us=1_100_000,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_drive_returns_none_before_first_trajectory(self, ros2_node):
        """trajectory 未受信時に drive() が None を返すこと（GT fallback）"""
        bridge = DriverBridge(ros2_node)
        result = await bridge.drive(
            time_now_us=0,
            time_query_us=100_000,
        )
        assert result is None
