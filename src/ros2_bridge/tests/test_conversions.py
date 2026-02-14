"""型変換のユニットテスト。ROS2 不要（Layer 1）。"""

import numpy as np
import pytest
from alpasim_grpc.v0 import common_pb2

from alpasim_ros2_bridge.conversions import (
    alpasim_pose_to_ros_pose,
    alpasim_quat_to_ros,
    autoware_trajectory_to_alpasim,
    ros_quat_to_alpasim,
    timestamp_us_to_sec_nanosec,
    sec_nanosec_to_timestamp_us,
)
from tests.mock_data import make_pose


# ---------------------------------------------------------------------------
# Timestamp 変換
# ---------------------------------------------------------------------------
class TestTimestampConversion:
    """AlpaSim timestamp_us ↔ ROS2 (sec, nanosec) の変換"""

    def test_zero(self):
        sec, nanosec = timestamp_us_to_sec_nanosec(0)
        assert sec == 0
        assert nanosec == 0

    def test_round_trip(self):
        original_us = 1_700_000_123_456
        sec, nanosec = timestamp_us_to_sec_nanosec(original_us)
        restored_us = sec_nanosec_to_timestamp_us(sec, nanosec)
        assert restored_us == original_us

    def test_microsecond_precision(self):
        sec, nanosec = timestamp_us_to_sec_nanosec(1_500_000)
        assert sec == 1
        assert nanosec == 500_000_000

    @pytest.mark.parametrize("us", [0, 1, 999_999, 1_000_000, 123_456_789_012])
    def test_non_negative(self, us):
        sec, nanosec = timestamp_us_to_sec_nanosec(us)
        assert sec >= 0
        assert 0 <= nanosec < 1_000_000_000

    def test_reverse_conversion(self):
        us = sec_nanosec_to_timestamp_us(2, 300_000_000)
        assert us == 2_300_000


# ---------------------------------------------------------------------------
# Quaternion 変換
# ---------------------------------------------------------------------------
class TestQuaternionConversion:
    """AlpaSim Quat(w,x,y,z) ↔ ROS2 Quaternion(x,y,z,w) の並び順変換"""

    def test_identity(self):
        ros_q = alpasim_quat_to_ros(w=1.0, x=0.0, y=0.0, z=0.0)
        assert ros_q == (0.0, 0.0, 0.0, 1.0)  # ROS: x,y,z,w

    def test_round_trip(self):
        w, x, y, z = 0.707, 0.0, 0.707, 0.0
        ros_q = alpasim_quat_to_ros(w, x, y, z)
        restored = ros_quat_to_alpasim(*ros_q)
        assert restored == pytest.approx((w, x, y, z))

    def test_arbitrary_values(self):
        ros_q = alpasim_quat_to_ros(w=0.5, x=0.1, y=0.2, z=0.3)
        # ROS order: (x, y, z, w)
        assert ros_q == pytest.approx((0.1, 0.2, 0.3, 0.5))


# ---------------------------------------------------------------------------
# Pose 変換
# ---------------------------------------------------------------------------
class TestPoseConversion:
    """AlpaSim Pose → ROS2 相当の Pose dict への変換"""

    def test_translation(self):
        pose = make_pose(x=1.0, y=2.0, z=3.0)
        ros_pose = alpasim_pose_to_ros_pose(pose)
        assert ros_pose["position"]["x"] == 1.0
        assert ros_pose["position"]["y"] == 2.0
        assert ros_pose["position"]["z"] == 3.0

    def test_quaternion_reorder(self):
        pose = make_pose(qw=0.707, qx=0.0, qy=0.0, qz=0.707)
        ros_pose = alpasim_pose_to_ros_pose(pose)
        q = ros_pose["orientation"]
        assert q["x"] == pytest.approx(0.0)
        assert q["y"] == pytest.approx(0.0)
        assert q["z"] == pytest.approx(0.707)
        assert q["w"] == pytest.approx(0.707)


# ---------------------------------------------------------------------------
# Trajectory 変換 (autoware → AlpaSim protobuf)
# ---------------------------------------------------------------------------
class TestTrajectoryConversion:
    """autoware TrajectoryPoint 相当のデータ → AlpaSim Trajectory の変換。

    autoware_planning_msgs は ROS2 環境でしか import できないため、
    ここでは dict 形式で TrajectoryPoint を模擬する。
    """

    @staticmethod
    def _make_point(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                    time_from_start_sec=0, time_from_start_nanosec=0,
                    longitudinal_velocity_mps=0.0, heading_rate_rps=0.0):
        """テスト用の TrajectoryPoint 相当の dict"""
        return {
            "pose": {
                "position": {"x": x, "y": y, "z": z},
                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            },
            "time_from_start": {
                "sec": time_from_start_sec,
                "nanosec": time_from_start_nanosec,
            },
            "longitudinal_velocity_mps": longitudinal_velocity_mps,
            "heading_rate_rps": heading_rate_rps,
        }

    def test_empty_trajectory(self):
        traj = autoware_trajectory_to_alpasim(points=[], header_stamp_sec=0, header_stamp_nanosec=0)
        assert len(traj.poses) == 0

    def test_single_point(self):
        point = self._make_point(
            x=1.0, y=2.0,
            time_from_start_sec=0, time_from_start_nanosec=500_000_000,
        )
        traj = autoware_trajectory_to_alpasim(
            points=[point],
            header_stamp_sec=1, header_stamp_nanosec=0,
        )
        # base time 1.0s + offset 0.5s = 1_500_000 us
        assert traj.poses[0].timestamp_us == 1_500_000

    def test_multi_point_ordering(self):
        """waypoint は時系列順であること"""
        points = [
            self._make_point(x=float(i), time_from_start_sec=i)
            for i in range(5)
        ]
        traj = autoware_trajectory_to_alpasim(
            points=points, header_stamp_sec=0, header_stamp_nanosec=0,
        )
        timestamps = [p.timestamp_us for p in traj.poses]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == 5

    def test_pose_values(self):
        """位置が正しく変換されること"""
        point = self._make_point(x=5.0, y=10.0, z=1.0)
        traj = autoware_trajectory_to_alpasim(
            points=[point], header_stamp_sec=0, header_stamp_nanosec=0,
        )
        pose = traj.poses[0].pose
        assert pose.vec.x == pytest.approx(5.0)
        assert pose.vec.y == pytest.approx(10.0)
        assert pose.vec.z == pytest.approx(1.0)

    def test_quaternion_to_alpasim_order(self):
        """ROS2(x,y,z,w) → AlpaSim(w,x,y,z) に変換されること"""
        point = self._make_point(qx=0.0, qy=0.0, qz=0.707, qw=0.707)
        traj = autoware_trajectory_to_alpasim(
            points=[point], header_stamp_sec=0, header_stamp_nanosec=0,
        )
        q = traj.poses[0].pose.quat
        assert q.w == pytest.approx(0.707)
        assert q.z == pytest.approx(0.707)
