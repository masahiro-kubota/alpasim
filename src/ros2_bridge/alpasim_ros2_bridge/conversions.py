"""AlpaSim ↔ ROS2 型変換ユーティリティ。

ROS2 メッセージ型を import せず、dict や protobuf 型のみで変換を行う。
Layer 1 テストで ROS2 なしにテスト可能。
"""

from __future__ import annotations

from alpasim_grpc.v0 import common_pb2


# ---------------------------------------------------------------------------
# Timestamp 変換
# ---------------------------------------------------------------------------

def timestamp_us_to_sec_nanosec(timestamp_us: int) -> tuple[int, int]:
    """AlpaSim timestamp_us → ROS2 (sec, nanosec)"""
    sec = int(timestamp_us // 1_000_000)
    nanosec = int((timestamp_us % 1_000_000) * 1_000)
    return sec, nanosec


def sec_nanosec_to_timestamp_us(sec: int, nanosec: int) -> int:
    """ROS2 (sec, nanosec) → AlpaSim timestamp_us"""
    return sec * 1_000_000 + nanosec // 1_000


# ---------------------------------------------------------------------------
# Quaternion 変換
# ---------------------------------------------------------------------------

def alpasim_quat_to_ros(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """AlpaSim Quat(w,x,y,z) → ROS2 (x,y,z,w)"""
    return (x, y, z, w)


def ros_quat_to_alpasim(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    """ROS2 (x,y,z,w) → AlpaSim Quat(w,x,y,z)"""
    return (w, x, y, z)


# ---------------------------------------------------------------------------
# Pose 変換
# ---------------------------------------------------------------------------

def alpasim_pose_to_ros_pose(pose: common_pb2.Pose) -> dict:
    """AlpaSim Pose → ROS2 相当の dict (position, orientation)。

    ROS2 メッセージ型を使わないため dict で返す。
    ROS2 ノード側で geometry_msgs.msg.Pose に変換する。
    """
    q = pose.quat
    return {
        "position": {
            "x": pose.vec.x,
            "y": pose.vec.y,
            "z": pose.vec.z,
        },
        "orientation": {
            "x": q.x,
            "y": q.y,
            "z": q.z,
            "w": q.w,
        },
    }


# ---------------------------------------------------------------------------
# Trajectory 変換 (autoware → AlpaSim protobuf)
# ---------------------------------------------------------------------------

def autoware_trajectory_to_alpasim(
    points: list[dict],
    header_stamp_sec: int,
    header_stamp_nanosec: int,
) -> common_pb2.Trajectory:
    """autoware TrajectoryPoint 相当の dict リスト → AlpaSim protobuf Trajectory。

    各 point は以下の構造:
        {
            "pose": {"position": {"x", "y", "z"}, "orientation": {"x", "y", "z", "w"}},
            "time_from_start": {"sec": int, "nanosec": int},
            "longitudinal_velocity_mps": float,
            "heading_rate_rps": float,
        }

    header_stamp は Trajectory の基準時刻 (sec, nanosec)。
    各 point の time_from_start を加算して absolute timestamp_us を計算する。
    """
    base_us = sec_nanosec_to_timestamp_us(header_stamp_sec, header_stamp_nanosec)

    grpc_poses = []
    for pt in points:
        tfs = pt["time_from_start"]
        offset_us = sec_nanosec_to_timestamp_us(tfs["sec"], tfs["nanosec"])
        timestamp_us = base_us + offset_us

        ori = pt["pose"]["orientation"]
        pos = pt["pose"]["position"]

        grpc_pose = common_pb2.PoseAtTime(
            timestamp_us=timestamp_us,
            pose=common_pb2.Pose(
                vec=common_pb2.Vec3(x=pos["x"], y=pos["y"], z=pos["z"]),
                quat=common_pb2.Quat(w=ori["w"], x=ori["x"], y=ori["y"], z=ori["z"]),
            ),
        )
        grpc_poses.append(grpc_pose)

    return common_pb2.Trajectory(poses=grpc_poses)
