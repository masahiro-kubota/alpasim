"""AlpaSim サービス出力の mock データ生成。

alpasim_grpc の protobuf 型を直接構築する。
テストから import して使う。
"""

import numpy as np
from alpasim_grpc.v0 import common_pb2


def make_pose(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """AlpaSim Pose の mock 生成"""
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=x, y=y, z=z),
        quat=common_pb2.Quat(w=qw, x=qx, y=qy, z=qz),
    )


def make_pose_at_time(timestamp_us, x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """タイムスタンプ付き Pose の mock 生成"""
    return common_pb2.PoseAtTime(
        pose=make_pose(x=x, y=y, z=z, qw=qw, qx=qx, qy=qy, qz=qz),
        timestamp_us=timestamp_us,
    )


def make_dynamic_state(vx=10.0, vy=0.0, vz=0.0, wx=0.0, wy=0.0, wz=0.0):
    """DynamicState の mock 生成"""
    return common_pb2.DynamicState(
        linear_velocity=common_pb2.Vec3(x=vx, y=vy, z=vz),
        angular_velocity=common_pb2.Vec3(x=wx, y=wy, z=wz),
    )


def make_trajectory(n_points=5, dt_us=100_000, start_us=0):
    """等間隔 waypoint の mock trajectory (protobuf)"""
    poses = []
    for i in range(n_points):
        t = start_us + i * dt_us
        poses.append(make_pose_at_time(t, x=float(i) * 1.0))
    return common_pb2.Trajectory(poses=poses)


def make_rgb_image(width=640, height=480):
    """テスト用 RGB 画像バイト列"""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()


def make_traffic_object(object_id, x=0.0, y=0.0, z=0.0, size_x=4.5, size_y=2.0, size_z=1.5,
                        is_static=False):
    """周辺車両の mock データ (dict 形式)"""
    return {
        "object_id": object_id,
        "pose": make_pose(x=x, y=y, z=z),
        "aabb": common_pb2.AABB(size_x=size_x, size_y=size_y, size_z=size_z),
        "is_static": is_static,
    }


def make_traffic_objects(n_actors=3, timestamp_us=0):
    """複数周辺車両の mock データ"""
    return [
        make_traffic_object(
            object_id=f"actor_{i}",
            x=float(i) * 10.0,
            y=5.0,
        )
        for i in range(n_actors)
    ]
