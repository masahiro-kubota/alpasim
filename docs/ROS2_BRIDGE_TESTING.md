# AlpaSim ROS2 Bridge テスト戦略

## 基本方針

- `colcon test --packages-select alpasim_ros2_bridge` で **bridge だけ** テスト可能にする
- AlpaSim 本体（runtime, sensorsim, driver 等）は一切不要
- AlpaSim の protobuf 定義（`alpasim_grpc`）のみ軽量依存として使用
- 開発ステップごとにテストを追加し、インクリメンタルに動作確認する

---

## テストレイヤー

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Integration Tests                             │
│  「1ステップ分の全データフローが正しく流れるか」          │
│  ROS2 ノード起動 + mock データ注入 + topic 検証          │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Node Tests                                    │
│  「各 ROS2 ノード機能が単体で動くか」                    │
│  rclpy 使用、AlpaSim 不要                               │
├─────────────────────────────────────────────────────────┤
│  Layer 1: Unit Tests                                    │
│  「型変換・データ変換が正しいか」                        │
│  Pure Python、ROS2 不要、pytest のみ                    │
└─────────────────────────────────────────────────────────┘
```

---

## テストディレクトリ構成

```
src/ros2_bridge/
├── package.xml
├── pyproject.toml
├── alpasim_ros2_bridge/
│   ├── __init__.py
│   ├── conversions.py
│   ├── clock_publisher.py
│   ├── sensor_publisher.py
│   ├── tf_broadcaster.py
│   ├── traffic_publisher.py
│   ├── driver_bridge.py
│   └── bridge_node.py
└── tests/
    ├── conftest.py                  # 共通フィクスチャ
    ├── mock_data.py                 # AlpaSim データの mock 生成
    │
    │  # Layer 1: Unit Tests (ROS2 不要)
    ├── test_conversions.py
    │
    │  # Layer 2: Node Tests (rclpy 使用)
    ├── test_clock_publisher.py
    ├── test_sensor_publisher.py
    ├── test_tf_broadcaster.py
    ├── test_traffic_publisher.py
    ├── test_driver_bridge.py
    │
    │  # Layer 3: Integration Tests
    └── test_step_flow.py
```

---

## Layer 1: Unit Tests（Pure Python、ROS2 不要）

### test_conversions.py

ROS2 や rclpy を一切 import せずにテスト可能。
`conversions.py` の関数が正しく動くことを確認する。

```python
"""型変換のユニットテスト。ROS2 不要。"""

import numpy as np
import pytest


class TestTimestampConversion:
    """AlpaSim timestamp_us ↔ ROS2 Time の変換"""

    def test_zero(self):
        sec, nanosec = timestamp_us_to_sec_nanosec(0)
        assert sec == 0
        assert nanosec == 0

    def test_round_trip(self):
        original_us = 1_700_000_123_456  # 任意のマイクロ秒値
        sec, nanosec = timestamp_us_to_sec_nanosec(original_us)
        restored_us = sec * 1_000_000 + nanosec // 1_000
        assert restored_us == original_us

    def test_microsecond_precision(self):
        # 1.5 秒 = 1_500_000 us
        sec, nanosec = timestamp_us_to_sec_nanosec(1_500_000)
        assert sec == 1
        assert nanosec == 500_000_000

    @pytest.mark.parametrize("us", [0, 1, 999_999, 1_000_000, 123_456_789_012])
    def test_non_negative(self, us):
        sec, nanosec = timestamp_us_to_sec_nanosec(us)
        assert sec >= 0
        assert 0 <= nanosec < 1_000_000_000


class TestQuaternionConversion:
    """AlpaSim Quat(w,x,y,z) ↔ ROS2 Quaternion(x,y,z,w) の並び順変換"""

    def test_identity(self):
        # AlpaSim: w=1, x=0, y=0, z=0
        ros_q = alpasim_quat_to_ros(w=1.0, x=0.0, y=0.0, z=0.0)
        assert ros_q == (0.0, 0.0, 0.0, 1.0)  # ROS: x,y,z,w

    def test_round_trip(self):
        w, x, y, z = 0.707, 0.0, 0.707, 0.0
        ros_q = alpasim_quat_to_ros(w, x, y, z)
        restored = ros_quat_to_alpasim(*ros_q)
        assert restored == pytest.approx((w, x, y, z))


class TestPoseConversion:
    """AlpaSim Pose → ROS2 geometry_msgs/Pose 相当の変換"""

    def test_translation(self):
        pose = mock_alpasim_pose(x=1.0, y=2.0, z=3.0)
        ros_pose = alpasim_pose_to_ros(pose)
        assert ros_pose.position.x == 1.0
        assert ros_pose.position.y == 2.0
        assert ros_pose.position.z == 3.0


class TestTrajectoryConversion:
    """autoware_planning_msgs/Trajectory → AlpaSim Trajectory の変換"""

    def test_empty_trajectory(self):
        traj = autoware_trajectory_to_alpasim(points=[], header_stamp=(0, 0))
        assert len(traj.timestamps_us) == 0

    def test_single_point(self):
        traj = autoware_trajectory_to_alpasim(
            points=[mock_trajectory_point(
                x=1.0, y=2.0, time_from_start_sec=0, time_from_start_nanosec=500_000_000,
            )],
            header_stamp=(1, 0),  # base time = 1 sec
        )
        # 1.0 sec + 0.5 sec = 1_500_000 us
        assert traj.timestamps_us[0] == 1_500_000

    def test_multi_point_ordering(self):
        """waypoint は時系列順であること"""
        points = [
            mock_trajectory_point(x=float(i), time_from_start_sec=i)
            for i in range(5)
        ]
        traj = autoware_trajectory_to_alpasim(points=points, header_stamp=(0, 0))
        assert np.all(np.diff(traj.timestamps_us) > 0)

    def test_velocity_preserved(self):
        """TrajectoryPoint の速度情報が DynamicState に反映されること"""
        traj = autoware_trajectory_to_alpasim(
            points=[mock_trajectory_point(
                x=0.0, longitudinal_velocity_mps=15.0, heading_rate_rps=0.1,
            )],
            header_stamp=(0, 0),
        )
        assert traj.dynamic_states[0].linear_velocity.x == pytest.approx(15.0)
        assert traj.dynamic_states[0].angular_velocity.z == pytest.approx(0.1)


class TestImageConversion:
    """AlpaSim 画像バイト → sensor_msgs/Image 相当の変換"""

    def test_rgb_dimensions(self):
        # 4x3 RGB 画像
        raw = np.zeros((3, 4, 3), dtype=np.uint8)
        img = alpasim_image_to_ros(raw, encoding="rgb8")
        assert img.height == 3
        assert img.width == 4
        assert img.step == 4 * 3

    def test_png_decode(self):
        """PNG バイト列をデコードして Image に変換できること"""
        png_bytes = create_test_png(width=2, height=2)
        img = alpasim_png_to_ros(png_bytes)
        assert img.height == 2
        assert img.width == 2
```

**実行:**

```bash
# ROS2 環境不要で実行可能
cd src/ros2_bridge
pytest tests/test_conversions.py -v
```

---

## Layer 2: Node Tests（rclpy 使用、AlpaSim 不要）

rclpy を使って ROS2 ノードの機能を個別にテストする。
AlpaSim の実サービスは不要。mock データを注入する。

### conftest.py（共通フィクスチャ）

```python
"""ROS2 テスト用共通フィクスチャ"""

import pytest
import rclpy


@pytest.fixture(scope="session")
def ros2_context():
    """セッション全体で1回だけ rclpy を初期化"""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def ros2_node(ros2_context):
    """テストごとに使い捨て ROS2 ノードを作成"""
    node = rclpy.create_node("test_node")
    yield node
    node.destroy_node()
```

### mock_data.py（AlpaSim データの mock）

```python
"""AlpaSim サービス出力の mock データ生成。
alpasim_grpc の protobuf 型を直接構築する。"""

import numpy as np
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2


def make_pose(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
    """AlpaSim Pose の mock 生成"""
    return common_pb2.Pose(
        vec=common_pb2.Vec3(x=x, y=y, z=z),
        quat=common_pb2.Quat(w=qw, x=qx, y=qy, z=qz),
    )


def make_pose_at_time(timestamp_us, x=0.0, y=0.0, z=0.0):
    return common_pb2.PoseAtTime(
        pose=make_pose(x=x, y=y, z=z),
        timestamp_us=timestamp_us,
    )


def make_dynamic_state(vx=10.0, vy=0.0, vz=0.0):
    return common_pb2.DynamicState(
        linear_velocity=common_pb2.Vec3(x=vx, y=vy, z=vz),
        angular_velocity=common_pb2.Vec3(x=0.0, y=0.0, z=0.0),
    )


def make_trajectory(n_points=5, dt_us=100_000, start_us=0):
    """等間隔 waypoint の mock trajectory"""
    poses = []
    for i in range(n_points):
        t = start_us + i * dt_us
        poses.append(make_pose_at_time(t, x=float(i) * 1.0))
    return common_pb2.Trajectory(poses=poses)


def make_rgb_image(width=640, height=480):
    """テスト用 RGB 画像バイト列"""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()


def make_traffic_objects(n_actors=3, timestamp_us=0):
    """周辺車両の mock データ"""
    objects = []
    for i in range(n_actors):
        objects.append({
            "object_id": f"actor_{i}",
            "pose": make_pose(x=float(i) * 10.0, y=5.0),
            "aabb": common_pb2.AABB(size_x=4.5, size_y=2.0, size_z=1.5),
            "timestamp_us": timestamp_us,
        })
    return objects
```

### test_clock_publisher.py

```python
"""ClockPublisher のテスト。/clock が正しいタイムスタンプで publish されるか。"""

import pytest
from rosgraph_msgs.msg import Clock


class TestClockPublisher:

    def test_publish_updates_clock(self, ros2_node):
        """publish した timestamp_us が /clock に反映されること"""
        received = []

        ros2_node.create_subscription(
            Clock, "/clock", lambda msg: received.append(msg), 10
        )

        clock_pub = ClockPublisher(ros2_node)
        clock_pub.publish(1_500_000)  # 1.5 sec

        # spin して subscription を処理
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(received) == 1
        assert received[0].clock.sec == 1
        assert received[0].clock.nanosec == 500_000_000

    def test_monotonic_clock(self, ros2_node):
        """連続 publish で時刻が単調増加すること"""
        received = []
        ros2_node.create_subscription(
            Clock, "/clock", lambda msg: received.append(msg), 10
        )

        clock_pub = ClockPublisher(ros2_node)
        for t in [100_000, 200_000, 300_000]:
            clock_pub.publish(t)
            rclpy.spin_once(ros2_node, timeout_sec=0.1)

        times = [m.clock.sec * 1_000_000_000 + m.clock.nanosec for m in received]
        assert times == sorted(times)
        assert len(set(times)) == 3  # 重複なし
```

### test_sensor_publisher.py

```python
"""SensorPublisher のテスト。mock 画像が正しい ROS2 メッセージになるか。"""

import pytest
from sensor_msgs.msg import Image, CameraInfo

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
```

### test_tf_broadcaster.py

```python
"""TFBroadcaster のテスト。ego + traffic の pose が /tf に正しく出るか。"""

import pytest
from tf2_ros import Buffer, TransformListener

from tests.mock_data import make_pose


class TestTFBroadcaster:

    def test_ego_transform(self, ros2_node):
        """ego pose が map → base_link として broadcast されること"""
        tf_buffer = Buffer()
        TransformListener(tf_buffer, ros2_node)

        broadcaster = TFBroadcaster(ros2_node)
        pose = make_pose(x=10.0, y=5.0, z=0.0)
        broadcaster.send_ego_transform(pose, timestamp_us=1_000_000)

        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        transform = tf_buffer.lookup_transform(
            "map", "base_link",
            rclpy.time.Time(seconds=1),
        )
        assert transform.transform.translation.x == pytest.approx(10.0)
        assert transform.transform.translation.y == pytest.approx(5.0)

    def test_quaternion_order(self, ros2_node):
        """AlpaSim(w,x,y,z) → ROS2(x,y,z,w) の変換が正しいこと"""
        tf_buffer = Buffer()
        TransformListener(tf_buffer, ros2_node)

        broadcaster = TFBroadcaster(ros2_node)
        # 90度 Z軸回転: w=0.707, x=0, y=0, z=0.707
        pose = make_pose(qw=0.707, qx=0.0, qy=0.0, qz=0.707)
        broadcaster.send_ego_transform(pose, timestamp_us=0)

        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        transform = tf_buffer.lookup_transform("map", "base_link",
                                                rclpy.time.Time())
        q = transform.transform.rotation
        assert q.w == pytest.approx(0.707, abs=0.01)
        assert q.z == pytest.approx(0.707, abs=0.01)

    def test_traffic_actors(self, ros2_node):
        """周辺車両が map → actor_{id} として broadcast されること"""
        tf_buffer = Buffer()
        TransformListener(tf_buffer, ros2_node)

        broadcaster = TFBroadcaster(ros2_node)
        broadcaster.send_actor_transform(
            actor_id="vehicle_42",
            pose=make_pose(x=20.0, y=-3.0),
            timestamp_us=1_000_000,
        )

        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        transform = tf_buffer.lookup_transform(
            "map", "actor_vehicle_42",
            rclpy.time.Time(seconds=1),
        )
        assert transform.transform.translation.x == pytest.approx(20.0)
```

### test_driver_bridge.py

```python
"""DriverBridge のテスト。
autoware_planning_msgs/Trajectory → AlpaSim drive() の接続。"""

import pytest
from autoware_planning_msgs.msg import Trajectory as AwTrajectory
from autoware_planning_msgs.msg import TrajectoryPoint
from builtin_interfaces.msg import Duration


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

        assert len(bridge.latest_trajectory.timestamps_us) == 5

    def test_velocity_is_converted(self, ros2_node):
        """TrajectoryPoint の速度情報が AlpaSim 側に反映されること"""
        bridge = DriverBridge(ros2_node)
        pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10
        )

        pub.publish(make_autoware_trajectory(n_points=3, velocity_mps=15.0))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        # 速度情報が保持されていること
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

        assert len(bridge.latest_trajectory.timestamps_us) == 7

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
```

### test_traffic_publisher.py

```python
"""TrafficPublisher のテスト。
AlpaSim traffic → autoware_perception_msgs/TrackedObjects の変換。"""

import pytest
from autoware_perception_msgs.msg import TrackedObjects, ObjectClassification, Shape
from autoware_vehicle_msgs.msg import VelocityReport

from tests.mock_data import make_pose, make_traffic_objects, make_dynamic_state


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
        # AABB: size_x=4.5, size_y=2.0, size_z=1.5 (mock_data で定義)
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
```

---

## Layer 3: Integration Tests

### test_step_flow.py

AlpaSim の 1 ステップ分のデータフローを mock データで再現し、
全トピックが正しく publish されることを確認する。

```python
"""1ステップ分のデータフロー統合テスト。
AlpaSim 本体は不要。mock データを bridge に注入する。"""

import pytest
import rclpy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from autoware_planning_msgs.msg import Trajectory as AwTrajectory
from autoware_perception_msgs.msg import TrackedObjects

from tests.mock_data import (
    make_pose, make_rgb_image, make_trajectory, make_traffic_objects,
)


class TestSingleStepFlow:
    """1 ステップの全トピック publish を検証"""

    @pytest.fixture
    def subscribers(self, ros2_node):
        """全トピックを subscribe して受信データを収集"""
        received = {
            "clock": [],
            "image": [],
            "tf": [],
        }
        ros2_node.create_subscription(
            Clock, "/clock",
            lambda m: received["clock"].append(m), 10,
        )
        ros2_node.create_subscription(
            Image, "/camera/front/image_raw",
            lambda m: received["image"].append(m), 10,
        )
        # TF は TransformListener 経由でも可
        return received

    def test_step_publishes_all_topics(self, ros2_node, subscribers):
        """bridge.step() で clock, image, tf が全て publish されること"""
        bridge = BridgeNode(ros2_node, camera_names=["front"])

        # 1 ステップ分の mock データを注入
        bridge.step(
            timestamp_us=1_000_000,
            ego_pose=make_pose(x=10.0, y=5.0),
            images={"front": make_rgb_image(4, 3)},
            image_width=4,
            image_height=3,
            traffic_objects=make_traffic_objects(n_actors=2),
        )

        # spin で全 callback を処理
        for _ in range(10):
            rclpy.spin_once(ros2_node, timeout_sec=0.01)

        assert len(subscribers["clock"]) >= 1
        assert len(subscribers["image"]) >= 1

    def test_timestamps_consistent(self, ros2_node, subscribers):
        """clock, image, tf のタイムスタンプが一致すること"""
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

        # 外部プランナーを模擬する publisher（autoware_planning_msgs/Trajectory）
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
```

---

## colcon でのテスト実行

### パッケージ単独テスト

```bash
# ros2_bridge だけテスト（AlpaSim 本体のビルド・テスト不要）
colcon test --packages-select alpasim_ros2_bridge

# 結果表示
colcon test-result --verbose
```

### package.xml のテスト依存設定

```xml
<package format="3">
  <name>alpasim_ros2_bridge</name>
  <!-- ... -->

  <!-- テスト依存（colcon test 時のみ必要） -->
  <test_depend>python3-pytest</test_depend>
  <test_depend>launch_testing</test_depend>
  <test_depend>launch_testing_ament_cmake</test_depend>
  <test_depend>launch_testing_ros</test_depend>
</package>
```

### pyproject.toml のテスト設定

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: 統合テスト（Layer 3）",
]
```

### レイヤー別の実行

```bash
# Layer 1 のみ（ROS2 不要、最速）
pytest tests/test_conversions.py -v

# Layer 2 のみ（rclpy 必要）
pytest tests/test_clock_publisher.py tests/test_sensor_publisher.py \
       tests/test_tf_broadcaster.py tests/test_driver_bridge.py -v

# Layer 3 のみ
pytest tests/test_step_flow.py -v

# 全レイヤー
colcon test --packages-select alpasim_ros2_bridge
```

---

## 各開発ステップとテストの対応

| 実装ステップ | 追加するテスト | 確認ポイント |
|---|---|---|
| Step 1: /clock | `test_conversions.py` (timestamp) + `test_clock_publisher.py` | 時刻変換の精度、publish 動作 |
| Step 2: TF broadcast | `test_conversions.py` (quat, pose) + `test_tf_broadcaster.py` | 座標系・四元数の並び順 |
| Step 3: センサー | `test_conversions.py` (image) + `test_sensor_publisher.py` | 画像フォーマット、frame_id |
| Step 4: Driver bridge | `test_driver_bridge.py` | subscribe 動作、None ハンドリング |
| Step 5: Traffic | `test_traffic_publisher.py` | TrackedObjects 変換、Shape, ObjectClassification |
| Step 6: 統合 | `test_step_flow.py` | 全トピックの整合性 |

各ステップ完了時に対応テストが全て通ることを確認してから次に進む。

---

## テストで AlpaSim 本体が不要な理由

```
alpasim_ros2_bridge が依存するもの:
  ├── alpasim_grpc                  ← protobuf 定義のみ（軽量、ビルド不要）
  ├── rclpy                         ← ROS2 Python ランタイム
  ├── sensor_msgs, geometry_msgs    ← ROS2 common_interfaces
  ├── autoware_planning_msgs        ← Trajectory, TrajectoryPoint
  ├── autoware_perception_msgs      ← TrackedObjects, Shape
  └── autoware_vehicle_msgs         ← VelocityReport

alpasim_ros2_bridge が依存しないもの:
  ├── alpasim_runtime     ← シミュレーションループ（重い）
  ├── alpasim_driver      ← PyTorch モデル（非常に重い）
  ├── sensorsim           ← NRE レンダラー（GPU 必要）
  ├── alpasim_controller  ← MPC ソルバー
  └── alpasim_physics     ← Warp-lang（GPU 必要）
```

bridge は AlpaSim のループから **Python オブジェクトとして** データを受け取る。
テストでは `mock_data.py` でそのデータを直接生成するため、
AlpaSim 本体のビルド・起動は一切不要。
