# AlpaSim ROS2 Bridge テスト戦略

## 基本方針

- Bridge は **別リポジトリ・別コンテナ** の gRPC サービスとして動作する
- AlpaSim 本体（runtime, sensorsim, driver 等）は一切不要
- AlpaSim の protobuf 定義（`alpasim_grpc`）のみ軽量依存として使用
- テストは Docker コンテナ内で `pytest` を直接実行する（colcon test は使わない）

---

## テストレイヤー

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: gRPC Integration Tests                        │
│  「gRPC step() 呼び出し → ROS2 publish → trajectory    │
│   受信 → gRPC レスポンス の全フローが正しく流れるか」   │
│  gRPC client + ROS2 ノード起動 + mock プランナー        │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Node Tests                                    │
│  「各 ROS2 publisher/subscriber が単体で動くか」        │
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
alpasim-ros2-bridge/
├── alpasim_ros2_bridge/
│   ├── __init__.py
│   ├── conversions.py
│   ├── clock_publisher.py
│   ├── sensor_publisher.py
│   ├── tf_broadcaster.py
│   ├── traffic_publisher.py
│   ├── trajectory_listener.py
│   └── server.py
└── tests/
    ├── conftest.py               # 共通フィクスチャ
    ├── mock_data.py              # AlpaSim データの mock 生成
    │
    │  # Layer 1: Unit Tests (ROS2 不要)
    ├── test_conversions.py
    │
    │  # Layer 2: Node Tests (rclpy 使用)
    ├── test_clock_publisher.py
    ├── test_sensor_publisher.py
    ├── test_tf_broadcaster.py
    ├── test_traffic_publisher.py
    ├── test_trajectory_listener.py
    │
    │  # Layer 3: gRPC Integration Tests
    └── test_server.py
```

---

## テスト実行方法

### Docker コンテナ内（全レイヤー）

```bash
# ビルド
docker build -t alpasim-bridge:latest .

# 全テスト実行（デフォルト CMD）
docker run --rm alpasim-bridge:latest

# レイヤー別
docker run --rm alpasim-bridge:latest python3 -m pytest tests/test_conversions.py -v
docker run --rm alpasim-bridge:latest python3 -m pytest tests/test_clock_publisher.py -v
docker run --rm alpasim-bridge:latest python3 -m pytest tests/test_server.py -v
```

### ホストマシン（Layer 1 のみ）

Layer 1 は ROS2 不要なのでホストでも実行可能。

```bash
cd alpasim-ros2-bridge
python3 -m pytest tests/test_conversions.py -v
```

Layer 2/3 はコンテナ内でのみ実行（rclpy + autoware_msgs が必要）。

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
        original_us = 1_700_000_123_456
        sec, nanosec = timestamp_us_to_sec_nanosec(original_us)
        restored_us = sec * 1_000_000 + nanosec // 1_000
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


class TestQuaternionConversion:
    """AlpaSim Quat(w,x,y,z) ↔ ROS2 Quaternion(x,y,z,w) の並び順変換"""

    def test_identity(self):
        ros_q = alpasim_quat_to_ros(w=1.0, x=0.0, y=0.0, z=0.0)
        assert ros_q == (0.0, 0.0, 0.0, 1.0)

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
    """autoware_planning_msgs/Trajectory → AlpaSim common_pb2.Trajectory の変換"""

    def test_empty_trajectory(self):
        traj = autoware_trajectory_to_alpasim(points=[], header_stamp=(0, 0))
        assert len(traj.poses) == 0

    def test_single_point(self):
        traj = autoware_trajectory_to_alpasim(
            points=[mock_trajectory_point(
                x=1.0, y=2.0, time_from_start_sec=0, time_from_start_nanosec=500_000_000,
            )],
            header_stamp=(1, 0),
        )
        assert traj.poses[0].timestamp_us == 1_500_000

    def test_multi_point_ordering(self):
        """waypoint は時系列順であること"""
        points = [
            mock_trajectory_point(x=float(i), time_from_start_sec=i)
            for i in range(5)
        ]
        traj = autoware_trajectory_to_alpasim(points=points, header_stamp=(0, 0))
        timestamps = [p.timestamp_us for p in traj.poses]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == 5
```

---

## Layer 2: Node Tests（rclpy 使用、AlpaSim 不要）

rclpy を使って ROS2 ノードの機能を個別にテストする。
AlpaSim の実サービスは不要。mock データを注入する。

### conftest.py（共通フィクスチャ）

```python
"""ROS2 テスト用共通フィクスチャ"""

import pytest

try:
    import rclpy
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False


@pytest.fixture(scope="session")
def ros2_context():
    """セッション全体で1回だけ rclpy を初期化"""
    if not HAS_RCLPY:
        pytest.skip("rclpy not available")
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def ros2_node(ros2_context):
    """テストごとに使い捨て ROS2 ノードを作成（ユニーク名）"""
    import uuid
    node = rclpy.create_node(f"test_node_{uuid.uuid4().hex[:8]}")
    yield node
    node.destroy_node()
```

### mock_data.py（AlpaSim データの mock）

```python
"""AlpaSim サービス出力の mock データ生成。
alpasim_grpc の protobuf 型を直接構築する。"""

from alpasim_grpc.v0 import common_pb2


def make_pose(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
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
    poses = []
    for i in range(n_points):
        t = start_us + i * dt_us
        poses.append(make_pose_at_time(t, x=float(i) * 1.0))
    return common_pb2.Trajectory(poses=poses)


def make_rgb_image(width=640, height=480):
    import numpy as np
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8).tobytes()


def make_traffic_object(object_id="actor_0", x=10.0, y=5.0):
    return {
        "object_id": object_id,
        "pose": make_pose(x=x, y=y),
        "aabb": common_pb2.AABB(size_x=4.5, size_y=2.0, size_z=1.5),
    }


def make_traffic_objects(n_actors=3):
    return [
        make_traffic_object(object_id=f"actor_{i}", x=float(i) * 10.0)
        for i in range(n_actors)
    ]
```

### test_clock_publisher.py

```python
"""ClockPublisher のテスト。/clock が正しいタイムスタンプで publish されるか。"""

class TestClockPublisher:

    def test_publish_updates_clock(self, ros2_node):
        received = []
        ros2_node.create_subscription(
            Clock, "/clock", lambda msg: received.append(msg), 10
        )
        clock_pub = ClockPublisher(ros2_node)
        clock_pub.publish(1_500_000)
        rclpy.spin_once(ros2_node, timeout_sec=0.1)
        assert len(received) == 1
        assert received[0].clock.sec == 1
        assert received[0].clock.nanosec == 500_000_000

    def test_monotonic_clock(self, ros2_node):
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
        assert len(set(times)) == 3
```

### test_trajectory_listener.py

```python
"""TrajectoryListener のテスト。
/planning/trajectory を subscribe し、asyncio.Event で待機する。"""

import asyncio
import pytest
from autoware_planning_msgs.msg import Trajectory as AwTrajectory


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


class TestTrajectoryListener:

    def test_initial_state_has_no_trajectory(self, ros2_node):
        listener = TrajectoryListener(ros2_node)
        assert listener._trajectory is None

    def test_receives_trajectory(self, ros2_node):
        listener = TrajectoryListener(ros2_node)
        pub = ros2_node.create_publisher(AwTrajectory, "/planning/trajectory", 10)
        pub.publish(make_autoware_trajectory(n_points=3))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)
        assert listener._trajectory is not None

    def test_trajectory_has_correct_waypoints(self, ros2_node):
        listener = TrajectoryListener(ros2_node)
        pub = ros2_node.create_publisher(AwTrajectory, "/planning/trajectory", 10)
        pub.publish(make_autoware_trajectory(n_points=5))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)
        assert len(listener._trajectory.poses) == 5

    def test_latest_wins(self, ros2_node):
        listener = TrajectoryListener(ros2_node)
        pub = ros2_node.create_publisher(AwTrajectory, "/planning/trajectory", 10)

        pub.publish(make_autoware_trajectory(n_points=3))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        pub.publish(make_autoware_trajectory(n_points=7))
        rclpy.spin_once(ros2_node, timeout_sec=0.1)

        assert len(listener._trajectory.poses) == 7

    @pytest.mark.asyncio
    async def test_wait_for_trajectory_resolves(self, ros2_node):
        listener = TrajectoryListener(ros2_node)
        pub = ros2_node.create_publisher(AwTrajectory, "/planning/trajectory", 10)

        async def publish_later():
            await asyncio.sleep(0.05)
            pub.publish(make_autoware_trajectory(n_points=4))
            rclpy.spin_once(ros2_node, timeout_sec=0.1)

        task = asyncio.create_task(publish_later())
        result = await listener.wait_for_trajectory(timeout_sec=1.0)
        await task

        assert result is not None
        assert len(result.poses) == 4

    @pytest.mark.asyncio
    async def test_wait_timeout_raises(self, ros2_node):
        listener = TrajectoryListener(ros2_node)
        with pytest.raises(asyncio.TimeoutError):
            await listener.wait_for_trajectory(timeout_sec=0.1)
```

### test_traffic_publisher.py / test_sensor_publisher.py / test_tf_broadcaster.py

（Layer 2 テストは既存と同じ構造。省略）

---

## Layer 3: gRPC Integration Tests

### test_server.py

gRPC クライアント → Bridge サーバー → ROS2 publish → プランナー応答 → gRPC レスポンスの
フルフローをテストする。

```python
"""gRPC + ROS2 統合テスト。
Bridge サーバーの step() RPC が全トピックを publish し、
プランナーの trajectory を gRPC レスポンスとして返すことを検証する。"""

import asyncio
import grpc
import pytest
from alpasim_grpc.v0 import bridge_pb2, bridge_pb2_grpc
from autoware_planning_msgs.msg import Trajectory as AwTrajectory


@pytest.fixture
async def bridge_server(ros2_node):
    """テスト用 gRPC サーバーを起動"""
    server = grpc.aio.server()
    servicer = BridgeServicer(ros2_node)
    bridge_pb2_grpc.add_ROS2BridgeServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")  # ランダムポート
    await server.start()
    yield port
    await server.stop(grace=0)


@pytest.fixture
async def bridge_stub(bridge_server):
    """テスト用 gRPC クライアント"""
    channel = grpc.aio.insecure_channel(f"localhost:{bridge_server}")
    stub = bridge_pb2_grpc.ROS2BridgeServiceStub(channel)
    yield stub
    await channel.close()


class TestBridgeStep:

    @pytest.mark.asyncio
    async def test_step_publishes_clock(self, bridge_stub, ros2_node):
        received = []
        ros2_node.create_subscription(
            Clock, "/clock", lambda m: received.append(m), 10
        )

        await bridge_stub.step(bridge_pb2.BridgeStepRequest(
            session_uuid="test",
            timestamp_us=1_000_000,
            ego_pose=make_pose(x=10.0),
            force_gt=True,
        ))

        rclpy.spin_once(ros2_node, timeout_sec=0.1)
        assert len(received) >= 1
        assert received[0].clock.sec == 1

    @pytest.mark.asyncio
    async def test_step_returns_trajectory(self, bridge_stub, ros2_node):
        planner_pub = ros2_node.create_publisher(
            AwTrajectory, "/planning/trajectory", 10
        )

        async def mock_planner():
            await asyncio.sleep(0.05)
            planner_pub.publish(make_autoware_trajectory(n_points=5))
            rclpy.spin_once(ros2_node, timeout_sec=0.1)

        task = asyncio.create_task(mock_planner())

        response = await bridge_stub.step(bridge_pb2.BridgeStepRequest(
            session_uuid="test",
            timestamp_us=1_000_000,
            ego_pose=make_pose(x=10.0),
            force_gt=False,
        ))
        await task

        assert response.has_trajectory is True
        assert len(response.trajectory.poses) == 5

    @pytest.mark.asyncio
    async def test_force_gt_skips_trajectory(self, bridge_stub):
        response = await bridge_stub.step(bridge_pb2.BridgeStepRequest(
            session_uuid="test",
            timestamp_us=1_000_000,
            ego_pose=make_pose(x=0.0),
            force_gt=True,
        ))
        assert response.has_trajectory is False


class TestBridgeSession:

    @pytest.mark.asyncio
    async def test_start_and_close_session(self, bridge_stub):
        await bridge_stub.start_session(bridge_pb2.BridgeSessionRequest(
            session_uuid="test-session",
            camera_names=["front"],
        ))
        await bridge_stub.close_session(bridge_pb2.BridgeSessionCloseRequest(
            session_uuid="test-session",
        ))

    @pytest.mark.asyncio
    async def test_multi_step_flow(self, bridge_stub, ros2_node):
        received_clocks = []
        ros2_node.create_subscription(
            Clock, "/clock", lambda m: received_clocks.append(m), 10
        )

        for i in range(3):
            await bridge_stub.step(bridge_pb2.BridgeStepRequest(
                session_uuid="test",
                timestamp_us=(i + 1) * 100_000,
                ego_pose=make_pose(x=float(i)),
                force_gt=True,
            ))
            rclpy.spin_once(ros2_node, timeout_sec=0.05)

        assert len(received_clocks) == 3
```

---

## 各開発ステップとテストの対応

| 実装ステップ | 追加するテスト | 確認ポイント |
|---|---|---|
| Step 1: conversions | `test_conversions.py` | 時刻・座標・trajectory 変換 |
| Step 2: /clock | `test_clock_publisher.py` | タイムスタンプ精度、publish 動作 |
| Step 3: TF broadcast | `test_tf_broadcaster.py` | 座標系・四元数の並び順 |
| Step 4: センサー | `test_sensor_publisher.py` | 画像フォーマット、frame_id |
| Step 5: trajectory_listener | `test_trajectory_listener.py` | subscribe + asyncio 待機、タイムアウト |
| Step 6: Traffic | `test_traffic_publisher.py` | TrackedObjects 変換、Shape, Classification |
| Step 7: server.py | `test_server.py` | gRPC → ROS2 → プランナー → gRPC フルフロー |

---

## テストで AlpaSim 本体が不要な理由

```
alpasim_ros2_bridge が依存するもの:
  ├── alpasim_grpc                  ← protobuf 定義のみ（軽量）
  ├── grpcio                        ← gRPC サーバー/クライアント
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

Bridge は AlpaSim のループから **gRPC 経由** でデータを受け取る。
テストでは gRPC クライアントから mock データを送信するため、
AlpaSim 本体のビルド・起動は一切不要。

Bridge コンテナ (Ubuntu 24.04 + ROS2 Jazzy + Python 3.12) と
AlpaSim コンテナ (Python 3.12) は完全に分離される。

---

## AlpaSim ↔ Bridge インテグレーションテスト

上記 Layer 1〜3 は Bridge コンテナ **単体** のテスト。
ここでは AlpaSim の `loop.py` から Bridge を gRPC 経由で呼び出す
**クロスコンテナ統合テスト** を定義する。

### 前提: AlpaSim 側の実装

Bridge を呼ぶために alpasim 側に必要な変更:

#### 1. BridgeService クラス（新規）

`src/runtime/alpasim_runtime/services/bridge_service.py`

既存サービス（driver, sensorsim 等）と同じ `ServiceBase` パターンで実装する。

```python
class BridgeService(ServiceBase):
    """ROS2 Bridge への gRPC クライアント。"""

    def __init__(self, address: str, skip: bool = False, id: int = 0):
        super().__init__(address, skip=skip, id=id)

    async def step(
        self,
        session_uuid: str,
        timestamp_us: int,
        ego_pose: Pose,
        ego_dynamic_state: DynamicState,
        camera_images: list[CameraImage],
        traffic_objects: list[TrafficObject],
        time_now_us: int,
        time_query_us: int,
        force_gt: bool,
    ) -> BridgeStepResponse:
        if self.skip:
            return BridgeStepResponse(has_trajectory=False)

        request = BridgeStepRequest(
            session_uuid=session_uuid,
            timestamp_us=timestamp_us,
            ego_pose=ego_pose,
            ego_dynamic_state=ego_dynamic_state,
            camera_images=camera_images,
            traffic_objects=traffic_objects,
            time_now_us=time_now_us,
            time_query_us=time_query_us,
            force_gt=force_gt,
        )
        return await self.stub.step(request)
```

#### 2. Dispatcher への追加

`src/runtime/alpasim_runtime/dispatcher.py`

```python
# acquire_all_services() に bridge を追加
pools = [
    ("driver", self.driver_pool),
    ("sensorsim", self.sensorsim_pool),
    ("physics", self.physics_pool),
    ("trafficsim", self.trafficsim_pool),
    ("controller", self.controller_pool),
    ("bridge", self.bridge_pool),        # ← 追加
]

# run_job() の bind() に bridge を追加
await rollout.bind(
    services["driver"],
    services["sensorsim"],
    services["physics"],
    services["trafficsim"],
    services["controller"],
    services["bridge"],                  # ← 追加
    self.camera_catalog,
).run()
```

#### 3. loop.py への追加

`src/runtime/alpasim_runtime/loop.py`

`driver.drive()` の直後に挿入。プランナーの trajectory があればそれを使い、
なければドライバーの GT trajectory にフォールバックする。

```python
# request driving (既存)
drive_trajectory_noisy = await self.driver.drive(
    time_now_us=now_us,
    time_query_us=future_us,
    renderer_data=self.data_sensorsim_to_driver,
)

# ── bridge (新規) ──
if self.bridge is not None:
    bridge_response = await self.bridge.step(
        session_uuid=str(self.unbound.rollout_uuid),
        timestamp_us=now_us,
        ego_pose=self.ego_trajectory.poses[-1].as_grpc(),
        ego_dynamic_state=self.dynamic_state,
        camera_images=self._last_rendered_images,
        traffic_objects=self._make_traffic_objects(),
        time_now_us=now_us,
        time_query_us=future_us,
        force_gt=force_gt,
    )
    if bridge_response.has_trajectory:
        drive_trajectory_noisy = Trajectory.from_grpc(
            bridge_response.trajectory
        )
```

#### 4. config への追加

```yaml
endpoints:
  bridge:
    enabled: false
    addresses: ["bridge:50060"]
```

`skip=True`（デフォルト）の場合、Bridge は呼ばれず既存動作と完全互換。

---

### docker-compose によるインテグレーションテスト

AlpaSim + Bridge + Mock Planner の 3 コンテナで動作確認する。

#### docker-compose.integration.yml

```yaml
services:
  bridge:
    build:
      context: ../alpasim-ros2-bridge
    command: ["python3", "-m", "alpasim_ros2_bridge.server", "--port", "50060"]
    network_mode: host   # ROS2 DDS に必要

  mock-planner:
    build:
      context: ../alpasim-ros2-bridge
    command: ["python3", "-m", "tests.mock_planner"]
    network_mode: host
    depends_on:
      - bridge

  alpasim-integration:
    build:
      context: .
      dockerfile: Dockerfile
    command: >
      python3 -m pytest tests/integration/test_bridge_integration.py -v
    network_mode: host
    environment:
      BRIDGE_ADDRESS: "localhost:50060"
    depends_on:
      - bridge
      - mock-planner
```

#### Mock Planner（alpasim-ros2-bridge 側に配置）

`alpasim-ros2-bridge/tests/mock_planner.py`

```python
"""テスト用ダミープランナー。
/clock を受信したら固定の /planning/trajectory を publish する。"""

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from autoware_planning_msgs.msg import Trajectory, TrajectoryPoint


class MockPlanner(Node):
    def __init__(self):
        super().__init__("mock_planner")
        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self._pub = self.create_publisher(
            Trajectory, "/planning/trajectory", 10
        )

    def _on_clock(self, msg):
        traj = Trajectory()
        traj.header.stamp = msg.clock
        traj.header.frame_id = "map"
        for i in range(10):
            pt = TrajectoryPoint()
            pt.pose.position.x = float(i)
            pt.pose.orientation.w = 1.0
            pt.longitudinal_velocity_mps = 10.0
            traj.points.append(pt)
        self._pub.publish(traj)


def main():
    rclpy.init()
    rclpy.spin(MockPlanner())


if __name__ == "__main__":
    main()
```

---

### テストシナリオ

#### テスト 1: gRPC 接続確認

Bridge コンテナが起動し、AlpaSim から gRPC で疎通できることを確認。

```python
async def test_bridge_connection():
    """AlpaSim → Bridge の gRPC 接続が確立できること"""
    channel = grpc.aio.insecure_channel("localhost:50060")
    stub = bridge_pb2_grpc.ROS2BridgeServiceStub(channel)
    version = await stub.get_version(common_pb2.Empty())
    assert version.version_id != ""
    await channel.close()
```

#### テスト 2: step() の往復

AlpaSim から step() を送り、Mock Planner 経由で trajectory が返ること。

```python
async def test_step_round_trip():
    """step() → Bridge → ROS2 → Mock Planner → Bridge → response の往復"""
    channel = grpc.aio.insecure_channel("localhost:50060")
    stub = bridge_pb2_grpc.ROS2BridgeServiceStub(channel)

    await stub.start_session(bridge_pb2.BridgeSessionRequest(
        session_uuid="integration-test",
        camera_names=["front"],
    ))

    response = await stub.step(bridge_pb2.BridgeStepRequest(
        session_uuid="integration-test",
        timestamp_us=1_000_000,
        ego_pose=make_pose(x=10.0),
        force_gt=False,
    ))

    assert response.has_trajectory is True
    assert len(response.trajectory.poses) > 0

    await stub.close_session(bridge_pb2.BridgeSessionCloseRequest(
        session_uuid="integration-test",
    ))
    await channel.close()
```

#### テスト 3: 複数ステップのループ

10 ステップの連続実行で、毎回 trajectory が返り、
タイムスタンプが進行していることを確認。

```python
async def test_multi_step_loop():
    """10ステップ連続で trajectory が安定して返ること"""
    channel = grpc.aio.insecure_channel("localhost:50060")
    stub = bridge_pb2_grpc.ROS2BridgeServiceStub(channel)

    await stub.start_session(bridge_pb2.BridgeSessionRequest(
        session_uuid="loop-test",
        camera_names=[],
    ))

    for i in range(10):
        response = await stub.step(bridge_pb2.BridgeStepRequest(
            session_uuid="loop-test",
            timestamp_us=(i + 1) * 100_000,
            ego_pose=make_pose(x=float(i)),
            force_gt=False,
        ))
        assert response.has_trajectory is True

    await stub.close_session(bridge_pb2.BridgeSessionCloseRequest(
        session_uuid="loop-test",
    ))
    await channel.close()
```

### 実行方法

```bash
# alpasim リポジトリのルートから
docker compose -f docker-compose.integration.yml up --build --abort-on-container-exit
```
