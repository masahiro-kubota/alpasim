# AlpaSim ROS2 Bridge 設計・実装計画

## 目的

AlpaSim のシミュレーション出力を ROS2 エコシステムに接続し、
ROS2 ベースのプランナー（運転ポリシー）で AlpaSim のクローズドループシミュレーションを駆動できるようにする。

リアルタイム実行は目的としない。`/clock` によるシミュレーション時刻管理を前提とする。

---

## アーキテクチャ概要

AlpaSim の既存マイクロサービス構成に **ROS2 Bridge を gRPC サービスとして追加** する。
sensorsim, controller, physics, traffic と同列のサービスとして動作する。

Bridge は **別プロセス・別コンテナ** で動作する。
AlpaSim 本体と完全に分離されたリポジトリ（`alpasim-ros2-bridge`）で管理する。

```
┌─── AlpaSim Container (既存, Python 3.12) ─────────────────┐
│                                                             │
│  loop.py ──gRPC──→ sensorsim   (画像レンダリング)           │
│           ──gRPC──→ controller (trajectory → 車両制御)      │
│           ──gRPC──→ physics    (地面交差判定)                │
│           ──gRPC──→ traffic    (周辺車両シミュレーション)    │
│           ──gRPC──→ bridge     (★ 新規追加)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                        │ gRPC
                        ▼
┌─── Bridge Container (新規, Ubuntu 24.04, ROS2 Jazzy) ─────┐
│                                                             │
│  ROS2 Bridge Service (gRPC server + ROS2 node)              │
│    ├─ gRPC: AlpaSim からデータを受信                        │
│    ├─ ROS2 Publish:                                         │
│    │    /clock              (rosgraph_msgs/Clock)            │
│    │    /camera/{name}/image_raw  (sensor_msgs/Image)        │
│    │    /camera/{name}/camera_info (sensor_msgs/CameraInfo)  │
│    │    /tf                 (tf2_msgs/TFMessage)             │
│    │    /perception/objects (TrackedObjects)                  │
│    │    /vehicle/status/velocity (VelocityReport)            │
│    │                                                         │
│    └─ ROS2 Subscribe:                                        │
│         /planning/trajectory (autoware Trajectory)           │
│         ← プランナーの応答を待ってから gRPC レスポンスを返す  │
│                                                             │
│  Python 3.12 + ROS2 Jazzy + autoware_msgs                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                        ▲
                        │ ROS2 Topics
                        ▼
┌─── Planner Container (ユーザー実装) ───────────────────────┐
│                                                             │
│  ROS2 Planner Node                                          │
│  Subscribe: /camera/*, /tf, /perception/objects, ...        │
│  Publish:   /planning/trajectory                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## gRPC API 定義 (bridge.proto)

Bridge リポジトリの `alpasim_grpc/v0/bridge.proto` で定義。
必要な `.proto` ファイル（`common.proto`, `bridge.proto`）を Bridge リポジトリに直接配置し、
Docker ビルド時に `grpcio-tools` で `_pb2.py` にコンパイルする。

```protobuf
syntax = "proto3";

package bridge;

import "alpasim_grpc/v0/common.proto";

service ROS2BridgeService {
    rpc start_session (BridgeSessionRequest) returns (common.SessionRequestStatus);
    rpc close_session (BridgeSessionCloseRequest) returns (common.Empty);
    rpc step (BridgeStepRequest) returns (BridgeStepResponse);
    rpc get_version (common.Empty) returns (common.VersionId);
    rpc shut_down (common.Empty) returns (common.Empty);
}

message BridgeSessionRequest {
    string session_uuid = 1;
    repeated string camera_names = 2;
}

message BridgeSessionCloseRequest {
    string session_uuid = 1;
}

message CameraImage {
    string camera_id = 1;
    bytes image_bytes = 2;
    uint32 width = 3;
    uint32 height = 4;
}

message TrafficObject {
    string object_id = 1;
    common.Pose pose = 2;
    common.AABB aabb = 3;
    bool is_static = 4;
}

message BridgeStepRequest {
    string session_uuid = 1;
    fixed64 timestamp_us = 2;
    common.Pose ego_pose = 3;
    common.DynamicState ego_dynamic_state = 4;
    repeated CameraImage camera_images = 5;
    repeated TrafficObject traffic_objects = 6;
    fixed64 time_now_us = 7;
    fixed64 time_query_us = 8;
    bool force_gt = 9;
}

message BridgeStepResponse {
    common.Trajectory trajectory = 1;
    bool has_trajectory = 2;
}
```

---

## データフロー（1ステップの流れ）

```
AlpaSim loop.py                          Bridge Service
     │                                        │
     ├─ sensorsim.render_rgb() [既存]          │
     │    → 画像取得                           │
     │                                         │
     ├─ bridge.step(BridgeStepRequest) ──gRPC──┤
     │   {timestamp, ego_pose, images,         │
     │    traffic_objects, time_now/query}      │
     │                                         ├─ 1. /clock publish
     │        (gRPC 応答待ち)                  ├─ 2. /tf publish (ego)
     │                                         ├─ 3. /camera/* publish
     │                                         ├─ 4. /perception/objects publish
     │                                         ├─ 5. /vehicle/status/velocity publish
     │                                         │
     │                                         ├─ 6. /planning/trajectory を待つ
     │                                         │     ← ROS2 Planner
     │                                         │
     │   BridgeStepResponse ←──────────gRPC────┤
     │   {trajectory}                          │
     │                                         │
     ├─ controller.run() [既存]                │
     ├─ physics.ground_intersection() [既存]   │
     ├─ traffic.simulate() [既存]              │
     └─ update_pose → Step N+1                 │
```

### バリア同期の実現

gRPC の request/response モデルにより、バリア同期が自然に実現される:

1. AlpaSim が `bridge.step()` を gRPC で呼ぶ
2. Bridge はデータを ROS2 トピックに publish
3. Bridge は `/planning/trajectory` の受信を**待機**
4. プランナーが trajectory を publish
5. Bridge が trajectory を gRPC レスポンスとして返す
6. AlpaSim が次の処理 (controller) に進む

非リアルタイムなので待機コストは問題にならない。
プランナーの推論速度がシミュレーション速度を決定する。

---

## リポジトリ構成

Bridge は AlpaSim とは **別リポジトリ** で管理する。

### alpasim-ros2-bridge/ （Bridge リポジトリ）

```
alpasim-ros2-bridge/
├── Dockerfile
├── pyproject.toml            # pytest 設定、メタデータ（ビルドには使わない）
├── .pre-commit-config.yaml
├── alpasim_grpc/             # proto 定義（alpasim からコピー、Docker ビルド時にコンパイル）
│   ├── __init__.py
│   └── v0/
│       ├── __init__.py
│       ├── common.proto
│       └── bridge.proto
├── alpasim_ros2_bridge/      # Bridge 本体
│   ├── __init__.py
│   ├── server.py             # gRPC サーバー + ROS2 ノード統合
│   ├── clock_publisher.py
│   ├── sensor_publisher.py
│   ├── tf_broadcaster.py
│   ├── traffic_publisher.py
│   ├── trajectory_listener.py
│   └── conversions.py
├── tests/
│   ├── conftest.py
│   ├── mock_data.py
│   ├── test_conversions.py   # Layer 1 (ROS2 不要)
│   ├── test_clock_publisher.py
│   ├── test_tf_broadcaster.py
│   ├── test_sensor_publisher.py
│   ├── test_traffic_publisher.py
│   ├── test_trajectory_listener.py
│   └── test_server.py        # Layer 3 (gRPC + ROS2)
└── config/
    └── bridge_params.yaml
```

### alpasim/ （AlpaSim リポジトリ、変更箇所）

```
src/grpc/alpasim_grpc/v0/
    bridge.proto              # Bridge gRPC API 定義を追加
```

---

## Docker ビルド戦略

### 設計方針

- **ベースイメージ**: `ros:jazzy`（Ubuntu 24.04 + Python 3.12）
- **autoware_msgs は apt パッケージ**: `ros-jazzy-autoware-{planning,perception,vehicle}-msgs`
- **alpasim_grpc は .proto をローカル管理**: Docker ビルド時に `grpcio-tools` でコンパイル
- **Bridge コードは colcon 不使用**: PYTHONPATH に配置するだけで十分
- **Python パッケージ管理は uv + venv**: `--system-site-packages` で apt の ROS2 パッケージが見える

### Dockerfile

```dockerfile
FROM ros:jazzy

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install autoware message packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-autoware-planning-msgs \
        ros-jazzy-autoware-perception-msgs \
        ros-jazzy-autoware-vehicle-msgs \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 3. Create venv (--system-site-packages to access rclpy and apt packages)
RUN uv venv --system-site-packages --python python3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"

# 4. Install Python dependencies
RUN uv pip install grpcio grpcio-tools "protobuf>=4.0.0,<5.0.0" \
    dataclasses-json numpy pytest pytest-asyncio

# 5. Copy proto definitions and compile (.proto -> _pb2.py)
COPY alpasim_grpc/ /app/alpasim_grpc/
RUN cd /app && python -c \
    "from grpc_tools.command import build_package_protos; build_package_protos('.', strict_mode=True)"

# 6. Place bridge code on PYTHONPATH
COPY alpasim_ros2_bridge/ /app/alpasim_ros2_bridge/
COPY tests/ /app/tests/
COPY config/ /app/config/

WORKDIR /app

ENTRYPOINT ["/bin/bash", "-c", \
    ". /opt/ros/jazzy/setup.sh && exec \"$@\"", "--"]
CMD ["python3", "-m", "pytest", "tests/", "-v"]
```

### venv + `--system-site-packages` について

uv で venv を作成し `--system-site-packages` を指定することで:
- venv 内から **rclpy, sensor_msgs 等の apt パッケージが見える**
- grpcio, pytest 等は venv 内にクリーンにインストール
- apt パッケージとの競合が発生しない（pluggy 等）
- `--break-system-packages` や `--ignore-installed` 等のハックが不要

---

## AlpaSim 側の変更 (最小限)

> **注**: 以下の変更は Bridge コンテナが動作確認できた後に行う。
> 現時点で必要な変更は `bridge.proto` の追加のみ（済み）。

### loop.py への追加（将来）

```python
if self.bridge is not None:
    bridge_response = await self.bridge.step(
        BridgeStepRequest(
            session_uuid=self.session_uuid,
            timestamp_us=now_us,
            ego_pose=ego_pose,
            ego_dynamic_state=dynamic_state,
            camera_images=rendered_images,
            traffic_objects=prev_traffic_objects,
            time_now_us=now_us,
            time_query_us=future_us,
            force_gt=force_gt,
        )
    )
    if bridge_response.has_trajectory:
        drive_trajectory = Trajectory.from_grpc(bridge_response.trajectory)
```

### config への追加

```yaml
ros2_bridge:
    enabled: false
    grpc_address: "bridge:50060"
```

---

## 座標系マッピング

| AlpaSim | ROS2 TF frame |
|---------|---------------|
| `local` (world) | `map` |
| `rig` (vehicle body) | `base_link` |
| camera frames | `camera_{name}` |
| `aabb` (bbox center) | `actor_{id}` |

Quaternion の並び順:
- AlpaSim (protobuf): `Quat { w, x, y, z }`
- ROS2: `geometry_msgs/Quaternion { x, y, z, w }`

---

## ROS2 プランナー側の要件（ユーザー実装）

- **Subscribe**（入力）:
  - `/camera/{name}/image_raw` (`sensor_msgs/Image`)
  - `/camera/{name}/camera_info` (`sensor_msgs/CameraInfo`)
  - `/tf` (`tf2_msgs/TFMessage`) — ego + traffic
  - `/perception/objects` (`autoware_perception_msgs/TrackedObjects`)
  - `/vehicle/status/velocity` (`autoware_vehicle_msgs/VelocityReport`)
- **Publish**（出力）:
  - `/planning/trajectory` (`autoware_planning_msgs/Trajectory`)
- **パラメータ**: `use_sim_time: true`

---

## 依存関係

### Bridge Container

```
apt (ros:jazzy ベースイメージに同梱):
  rclpy, sensor_msgs, geometry_msgs, tf2_ros, rosgraph_msgs

apt (追加インストール):
  ros-jazzy-autoware-planning-msgs
  ros-jazzy-autoware-perception-msgs
  ros-jazzy-autoware-vehicle-msgs

uv pip (venv 内にインストール):
  grpcio, grpcio-tools, protobuf>=4.0.0,<5.0.0
  dataclasses-json, numpy, pytest, pytest-asyncio

alpasim_grpc:
  .proto ファイルを Bridge リポジトリに直接配置
  Docker ビルド時に grpcio-tools で _pb2.py にコンパイル
```

### AlpaSim Container (既存)

変更なし。`alpasim_grpc` に bridge.proto を追加するのみ。

---

## 補足: Traffic データの1ステップ遅延

周辺車両データは loop.py 内で `traffic.simulate()` の後に取得される。
bridge.step() は driver.drive() の前に呼ばれるため、
traffic データは **前ステップの結果** を送る（1ステップ遅延）。

```
Step N:   bridge.step(traffic=Step N-1 の結果) → drive()
Step N+1: bridge.step(traffic=Step N の結果)   → drive()
```

プランナーにとっては「100ms 前の周辺車両情報」で計画を立てることになるが、
実車でもセンサー遅延は存在するため、実用上問題はない。
