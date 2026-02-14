# AlpaSim ROS2 Bridge 設計・実装計画

## 目的

AlpaSim のシミュレーション出力を ROS2 エコシステムに接続し、
ROS2 ベースのプランナー（運転ポリシー）で AlpaSim のクローズドループシミュレーションを駆動できるようにする。

リアルタイム実行は目的としない。`/clock` によるシミュレーション時刻管理を前提とする。

---

## アーキテクチャ概要

既存の AlpaSim コア（gRPC マイクロサービス構成）はそのまま維持し、
**ROS2 ブリッジノード** を追加する「ブリッジ方式」を採用する。

```
┌──────────────────────────────────────────────────────┐
│  AlpaSim Core (既存)                                  │
│                                                       │
│  loop.py ──gRPC──→ sensorsim / controller /           │
│                     physics / traffic                  │
│     │                                                  │
│     │  driver.drive() の代わりに                       │
│     │  ROS2 Bridge を呼ぶ                              │
│     ▼                                                  │
│  ┌──────────────────────────────────────────┐          │
│  │  ROS2 Bridge Node                        │          │
│  │                                           │          │
│  │  Publisher:                                │          │
│  │    /clock              (rosgraph_msgs/Clock)        │
│  │    /camera/{name}/image_raw (sensor_msgs/Image)     │
│  │    /camera/{name}/camera_info (sensor_msgs/CameraInfo)
│  │    /lidar/points       (sensor_msgs/PointCloud2)    │
│  │    /tf                 (tf2_msgs/TFMessage)         │
│  │    /perception/objects  (autoware_perception_msgs/  │
│  │                          TrackedObjects)            │
│  │    /vehicle/status/velocity                         │
│  │         (autoware_vehicle_msgs/VelocityReport)      │
│  │    /planning/route     (nav_msgs/Path)              │
│  │                                           │          │
│  │  Subscriber:                              │          │
│  │    /planning/trajectory                             │
│  │         (autoware_planning_msgs/Trajectory)         │
│  │         ← ROS2 プランナー                  │          │
│  └──────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘

                          ▲
                          │ ROS2 Topics
                          ▼

┌──────────────────────────────────────────────────────┐
│  ROS2 Planner Node (外部・ユーザー実装)               │
│                                                       │
│  Subscribe: /camera/*, /lidar/*, /tf,                 │
│             /perception/objects, ...                   │
│  Publish:   /planning/trajectory                      │
│             (autoware_planning_msgs/Trajectory)        │
└──────────────────────────────────────────────────────┘
```

---

## ディレクトリ構成

```
src/ros2_bridge/
├── package.xml
├── pyproject.toml
├── setup.cfg
├── launch/
│   └── bridge.launch.py
├── config/
│   └── bridge_params.yaml
└── alpasim_ros2_bridge/
    ├── __init__.py
    ├── bridge_node.py          # メイン ROS2 ノード
    ├── clock_publisher.py      # /clock publish
    ├── sensor_publisher.py     # カメラ・LiDAR publish
    ├── tf_broadcaster.py       # ego + traffic の TF broadcast
    ├── traffic_publisher.py    # 周辺車両情報 publish
    ├── driver_bridge.py        # ROS2 planner ↔ AlpaSim driver 接続
    └── conversions.py          # AlpaSim ↔ ROS2 型変換ユーティリティ
```

---

## 実装ステップ

### Step 1: パッケージ雛形と /clock publisher

ROS2 パッケージ `alpasim_ros2_bridge` を作成し、`/clock` の publish を実装する。

**やること:**

- `package.xml`, `pyproject.toml` 作成（ament_python ビルド）
- `clock_publisher.py`: AlpaSim の `timestamp_us` → `rosgraph_msgs/Clock` 変換・publish
- AlpaSim の `loop.py` に bridge 呼び出しのフックポイントを追加

**時刻変換:**

```python
def timestamp_us_to_ros_time(timestamp_us: int) -> Time:
    return Time(
        sec=int(timestamp_us // 1_000_000),
        nanosec=int((timestamp_us % 1_000_000) * 1000),
    )
```

**AlpaSim 側の変更:**

`loop.py` の `_loop()` 各ステップ冒頭で bridge のコールバックを呼ぶ。
config に `ros2_bridge.enabled: bool = false` を追加し、既存動作に影響しないようにする。

---

### Step 2: TF broadcast（ego pose）

ego 車両のポーズを `/tf` で broadcast する。

**やること:**

- `tf_broadcaster.py`: AlpaSim の `pose_local_to_rig` → TF (`map` → `base_link`)
- AlpaSim 座標系 → ROS2 座標系のマッピング定義

**座標系マッピング:**

| AlpaSim | ROS2 TF frame |
|---------|---------------|
| `local` (world) | `map` |
| `rig` (vehicle body) | `base_link` |
| camera frames | `camera_{name}` |
| `aabb` (bbox center) | `actor_{id}` |

---

### Step 3: センサーデータ publish

sensorsim のレンダリング結果を ROS2 トピックに変換・publish する。

**やること:**

- `sensor_publisher.py`:
  - `RGBRenderReturn.image_bytes` → `sensor_msgs/Image` (encoding: `rgb8` or `bgr8`)
  - カメラ intrinsics (`CameraSpec`) → `sensor_msgs/CameraInfo`
  - `LidarRenderReturn` → `sensor_msgs/PointCloud2`
- 各メッセージの `header.stamp` に AlpaSim の `timestamp_us` から変換した ROS2 時刻をセット
- `header.frame_id` にカメラ名 / lidar フレーム名をセット

**フック箇所:**

`loop.py` の `_send_images()` 完了後、レンダリング結果を bridge に渡す。

---

### Step 4: Driver Bridge（ROS2 プランナー接続）

AlpaSim の `driver.drive()` を ROS2 プランナーからの trajectory で置き換える。

**やること:**

- `driver_bridge.py`:
  - `/planning/trajectory` (`autoware_planning_msgs/Trajectory`) を subscribe
  - 最新の trajectory を保持
  - `drive()` 呼び出し時に最新 trajectory を AlpaSim 形式に変換して返す
- AlpaSim runtime の `DriverServiceWrapper` と同じインターフェースを実装

**autoware_planning_msgs/Trajectory を使う理由:**

`nav_msgs/Path` は pose のリストでしかないが、
`autoware_planning_msgs/Trajectory` は各ポイントに以下を含む:

- `time_from_start` — 開始からの相対時間
- `longitudinal_velocity_mps` — 前進速度 [m/s]
- `acceleration_mps2` — 加速度 [m/s²]
- `heading_rate_rps` — ヨーレート [rad/s]
- `front_wheel_angle_rad` — 前輪舵角 [rad]

AlpaSim の controller が必要とする速度・加速度情報がそのまま含まれるため、
`nav_msgs/Path` よりも適切。

**実装（案1: 最新 trajectory 方式）:**

```python
from autoware_planning_msgs.msg import Trajectory as AwTrajectory

class DriverBridge:
    """ROS2 プランナーからの最新 trajectory を AlpaSim に渡す"""

    def __init__(self, node):
        self.latest_trajectory = None
        self.sub = node.create_subscription(
            AwTrajectory, '/planning/trajectory', self._on_trajectory, 10
        )

    def _on_trajectory(self, msg: AwTrajectory):
        self.latest_trajectory = autoware_trajectory_to_alpasim(msg)

    async def drive(self, time_now_us, time_query_us, **kwargs):
        if self.latest_trajectory is None:
            return None  # force_gt 期間中は AlpaSim が GT を使う
        return self.latest_trajectory
```

**Trajectory 変換の詳細:**

```python
def autoware_trajectory_to_alpasim(msg: AwTrajectory):
    """autoware_planning_msgs/Trajectory → AlpaSim Trajectory

    TrajectoryPoint のフィールドマッピング:
      .pose                       → PoseAtTime.pose
      .time_from_start            → PoseAtTime.timestamp_us (header.stamp 基準)
      .longitudinal_velocity_mps  → DynamicState.linear_velocity.x
      .heading_rate_rps           → DynamicState.angular_velocity.z
    """
```

**AlpaSim 側の変更:**

`loop.py` の `self.driver.drive()` 呼び出し箇所で、
`ros2_bridge.enabled` の場合は `DriverBridge.drive()` を呼ぶように分岐する。

---

### Step 5: Traffic（周辺車両）publish

TrafficService の出力を ROS2 トピックで配信する。

**やること:**

- `traffic_publisher.py`:
  - `ObjectTrajectoryUpdate[]` → `/tf` で各アクターの pose を broadcast
  - `ObjectTrajectoryUpdate[]` → `autoware_perception_msgs/TrackedObjects` として publish
- プランナーが周辺車両を認識できるようにする

**AlpaSim → autoware_perception_msgs マッピング:**

| AlpaSim | autoware_perception_msgs |
|---------|--------------------------|
| `object_id` | `TrackedObject.object_id` (UUID) |
| `AABB(size_x, y, z)` | `Shape(type=BOUNDING_BOX, dimensions=Vector3(x,y,z))` |
| `pose_local_to_aabb` | `TrackedObjectKinematics.pose_with_covariance` |
| `is_static` | `TrackedObjectKinematics.is_stationary` |
| 速度（差分計算） | `TrackedObjectKinematics.twist_with_covariance` |

**ObjectClassification:**

AlpaSim の traffic オブジェクトは車両として扱い、
`ObjectClassification.label = CAR (1)` をデフォルトで設定する。

**VelocityReport の publish:**

ego 車両の速度情報を `autoware_vehicle_msgs/VelocityReport` で配信:

```python
def publish_velocity_report(self, dynamic_state, timestamp_us):
    msg = VelocityReport()
    msg.header.stamp = timestamp_us_to_ros_time(timestamp_us)
    msg.header.frame_id = "base_link"
    msg.longitudinal_velocity = dynamic_state.linear_velocity.x
    msg.lateral_velocity = dynamic_state.linear_velocity.y
    msg.heading_rate = dynamic_state.angular_velocity.z
    self.velocity_pub.publish(msg)
```

---

### Step 6: launch ファイルと設定

**やること:**

- `bridge.launch.py`: bridge ノード起動 + `use_sim_time:=true` 設定
- `bridge_params.yaml`: トピック名、フレーム名、画像フォーマット等の設定
- rviz2 用の設定ファイル（可視化確認用）

**launch 例:**

```python
LaunchDescription([
    Node(
        package='alpasim_ros2_bridge',
        executable='bridge_node',
        parameters=[{
            'use_sim_time': True,
        }],
    ),
])
```

---

## データフロー（1ステップの流れ）

```
Step N 開始
  │
  ├─ 1. /clock publish (now_us)
  │
  ├─ 2. sensorsim.render_rgb() [gRPC, 既存]
  │      └─→ sensor_publisher: /camera/*/image_raw publish
  │
  ├─ 3. tf_broadcaster: ego pose → /tf (map → base_link)
  │
  ├─ 4. driver_bridge.drive()
  │      └─→ 最新の /planning/trajectory を返す
  │          (ROS2 プランナーが非同期に更新)
  │
  ├─ 5. controller.run_controller_and_vehicle() [gRPC, 既存]
  │
  ├─ 6. physics.ground_intersection() [gRPC, 既存]
  │
  ├─ 7. traffic.simulate() [gRPC, 既存]
  │      └─→ traffic_publisher: /perception/objects, /tf publish
  │      └─→ velocity_publisher: /vehicle/status/velocity publish
  │
  └─ 8. update_pose → Step N+1 へ
```

---

## ROS2 プランナー側の要件（ユーザー実装）

ROS2 プランナーは以下を満たす必要がある:

- **Subscribe**（入力）:
  - `/camera/{name}/image_raw` (`sensor_msgs/Image`)
  - `/camera/{name}/camera_info` (`sensor_msgs/CameraInfo`)
  - `/lidar/points` (`sensor_msgs/PointCloud2`) — 必要に応じて
  - `/tf` (`tf2_msgs/TFMessage`) — ego + traffic の座標変換
  - `/perception/objects` (`autoware_perception_msgs/TrackedObjects`) — 周辺車両
  - `/vehicle/status/velocity` (`autoware_vehicle_msgs/VelocityReport`) — 自車速度
- **Publish**（出力）:
  - `/planning/trajectory` (`autoware_planning_msgs/Trajectory`)
    - `header.stamp`: 対応するシミュレーション時刻
    - `header.frame_id`: `map`
    - `points[]`: 将来の `TrajectoryPoint` 列
      - `time_from_start`: 開始からの相対時間
      - `pose`: 各ポイントの位置・姿勢
      - `longitudinal_velocity_mps`: 目標速度 [m/s]
      - `acceleration_mps2`: 目標加速度 [m/s²]（オプション）
      - `heading_rate_rps`: ヨーレート [rad/s]（オプション）
- **パラメータ**: `use_sim_time: true` で起動すること

---

## 補足: レースコンディションと改善の余地

### 現状の制限

案1（最新 trajectory 方式）では、以下のレースコンディションが発生し得る:

```
壁時計 t=0ms: AlpaSim が step N のセンサーを publish
壁時計 t=1ms: AlpaSim が drive() 呼び出し → step N-1 の trajectory を使用
壁時計 t=50ms: ROS2 プランナーが step N のセンサーを処理完了 → publish
               → AlpaSim は既に step N+1 に進んでいる
```

つまり、プランナーは常に **1ステップ以上遅れた計画** で制御することになる。
プランナーの処理速度が AlpaSim のステップ速度より遅い場合、遅延が蓄積する。

### 影響

- 評価結果が壁時計速度（マシン性能）に依存し、**再現性が低下する**可能性がある
- プランナーがフレームをスキップする可能性がある

### 将来の改善案: バリア同期方式

`drive()` 呼び出し時に、プランナーが現在ステップのセンサーを処理し終わるまで待機する:

```python
async def drive(self, time_now_us, time_query_us, **kwargs):
    self.expected_stamp = timestamp_us_to_ros_time(time_now_us)
    self.trajectory_event.clear()

    # プランナーの応答を待つ（非リアルタイムなので待機コストなし）
    await self.trajectory_event.wait()

    return self.latest_trajectory

def _on_trajectory(self, msg: AwTrajectory):
    self.latest_trajectory = autoware_trajectory_to_alpasim(msg)
    if ros_time_to_us(msg.header.stamp) >= self.expected_stamp:
        self.trajectory_event.set()
```

これにより毎ステップ決定論的に動作し、再現性が保証される。
シミュレーション速度はプランナーの推論速度がボトルネックになるが、
非リアルタイム用途では問題にならない。

---

## 技術的な注意事項

### 座標系変換

AlpaSim のポーズは「アクティブ変換」で表現される（`common.proto` の規約）。
ROS2 TF も同様にアクティブ変換なので、直接マッピング可能:

```
AlpaSim: pose_local_to_rig (local → rig への変換)
ROS2:    TF map → base_link
```

Quaternion の並び順に注意:
- AlpaSim (protobuf): `Quat { w, x, y, z }`
- ROS2: `geometry_msgs/Quaternion { x, y, z, w }`

### 依存パッケージ

**ROS2 標準 (common_interfaces):**

- `rclpy`
- `sensor_msgs` — Image, CameraInfo, PointCloud2
- `geometry_msgs` — Pose, PoseStamped, TransformStamped, Twist, Vector3, Quaternion
- `nav_msgs` — Path（ルート可視化用）
- `tf2_ros`, `tf2_msgs` — TF broadcast
- `rosgraph_msgs` — Clock
- `cv_bridge`（画像変換用）

**Autoware メッセージ (autoware_msgs):**

- `autoware_planning_msgs` — Trajectory, TrajectoryPoint（プランナー出力）
- `autoware_perception_msgs` — TrackedObjects, TrackedObject, Shape, ObjectClassification（周辺車両）
- `autoware_vehicle_msgs` — VelocityReport（自車速度）

**AlpaSim:**

- `alpasim_grpc` — protobuf 定義のみ（軽量）

### 既存コードへの影響を最小化

- `loop.py` への変更はコールバックフックの追加のみ
- `config.py` に `ros2_bridge` セクションを追加（`enabled: false` がデフォルト）
- 既存の gRPC サービス群には一切手を入れない
