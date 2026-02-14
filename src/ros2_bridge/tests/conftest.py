"""ROS2 テスト用共通フィクスチャ"""

import uuid

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
    """テストごとに使い捨て ROS2 ノードを作成。

    ノード名はユニークにして、テスト間の干渉を防ぐ。
    """
    name = f"test_node_{uuid.uuid4().hex[:8]}"
    node = rclpy.create_node(name)
    yield node
    node.destroy_node()
