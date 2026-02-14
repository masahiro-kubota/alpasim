"""ClockPublisher のテスト（Layer 2: rclpy 必要）。

/clock が正しいタイムスタンプで publish されるか検証。
"""

rclpy = __import__("pytest").importorskip("rclpy")

from rosgraph_msgs.msg import Clock

from alpasim_ros2_bridge.clock_publisher import ClockPublisher


class TestClockPublisher:

    def test_publish_updates_clock(self, ros2_node):
        """publish した timestamp_us が /clock に反映されること"""
        received = []
        ros2_node.create_subscription(
            Clock, "/clock", lambda msg: received.append(msg), 10
        )

        clock_pub = ClockPublisher(ros2_node)
        clock_pub.publish(1_500_000)  # 1.5 sec

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
