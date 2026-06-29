from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from smartcar_interfaces.msg import Lane

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - handled at runtime on Atlas
    ort = None


class LaneLFNetNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_lfnet_node")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("lane_topic", "/perception/lane")
        self.declare_parameter("backend", "mock")
        self.declare_parameter("model_path", "models/lfnet/lfnet.onnx")
        self.declare_parameter("input_width", 320)
        self.declare_parameter("input_height", 180)
        self.declare_parameter("center_angle", 90.0)
        self.declare_parameter("mock_valid", True)
        self.declare_parameter("mock_hz", 10.0)

        self.backend = str(self.get_parameter("backend").value)
        self.model_path = Path(str(self.get_parameter("model_path").value))
        self.input_width = int(self.get_parameter("input_width").value)
        self.input_height = int(self.get_parameter("input_height").value)
        self.center_angle = float(self.get_parameter("center_angle").value)
        self.mock_valid = bool(self.get_parameter("mock_valid").value)
        self.bridge = CvBridge()
        self.session = self._load_session()

        image_topic = str(self.get_parameter("image_topic").value)
        lane_topic = str(self.get_parameter("lane_topic").value)
        self.publisher = self.create_publisher(Lane, lane_topic, 10)
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f"Lane node subscribed to {image_topic}, publishing {lane_topic}.")
        self.mock_timer = None
        if self.backend == "mock":
            mock_hz = float(self.get_parameter("mock_hz").value)
            self.mock_timer = self.create_timer(1.0 / max(mock_hz, 1.0), self.publish_mock_lane)

    def _load_session(self):
        if self.backend != "onnx":
            self.get_logger().info(f"LFNet backend is {self.backend}.")
            return None
        if ort is None:
            self.get_logger().error("onnxruntime is not installed.")
            return None
        if not self.model_path.exists():
            self.get_logger().error(f"LFNet ONNX model not found: {self.model_path}")
            return None
        return ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def publish_mock_lane(self) -> None:
        lane = Lane()
        lane.header.stamp = self.get_clock().now().to_msg()
        lane.header.frame_id = "mock_lane"
        lane.source = self.backend
        lane.steering_angle = self.center_angle
        lane.lateral_offset = 0.0
        lane.confidence = 1.0 if self.mock_valid else 0.0
        lane.valid = self.mock_valid
        self.publisher.publish(lane)

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        lane = Lane()
        lane.header = msg.header
        lane.source = self.backend

        if self.backend == "mock" or self.session is None:
            lane.steering_angle = self.center_angle
            lane.lateral_offset = 0.0
            lane.confidence = 1.0 if self.mock_valid else 0.0
            lane.valid = self.mock_valid
            self.publisher.publish(lane)
            return

        angle = self._infer_onnx(frame)
        lane.steering_angle = float(angle)
        lane.lateral_offset = float((angle - self.center_angle) / max(self.center_angle, 1.0))
        lane.confidence = 1.0
        lane.valid = True
        self.publisher.publish(lane)

    def _infer_onnx(self, frame: np.ndarray) -> float:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: tensor})[0]
        return float(output.reshape(-1)[0])


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LaneLFNetNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
