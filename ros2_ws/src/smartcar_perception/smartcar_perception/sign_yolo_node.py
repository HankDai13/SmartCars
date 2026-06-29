from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from smartcar_interfaces.msg import Detection, DetectionArray

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - handled at runtime on Atlas
    ort = None


class SignYoloNode(Node):
    def __init__(self) -> None:
        super().__init__("sign_yolo_node")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detection_topic", "/perception/detections")
        self.declare_parameter("backend", "mock")
        self.declare_parameter("model_path", "models/yolov5/smartcar_yolov5.onnx")
        self.declare_parameter("input_size", 640)
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter(
            "class_names",
            ["left", "right", "turnaround", "park", "person", "obstacle", "crosswalk"],
        )

        self.backend = str(self.get_parameter("backend").value)
        self.model_path = Path(str(self.get_parameter("model_path").value))
        self.input_size = int(self.get_parameter("input_size").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.nms_threshold = float(self.get_parameter("nms_threshold").value)
        self.class_names = list(self.get_parameter("class_names").value)
        self.bridge = CvBridge()
        self.session = self._load_session()

        image_topic = str(self.get_parameter("image_topic").value)
        detection_topic = str(self.get_parameter("detection_topic").value)
        self.publisher = self.create_publisher(DetectionArray, detection_topic, 10)
        self.subscription = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.get_logger().info(f"YOLO node subscribed to {image_topic}, publishing {detection_topic}.")

    def _load_session(self):
        if self.backend != "onnx":
            self.get_logger().info(f"YOLO backend is {self.backend}.")
            return None
        if ort is None:
            self.get_logger().error("onnxruntime is not installed.")
            return None
        if not self.model_path.exists():
            self.get_logger().error(f"YOLO ONNX model not found: {self.model_path}")
            return None
        return ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def on_image(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        out = DetectionArray()
        out.header = msg.header
        if self.backend == "mock" or self.session is None:
            self.publisher.publish(out)
            return

        out.detections = self._infer_onnx(frame)
        self.publisher.publish(out)

    def _infer_onnx(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        input_name = self.session.get_inputs()[0].name
        pred = self.session.run(None, {input_name: tensor})[0]
        pred = np.squeeze(pred, axis=0)

        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in pred:
            objectness = float(row[4])
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            confidence = objectness * float(class_scores[class_id])
            if confidence < self.confidence_threshold:
                continue
            cx, cy, bw, bh = row[:4]
            xmin = int((cx - bw / 2.0) / self.input_size * width)
            ymin = int((cy - bh / 2.0) / self.input_size * height)
            xmax = int((cx + bw / 2.0) / self.input_size * width)
            ymax = int((cy + bh / 2.0) / self.input_size * height)
            boxes.append([xmin, ymin, max(1, xmax - xmin), max(1, ymax - ymin)])
            scores.append(confidence)
            class_ids.append(class_id)

        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        detections: list[Detection] = []
        for idx in np.array(keep).reshape(-1).tolist() if len(keep) else []:
            x, y, w, h = boxes[idx]
            msg = Detection()
            msg.class_id = int(class_ids[idx])
            msg.class_name = self.class_names[msg.class_id] if msg.class_id < len(self.class_names) else str(msg.class_id)
            msg.confidence = float(scores[idx])
            msg.xmin = float(max(0, x))
            msg.ymin = float(max(0, y))
            msg.xmax = float(min(width - 1, x + w))
            msg.ymax = float(min(height - 1, y + h))
            detections.append(msg)
        return detections


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SignYoloNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

