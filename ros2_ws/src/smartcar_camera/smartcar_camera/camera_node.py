from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("topic", "/camera/image_raw")

        self.frame_id = str(self.get_parameter("frame_id").value)
        topic = str(self.get_parameter("topic").value)
        fps = float(self.get_parameter("fps").value)
        camera_index = int(self.get_parameter("camera_index").value)

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, topic, 10)
        self.capture = cv2.VideoCapture(camera_index)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

        if not self.capture.isOpened():
            self.get_logger().error(f"Failed to open camera index {camera_index}.")
        else:
            self.get_logger().info(f"Publishing camera {camera_index} to {topic}.")

        self.timer = self.create_timer(1.0 / max(fps, 1.0), self.publish_frame)

    def publish_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warning("Camera read failed.")
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)

    def destroy_node(self) -> bool:
        if hasattr(self, "capture"):
            self.capture.release()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

