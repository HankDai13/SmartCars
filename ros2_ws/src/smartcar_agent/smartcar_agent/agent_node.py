from __future__ import annotations

import rclpy
from rclpy.node import Node
from smartcar_interfaces.msg import AgentAdvice, ChassisStatus, Detection, DetectionArray, Lane


class AgentNode(Node):
    def __init__(self) -> None:
        super().__init__("agent_node")
        self.declare_parameter("enabled", False)
        self.declare_parameter("policy_backend", "rule")
        self.declare_parameter("lane_topic", "/perception/lane")
        self.declare_parameter("detection_topic", "/perception/detections")
        self.declare_parameter("status_topic", "/chassis/status")
        self.declare_parameter("advice_topic", "/agent/advice")
        self.declare_parameter("agent_hz", 2.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.policy_backend = str(self.get_parameter("policy_backend").value)
        self.last_lane: Lane | None = None
        self.last_detections: list[Detection] = []
        self.last_status: ChassisStatus | None = None

        self.create_subscription(Lane, str(self.get_parameter("lane_topic").value), self.on_lane, 10)
        self.create_subscription(
            DetectionArray,
            str(self.get_parameter("detection_topic").value),
            self.on_detections,
            10,
        )
        self.create_subscription(ChassisStatus, str(self.get_parameter("status_topic").value), self.on_status, 10)
        self.publisher = self.create_publisher(AgentAdvice, str(self.get_parameter("advice_topic").value), 10)
        hz = float(self.get_parameter("agent_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 1.0), self.tick)

    def on_lane(self, msg: Lane) -> None:
        self.last_lane = msg

    def on_detections(self, msg: DetectionArray) -> None:
        self.last_detections = list(msg.detections)

    def on_status(self, msg: ChassisStatus) -> None:
        self.last_status = msg

    def tick(self) -> None:
        advice = AgentAdvice()
        advice.header.stamp = self.get_clock().now().to_msg()
        advice.active = False

        if self.enabled:
            advice = self.rule_policy()
            advice.header.stamp = self.get_clock().now().to_msg()

        self.publisher.publish(advice)

    def rule_policy(self) -> AgentAdvice:
        advice = AgentAdvice()
        advice.active = True
        advice.action = "follow_lane"
        advice.confidence = 0.55
        advice.rationale = f"{self.policy_backend}:nominal"

        if self.last_lane is not None and not self.last_lane.valid:
            advice.action = "stop"
            advice.confidence = 0.7
            advice.rationale = "lane_invalid"
            return advice

        for det in sorted(self.last_detections, key=lambda item: item.confidence, reverse=True):
            if det.confidence < 0.45:
                continue
            if det.class_name == "turnaround":
                advice.action = "u_turn"
                advice.rationale = "agent_route:turnaround"
                return advice
            if det.class_name == "park":
                advice.action = "park"
                advice.rationale = "agent_route:park"
                return advice

        return advice


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

