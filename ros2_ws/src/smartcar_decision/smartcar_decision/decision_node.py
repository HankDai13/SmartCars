from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from smartcar_interfaces.msg import ActionCommand, AgentAdvice, Detection, DetectionArray, Lane


ACTION_MAP = {
    "left": "turn_left",
    "right": "turn_right",
    "turnaround": "u_turn",
    "park": "park",
    "crosswalk": "slow",
}


class DecisionNode(Node):
    def __init__(self) -> None:
        super().__init__("decision_node")
        self.declare_parameter("lane_topic", "/perception/lane")
        self.declare_parameter("detection_topic", "/perception/detections")
        self.declare_parameter("agent_topic", "/agent/advice")
        self.declare_parameter("action_topic", "/decision/action_command")
        self.declare_parameter("decision_hz", 10.0)
        self.declare_parameter("normal_speed", 0.32)
        self.declare_parameter("slow_speed", 0.18)
        self.declare_parameter("min_detection_confidence", 0.35)
        self.declare_parameter("detection_cooldown_s", 1.5)
        self.declare_parameter("enable_agent_advice", False)
        self.declare_parameter("emergency_classes", ["person", "obstacle"])

        self.normal_speed = float(self.get_parameter("normal_speed").value)
        self.slow_speed = float(self.get_parameter("slow_speed").value)
        self.min_detection_confidence = float(self.get_parameter("min_detection_confidence").value)
        self.detection_cooldown_s = float(self.get_parameter("detection_cooldown_s").value)
        self.enable_agent_advice = bool(self.get_parameter("enable_agent_advice").value)
        self.emergency_classes = set(self.get_parameter("emergency_classes").value)

        self.last_lane: Lane | None = None
        self.last_detections: list[Detection] = []
        self.last_agent_advice: AgentAdvice | None = None
        self.last_discrete_action_time = 0.0

        self.create_subscription(Lane, str(self.get_parameter("lane_topic").value), self.on_lane, 10)
        self.create_subscription(
            DetectionArray,
            str(self.get_parameter("detection_topic").value),
            self.on_detections,
            10,
        )
        self.create_subscription(AgentAdvice, str(self.get_parameter("agent_topic").value), self.on_agent_advice, 10)
        self.publisher = self.create_publisher(ActionCommand, str(self.get_parameter("action_topic").value), 10)
        hz = float(self.get_parameter("decision_hz").value)
        self.timer = self.create_timer(1.0 / max(hz, 1.0), self.tick)

    def on_lane(self, msg: Lane) -> None:
        self.last_lane = msg

    def on_detections(self, msg: DetectionArray) -> None:
        self.last_detections = list(msg.detections)

    def on_agent_advice(self, msg: AgentAdvice) -> None:
        self.last_agent_advice = msg

    def tick(self) -> None:
        command = self.decide()
        command.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(command)

    def decide(self) -> ActionCommand:
        emergency = self.best_detection(self.emergency_classes)
        if emergency is not None:
            return self.command("stop", 0.0, 90.0, 0.0, f"emergency:{emergency.class_name}")

        sign = self.best_detection(set(ACTION_MAP.keys()))
        now = time.monotonic()
        if sign is not None and now - self.last_discrete_action_time >= self.detection_cooldown_s:
            self.last_discrete_action_time = now
            action = ACTION_MAP[sign.class_name]
            speed = self.slow_speed if action == "slow" else self.normal_speed
            return self.command(action, speed, 90.0, 0.0, f"sign:{sign.class_name}")

        if self.enable_agent_advice and self.last_agent_advice and self.last_agent_advice.active:
            advice = self.last_agent_advice
            if advice.confidence >= 0.5 and advice.action in {
                "follow_lane",
                "turn_left",
                "turn_right",
                "u_turn",
                "park",
                "stop",
                "resume",
            }:
                return self.command(advice.action, self.slow_speed, 90.0, 0.0, f"agent:{advice.rationale}")

        lane = self.last_lane
        if lane is not None and lane.valid:
            return self.command(
                "follow_lane",
                self.normal_speed,
                lane.steering_angle,
                lane.lateral_offset,
                "lane",
            )

        return self.command("stop", 0.0, 90.0, 0.0, "missing_lane")

    def best_detection(self, classes: set[str]) -> Detection | None:
        candidates = [
            det
            for det in self.last_detections
            if det.class_name in classes and det.confidence >= self.min_detection_confidence
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda det: det.confidence)

    @staticmethod
    def command(action: str, speed: float, steering: float, offset: float, reason: str) -> ActionCommand:
        msg = ActionCommand()
        msg.action = action
        msg.speed = float(speed)
        msg.steering_angle = float(steering)
        msg.lateral_offset = float(offset)
        msg.reason = reason
        return msg


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

