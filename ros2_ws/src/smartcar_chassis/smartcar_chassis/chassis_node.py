from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from smartcar_interfaces.msg import ActionCommand, ChassisStatus


class ChassisNode(Node):
    def __init__(self) -> None:
        super().__init__("chassis_node")
        self.declare_parameter("action_topic", "/decision/action_command")
        self.declare_parameter("status_topic", "/chassis/status")
        self.declare_parameter("enable_hardware", False)
        self.declare_parameter("car_python_path", "external/ascend-devkit/src/E2E-Sample/Car/python")
        self.declare_parameter("center_angle", 90.0)
        self.declare_parameter("deadband_deg", 5.0)

        self.enable_hardware = bool(self.get_parameter("enable_hardware").value)
        self.center_angle = float(self.get_parameter("center_angle").value)
        self.deadband_deg = float(self.get_parameter("deadband_deg").value)
        self.controller: Any | None = None
        self.base_actions: Any | None = None
        self.complex_actions: Any | None = None
        self.connected = False
        self.last_action = "none"
        self.detail = "mock"

        if self.enable_hardware:
            self._load_original_car_stack()

        self.publisher = self.create_publisher(
            ChassisStatus,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.subscription = self.create_subscription(
            ActionCommand,
            str(self.get_parameter("action_topic").value),
            self.on_command,
            10,
        )
        self.timer = self.create_timer(0.1, self.publish_status)

    def _load_original_car_stack(self) -> None:
        car_python_path = Path(str(self.get_parameter("car_python_path").value))
        if not car_python_path.exists():
            self.get_logger().error(f"Original car python path not found: {car_python_path}")
            self.detail = "car_python_path_missing"
            return
        sys.path.insert(0, str(car_python_path.resolve()))
        try:
            controller_mod = importlib.import_module("src.utils.controller")
            self.base_actions = importlib.import_module("src.actions.base_action")
            self.complex_actions = importlib.import_module("src.actions.complex_actions")
            controller_cls = getattr(controller_mod, "Controller")
            self.controller = controller_cls()
            self.connected = True
            self.detail = "hardware_ready"
            self.get_logger().info("Loaded original car controller and action modules.")
        except Exception as exc:  # pragma: no cover - depends on vendor package
            self.connected = False
            self.detail = f"load_failed:{exc}"
            self.get_logger().exception("Failed to load original car stack.")

    def on_command(self, msg: ActionCommand) -> None:
        self.last_action = msg.action
        if not self.enable_hardware or not self.connected:
            self.detail = f"mock_execute:{msg.action}:{msg.reason}"
            return

        action_obj = self._map_action(msg)
        if action_obj is None:
            self.detail = f"no_action_mapping:{msg.action}"
            return
        try:
            self.controller.execute(action_obj)
            self.detail = f"executed:{msg.action}:{msg.reason}"
        except Exception as exc:  # pragma: no cover - depends on hardware
            self.connected = False
            self.detail = f"execute_failed:{exc}"
            self.get_logger().exception("Failed to execute chassis command.")

    def _map_action(self, msg: ActionCommand) -> Any | None:
        if msg.action == "follow_lane":
            if msg.steering_angle < self.center_angle - self.deadband_deg:
                return self._make_base_action(["TurnLeft", "Left"], msg.speed)
            if msg.steering_angle > self.center_angle + self.deadband_deg:
                return self._make_base_action(["TurnRight", "Right"], msg.speed)
            return self._make_base_action(["Advance", "Forward"], msg.speed)
        if msg.action == "stop":
            return self._make_base_action(["Stop"], 0.0)
        if msg.action == "slow":
            return self._make_base_action(["Advance", "Forward"], msg.speed)
        if msg.action == "turn_left":
            return self._make_base_action(["TurnLeft", "Left"], msg.speed)
        if msg.action == "turn_right":
            return self._make_base_action(["TurnRight", "Right"], msg.speed)
        if msg.action == "u_turn":
            return self._make_complex_action(["TurnAround", "UTurn", "TurnRound"], msg.speed)
        if msg.action == "park":
            return self._make_complex_action(["Parking", "Park"], msg.speed)
        return None

    def _make_base_action(self, names: list[str], speed: float) -> Any | None:
        return self._make_action(self.base_actions, names, speed)

    def _make_complex_action(self, names: list[str], speed: float) -> Any | None:
        return self._make_action(self.complex_actions, names, speed) or self._make_base_action(["Stop"], 0.0)

    @staticmethod
    def _make_action(module: Any | None, names: list[str], speed: float) -> Any | None:
        if module is None:
            return None
        for name in names:
            cls = getattr(module, name, None)
            if cls is None:
                continue
            for args in ((speed,), (), (int(speed * 100),)):
                try:
                    signature = inspect.signature(cls)
                    signature.bind_partial(*args)
                    return cls(*args)
                except Exception:
                    continue
        return None

    def publish_status(self) -> None:
        msg = ChassisStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.connected = bool(self.connected)
        msg.state = "hardware" if self.enable_hardware else "mock"
        msg.last_action = self.last_action
        msg.battery_voltage = 0.0
        msg.detail = self.detail
        self.publisher.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ChassisNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
