from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    camera_index = LaunchConfiguration("camera_index")
    enable_hardware = LaunchConfiguration("enable_hardware")
    car_python_path = LaunchConfiguration("car_python_path")
    lane_backend = LaunchConfiguration("lane_backend")
    yolo_backend = LaunchConfiguration("yolo_backend")
    enable_agent = LaunchConfiguration("enable_agent")

    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_index", default_value="0"),
            DeclareLaunchArgument("enable_hardware", default_value="false"),
            DeclareLaunchArgument(
                "car_python_path",
                default_value="external/ascend-devkit/src/E2E-Sample/Car/python",
            ),
            DeclareLaunchArgument("lane_backend", default_value="mock"),
            DeclareLaunchArgument("yolo_backend", default_value="mock"),
            DeclareLaunchArgument("enable_agent", default_value="false"),
            Node(
                package="smartcar_camera",
                executable="camera_node",
                name="camera_node",
                output="screen",
                parameters=[
                    {
                        "camera_index": ParameterValue(camera_index, value_type=int),
                        "topic": "/camera/image_raw",
                    }
                ],
            ),
            Node(
                package="smartcar_perception",
                executable="lane_lfnet_node",
                name="lane_lfnet_node",
                output="screen",
                parameters=[
                    {
                        "backend": lane_backend,
                        "model_path": "models/lfnet/lfnet.onnx",
                        "image_topic": "/camera/image_raw",
                        "lane_topic": "/perception/lane",
                    }
                ],
            ),
            Node(
                package="smartcar_perception",
                executable="sign_yolo_node",
                name="sign_yolo_node",
                output="screen",
                parameters=[
                    {
                        "backend": yolo_backend,
                        "model_path": "models/yolov5/smartcar_yolov5.onnx",
                        "image_topic": "/camera/image_raw",
                        "detection_topic": "/perception/detections",
                    }
                ],
            ),
            Node(
                package="smartcar_decision",
                executable="decision_node",
                name="decision_node",
                output="screen",
                parameters=[
                    {
                        "enable_agent_advice": ParameterValue(enable_agent, value_type=bool),
                    }
                ],
            ),
            Node(
                package="smartcar_chassis",
                executable="chassis_node",
                name="chassis_node",
                output="screen",
                parameters=[
                    {
                        "enable_hardware": ParameterValue(enable_hardware, value_type=bool),
                        "car_python_path": car_python_path,
                    }
                ],
            ),
            Node(
                package="smartcar_agent",
                executable="agent_node",
                name="agent_node",
                output="screen",
                condition=IfCondition(enable_agent),
                parameters=[
                    {
                        "enabled": ParameterValue(enable_agent, value_type=bool),
                    }
                ],
            ),
        ]
    )

