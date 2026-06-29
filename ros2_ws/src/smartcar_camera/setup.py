from setuptools import setup

package_name = "smartcar_camera"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SmartCars Team",
    maintainer_email="team@example.com",
    description="Camera publisher for the smart car.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = smartcar_camera.camera_node:main",
        ],
    },
)

