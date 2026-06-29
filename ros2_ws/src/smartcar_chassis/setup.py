from setuptools import setup

package_name = "smartcar_chassis"

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
    description="Chassis adapter for the original smart car controller.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "chassis_node = smartcar_chassis.chassis_node:main",
        ],
    },
)

