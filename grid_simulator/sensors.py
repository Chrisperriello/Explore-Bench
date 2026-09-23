"""Standalone adapter for the canonical Level-0 sensor implementations."""

import importlib.util
import os


sensor_module_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "onpolicy",
        "onpolicy",
        "envs",
        "GridEnv",
        "sensors.py",
    )
)
sensor_module_spec = importlib.util.spec_from_file_location(
    "explore_bench_level0_sensors", sensor_module_path
)
sensor_module = importlib.util.module_from_spec(sensor_module_spec)
sensor_module_spec.loader.exec_module(sensor_module)

__all__ = sensor_module.__all__
for exported_name in __all__:
    globals()[exported_name] = getattr(sensor_module, exported_name)
