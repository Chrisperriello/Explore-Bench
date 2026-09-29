"""Standalone adapter for the canonical Level-0 communication model."""

import importlib.util
import os


communication_module_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "onpolicy",
        "onpolicy",
        "envs",
        "GridEnv",
        "communication.py",
    )
)
communication_module_spec = importlib.util.spec_from_file_location(
    "explore_bench_level0_communication", communication_module_path
)
communication_module = importlib.util.module_from_spec(communication_module_spec)
communication_module_spec.loader.exec_module(communication_module)

__all__ = communication_module.__all__
for exported_name in __all__:
    globals()[exported_name] = getattr(communication_module, exported_name)
