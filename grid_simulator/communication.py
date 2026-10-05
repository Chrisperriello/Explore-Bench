"""Expose the canonical Level-0 broker to direct standalone scripts.

The authoritative implementation lives in
``onpolicy/onpolicy/envs/GridEnv/communication.py``.  The historical
``grid_simulator/GridEnv.py`` entry point is commonly executed directly and
does not require the local ``onpolicy`` package to be installed.  This adapter
therefore loads the canonical file by repository-relative path and re-exports
its declared public API.

Keeping this file as a thin adapter is important experimentally: learned and
handwritten controllers use identical range, loss, collision, budget, and
belief-fusion semantics instead of two implementations that may drift.
"""

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
