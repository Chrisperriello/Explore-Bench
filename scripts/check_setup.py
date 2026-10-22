"""Check the installed Level-0 stack without checkout/PYTHONPATH assistance."""

import argparse
import datetime
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import distribution, distributions
from pathlib import Path


def write_record(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_dependencies():
    result = subprocess.run([sys.executable, "-m", "pip", "check"],
                            capture_output=True, text=True, timeout=60)
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        print(output, flush=True)
        return {"status": "pass", "output": output}
    # grpcio's CPython 3.8 universal2 wheel contains both native architectures,
    # but its internal WHEEL tag says x86_64. Never suppress dependency conflicts
    # or Linux failures. Accept only this observed macOS metadata defect after
    # the native extension actually loads; retain the warning in the report.
    if (sys.platform == "darwin" and platform.machine() == "arm64" and
            output == "grpcio 1.70.0 is not supported on this platform"):
        package = distribution("grpcio")
        if (package.version == "1.70.0" and
                "Tag: cp38-cp38-macosx_10_14_x86_64" in package.read_text("WHEEL")):
            native = importlib.import_module("grpc._cython.cygrpc")
            print("Known grpcio universal2 metadata warning; native arm64 import passed:",
                  output, flush=True)
            return {"status": "known_metadata_warning", "output": output,
                    "native_extension": native.__file__}
    raise RuntimeError("Dependency validation failed:\n" + output)


def check(root, prefix, torch_build):
    if sys.version_info[:3] != (3, 8, 20):
        raise RuntimeError("Python 3.8.20 is required, got " + sys.version)
    if Path(sys.prefix).resolve() != prefix.resolve():
        raise RuntimeError("Checks are running in the wrong Python environment")
    if os.environ.get("PYTHONHOME") or os.environ.get("PYTHONPATH"):
        raise RuntimeError("Setup checks must not depend on PYTHONHOME or PYTHONPATH")
    modules = (
        "encodings", "numpy", "PIL.Image", "matplotlib.pyplot", "gym", "cv2",
        "torch", "torchvision", "imageio", "icecream", "tensorboard",
        "tensorboardX", "grpc", "setproctitle", "cloudpickle", "wandb", "pytest", "hypothesis",
        "onpolicy.envs.GridEnv.GridEnv", "onpolicy.envs.GridEnv.observations",
        "onpolicy.utils.shared_buffer", "onpolicy.algorithms.r_mappo.r_mappo",
        "explore_bench.config", "explore_bench.session", "explore_bench.training",
        "explore_bench.evaluation", "explore_bench.scoring",
    )
    origins = {}
    for name in modules:
        print("Checking import:", name, flush=True)
        module = importlib.import_module(name)
        origins[name] = str(getattr(module, "__file__", ""))
    import onpolicy
    import explore_bench
    import cv2
    import numpy as np
    import torch

    for module, expected in (
        (onpolicy, root / "onpolicy/onpolicy/__init__.py"),
        (explore_bench, root / "explore_bench/__init__.py"),
    ):
        if Path(module.__file__).resolve() != expected.resolve():
            raise RuntimeError("Wrong checkout imported: " + module.__file__)
    if torch.__version__.split("+")[0] != "2.3.1":
        raise RuntimeError("Unexpected torch version: " + torch.__version__)
    if torch_build == "cpu" and torch.version.cuda is not None:
        raise RuntimeError("Expected the CPU-only PyTorch wheel, not a CUDA build")
    if torch_build == "cu121" and torch.version.cuda != "12.1":
        raise RuntimeError("Expected the CUDA 12.1 PyTorch wheel")
    if cv2.__version__ != "4.8.1" or np.__version__ != "1.23.5":
        raise RuntimeError("Unexpected OpenCV or NumPy version")
    # Test native libraries and NumPy/PyTorch exchange, not just metadata.
    image = np.zeros((4, 4), dtype=np.uint8)
    if cv2.resize(image, (8, 8)).shape != (8, 8):
        raise RuntimeError("OpenCV resize check failed")
    values = torch.from_numpy(np.arange(4, dtype=np.float32))
    if torch.dot(values, values).item() != 14.0:
        raise RuntimeError("CPU tensor execution check failed")
    with tempfile.TemporaryDirectory(prefix="explore-setup-imports-") as directory:
        # Spawn workers and job commands need packages visible outside the repo.
        subprocess.run(
            [sys.executable, "-I", "-m", "explore_bench", "--help"],
            cwd=directory, check=True, timeout=60,
        )
    from explore_bench.config import configuration
    spec = configuration(root / "experiments/mappo_omni_3p5_v1.json")
    return {
        "status": "imports_passed_tests_pending",
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version, "prefix": str(prefix.resolve()),
        "repository_root": str(root.resolve()), "platform": platform.platform(),
        "torch": torch.__version__, "torch_build": torch_build,
        "cuda_available": torch.cuda.is_available(), "imports": origins,
        "dependency_check": check_dependencies(),
        "configuration": spec,
        "packages": {d.metadata["Name"]: d.version for d in distributions()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--torch-build", choices=("cpu", "cu121", "native"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--finish", type=Path)
    parser.add_argument("--tests", choices=("true", "false"))
    parser.add_argument("--log")
    parser.add_argument("--dependencies", action="store_true")
    args = parser.parse_args()
    if args.dependencies:
        check_dependencies()
        return
    if args.finish:
        if not args.tests or not args.log:
            parser.error("--finish requires --tests and --log")
        record = json.loads(args.finish.read_text(encoding="utf-8"))
        record.update(status="pass" if args.tests == "true" else "imports_only",
                      regression_tests="passed" if args.tests == "true" else "skipped",
                      setup_log=args.log)
        write_record(args.finish, record)
        return
    if not all((args.root, args.prefix, args.torch_build, args.output)):
        parser.error("--root, --prefix, --torch-build and --output are required")
    write_record(args.output, check(args.root, args.prefix, args.torch_build))


if __name__ == "__main__":
    main()
