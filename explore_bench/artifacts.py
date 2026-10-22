"""Atomic metadata, immutable attempts, and reproducibility records."""

import csv
import datetime
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

from .config import OBSERVATIONS, ROOT, SCHEMA, SEMANTICS, file_hash


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def provenance():
    def git(*arguments):
        return subprocess.check_output(["git"] + list(arguments), cwd=str(ROOT), text=True).strip()
    return {
        "repository_commit": git("rev-parse", "HEAD"),
        "repository_status": git("status", "--porcelain", "--untracked-files=no"),
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "hostname": platform.node(),
        "hardware": {"machine": platform.machine(), "processor": platform.processor(),
                     "platform": platform.platform(), "logical_cpus": os.cpu_count()},
        "package_versions": {item.metadata["Name"]: item.version for item in metadata.distributions()
                             if item.metadata["Name"]},
        "scheduler": {name: os.environ.get(name) for name in
                      ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")},
        "source_hashes": {str(path.relative_to(ROOT)): file_hash(path)
                          for folder in (ROOT / "explore_bench", ROOT / "onpolicy/onpolicy")
                          for path in sorted(folder.rglob("*.py"))},
    }


def new_run(directory, spec, kind, **metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": SCHEMA, "environment_semantics": SEMANTICS,
        "observation_version": OBSERVATIONS, "kind": kind,
        "status": "running", "started_at": now(), "completed_at": None,
        "configuration": spec, **provenance(), **metadata,
    }
    write_json(directory / "manifest.json", record)
    return record


class Table:
    """A streaming table with strict fields and immediate row flushing."""

    def __init__(self, path, fields):
        self.handle = Path(path).open("x", encoding="utf-8", newline="")
        self.fields = fields
        self.writer = csv.DictWriter(self.handle, fieldnames=fields)
        self.writer.writeheader()

    def append(self, row):
        self.writer.writerow({key: json.dumps(json_safe(value), separators=(",", ":"))
                              if isinstance(value, (list, tuple, dict, np.ndarray))
                              else json_safe(value) for key, value in row.items()})
        self.handle.flush()

    def close(self):
        self.handle.close()
