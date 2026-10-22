"""Versioned experiment configuration and explicit, paired episode tasks."""

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASETS = "onpolicy/onpolicy/envs/GridEnv/datasets/"
SEMANTICS = "level0_fixed_edge_v1"
OBSERVATIONS = "level0_ctde_v1"
SCHEMA = 2
DEFAULT = {
    "name": "mappo_omni_3p5_v1",
    "num_agents": 2,
    "resolution": 0.1,
    "sensor_type": "omnidirectional",
    "sensor_range": 3.5,
    "communication": {"mode": "perfect"},
    "target_coverage": 0.99,
    "max_steps": 1000,
    "thresholds": [0.90, 0.98, 0.99],
    "reward_new_cell": 0.02,
    "reward_success": 0.1,
    "training_seeds": [1, 2, 3, 4, 5],
    "maps": {
        "train": [DATASETS + name + ".pgm" for name in (
            "room_1", "room_2", "room_3", "room1_modified", "corridor",
            "corridor_sym", "corridor_asym", "corner")],
        "validation": [DATASETS + name + ".pgm" for name in (
            "room_with_corner", "room_with_corridor")],
        "test": [DATASETS + name + ".pgm" for name in (
            "square_loop", "loop_with_corridor_sym")],
    },
    "training": {
        "num_envs": 8, "rollout_length": 128, "total_steps": 1024000,
        "checkpoint_interval": 102400, "ppo_epochs": 5, "minibatches": 1,
        "learning_rate": 0.0005, "clip": 0.1, "gamma": 0.99,
        "gae_lambda": 0.95, "entropy_coefficient": 0.01, "hidden_size": 64,
        "max_grad_norm": 10.0,
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def map_path(name):
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def configuration(path=None):
    value = copy.deepcopy(DEFAULT)
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            supplied = json.load(handle)
        unknown = set(supplied) - set(value)
        if unknown:
            raise ValueError("unknown experiment keys: {}".format(sorted(unknown)))
        for key, item in supplied.items():
            if key == "training":
                if set(item) - set(value[key]):
                    raise ValueError("unknown training configuration key")
                value[key].update(item)
            else:
                value[key] = item
    if not isinstance(value["num_agents"], int) or value["num_agents"] != 2:
        raise ValueError("v1 is frozen to two robots")
    if value["sensor_type"] not in ("omnidirectional", "four_beam"):
        raise ValueError("unsupported sensor_type")
    for key in ("resolution", "sensor_range"):
        if not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    if (not 0 < value["target_coverage"] <= 1 or not isinstance(value["max_steps"], int)
            or value["max_steps"] < 1):
        raise ValueError("invalid target coverage or episode cap")
    if value["target_coverage"] != 0.99 or value["thresholds"] != [0.90, 0.98, 0.99]:
        raise ValueError("schema-v2 initial recipe requires target 0.99 and thresholds 0.90/0.98/0.99")
    if value["thresholds"] != sorted(set(value["thresholds"])) or any(
        not 0 < threshold <= value["target_coverage"] for threshold in value["thresholds"]
    ):
        raise ValueError("thresholds must be sorted, unique and at most the target")
    if any(not math.isfinite(value[k]) or value[k] < 0 for k in ("reward_new_cell", "reward_success")):
        raise ValueError("reward coefficients must be finite and nonnegative")
    if not value["training_seeds"] or len(set(value["training_seeds"])) != len(value["training_seeds"]):
        raise ValueError("training seeds must be nonempty and unique")
    if any(not isinstance(seed, int) or seed < 0 or seed >= 2 ** 32 for seed in value["training_seeds"]):
        raise ValueError("training seeds must be nonnegative 32-bit integers")
    settings = value["training"]
    for key in ("num_envs", "rollout_length", "total_steps", "checkpoint_interval", "ppo_epochs", "minibatches", "hidden_size"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(key + " must be a positive integer")
    batch = settings["num_envs"] * settings["rollout_length"]
    if settings["total_steps"] % batch or settings["checkpoint_interval"] % batch:
        raise ValueError("step budgets and checkpoint intervals must divide into complete rollouts")
    if batch * value["num_agents"] % settings["minibatches"]:
        raise ValueError("minibatches must divide the number of robot samples")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(key + " must be positive and finite")
    for key in ("clip", "gamma", "gae_lambda"):
        if not 0 < settings[key] <= 1:
            raise ValueError(key + " must be in (0, 1]")
    if not math.isfinite(settings["entropy_coefficient"]) or settings["entropy_coefficient"] < 0:
        raise ValueError("invalid entropy coefficient")
    from onpolicy.envs.GridEnv.communication import CommunicationConfig
    CommunicationConfig(**value["communication"])
    if set(value["maps"]) != {"train", "validation", "test"}:
        raise ValueError("maps must define train, validation and test")
    seen = set()
    for split in ("train", "validation", "test"):
        if not value["maps"][split]:
            raise ValueError("empty map split: " + split)
        for name in value["maps"][split]:
            fingerprint = file_hash(map_path(name))
            if fingerprint in seen:
                raise ValueError("duplicate map content in experiment splits: " + name)
            seen.add(fingerprint)
    return value


@dataclass
class EpisodeTask:
    task_id: str
    suite: str
    map_path: str
    map_sha256: str
    seed: int
    start_positions: list
    start_headings_rad: list

    def to_dict(self):
        return asdict(self)


def task_for(spec, name, seed, suite):
    grid = np.asarray(Image.open(map_path(name)))
    if grid.ndim != 2 or not set(np.unique(grid)) <= {0, 205, 254}:
        raise ValueError("map must be a 2D occupancy grid containing 0/205/254")
    free = np.argwhere(grid == 254)
    if len(free) < spec["num_agents"]:
        raise ValueError("map has insufficient free spawn cells")
    rng = random.Random(int(seed))
    indices = rng.sample(range(len(free)), spec["num_agents"])
    return EpisodeTask(
        "{}__{}__seed_{}".format(suite, Path(name).stem, seed), suite, str(name),
        file_hash(map_path(name)), int(seed), free[indices].tolist(),
        [rng.randrange(4) * math.pi / 2 for _ in indices],
    )


def evaluation_tasks(spec, suite):
    split, first = {
        "validation": ("validation", 10001),
        "familiar": ("train", 20001),
        "unseen": ("test", 30001),
    }[suite]
    return [task_for(spec, name, seed, suite) for name in spec["maps"][split]
            for seed in range(first, first + 10)]
