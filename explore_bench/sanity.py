"""Small controlled exploration experiment, separate from research test maps."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import write_json
from .config import configuration
from .controllers import RandomController
from .evaluation import evaluate, prepare, read_tasks, validate
from .mappo import ActorController
from .scoring import checked_rows, select, summary
from .training import train


def learning_check(output, allow_dirty=False):
    """Demonstrate learning on generated corridors, not the held-out paper maps.

    The predeclared check is an AUC increase of 0.01 over the untrained actor
    on validation, with a selected nonzero training step. This engineering
    check is descriptive and is not a statistical performance claim.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for index in range(3):
        grid = np.full((29, 29), 205, dtype=np.uint8)
        row = 13 + index
        grid[row - 1:row + 2, 1:28] = 0
        grid[row, 2:27] = 254
        path = output / "corridor_{}.pgm".format(index)
        Image.fromarray(grid).save(str(path))
        paths.append(str(path.resolve()))
    source = {"name": "controlled_corridor_learning_v1", "sensor_range": 0.3,
              "max_steps": 40, "training_seeds": [7],
              "maps": dict(zip(("train", "validation", "test"), [[path] for path in paths])),
              "training": {"num_envs": 2, "rollout_length": 32, "total_steps": 8192,
                           "checkpoint_interval": 1024, "ppo_epochs": 5}}
    write_json(output / "configuration.json", source)
    spec = configuration(output / "configuration.json")
    prepare(spec, output / "tasks.json")
    train(spec, output / "train", 7, allow_dirty=allow_dirty)
    validate(output / "train", output / "tasks.json", output / "validation")
    choice = select(output / "validation", output / "selection.json")
    _, baseline_rows, _ = checked_rows(output / "validation/step_000000000000")
    baseline = summary(baseline_rows, spec["max_steps"])
    evaluate(spec, read_tasks(output / "tasks.json", spec, "validation"), output / "random_validation", RandomController())
    _, random_rows, _ = checked_rows(output / "random_validation")
    actor = ActorController(choice["checkpoint"], spec)
    evaluate(spec, read_tasks(output / "tasks.json", spec, "unseen"), output / "test", actor, 7, actor.checkpoint_hash)
    _, test_rows, _ = checked_rows(output / "test")
    improvement = choice["score"]["mean_coverage_auc"] - baseline["mean_coverage_auc"]
    record = {"check": "controlled_corridor_learning_v1", "status": "pass" if
              choice["environment_steps"] > 0 and improvement >= 0.01 else "fail",
              "untrained_validation": baseline, "selected_validation": choice["score"],
              "selected_step": choice["environment_steps"], "validation_auc_improvement": improvement,
              "random_validation": summary(random_rows, spec["max_steps"]),
              "selected_unseen": summary(test_rows, spec["max_steps"]),
              "criterion": "selected step > 0 and validation AUC improves by >= 0.01",
              "limitation": "One synthetic training seed; engineering evidence only, not research performance."}
    write_json(output / "learning_check.json", record)
    print(json.dumps(record, indent=2))
    return record
