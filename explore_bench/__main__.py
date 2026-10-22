"""Explicit prepare/train/validate/select/evaluate/score experiment commands."""

import argparse
import json
from pathlib import Path

from .config import configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sanity = commands.add_parser("sanity")
    sanity.add_argument("--output", type=Path, required=True)
    sanity.add_argument("--allow-dirty", action="store_true")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    training.add_argument("--resume", type=Path)
    training.add_argument("--stop-after-updates", type=int)
    training.add_argument("--allow-dirty", action="store_true")
    validation = commands.add_parser("validate")
    validation.add_argument("--training-run", type=Path, required=True)
    validation.add_argument("--tasks", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    selection = commands.add_parser("select")
    selection.add_argument("--validation", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--config", type=Path)
    evaluation.add_argument("--selection", type=Path)
    evaluation.add_argument("--controller", choices=("mappo", "random", "wait", "cost", "mmpf"), default="mappo")
    evaluation.add_argument("--tasks", type=Path, required=True)
    evaluation.add_argument("--suite", nargs="+", choices=("familiar", "unseen", "validation"), default=["familiar", "unseen"])
    evaluation.add_argument("--output", type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--inputs", nargs="+", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "sanity":
        from .sanity import learning_check
        return 0 if learning_check(args.output, args.allow_dirty)["status"] == "pass" else 1
    elif args.command == "prepare":
        from .evaluation import prepare
        prepare(configuration(args.config), args.output)
    elif args.command == "train":
        from .training import train
        if args.stop_after_updates is not None and args.stop_after_updates < 1:
            parser.error("--stop-after-updates must be positive")
        result = train(configuration(args.config), args.output, args.seed, args.device,
                       args.resume, args.stop_after_updates, args.allow_dirty)
        return 0 if result["status"] == "complete" or args.stop_after_updates is not None else 75
    elif args.command == "validate":
        from .evaluation import validate
        validate(args.training_run, args.tasks, args.output)
    elif args.command == "select":
        from .scoring import select
        selected = select(args.validation, args.output)
        print("selected:", selected["checkpoint"])
    elif args.command == "evaluate":
        from .controllers import ClassicalController, RandomController, WaitController
        from .evaluation import evaluate, read_tasks
        seed, checkpoint_hash = "", ""
        if args.controller == "mappo":
            if not args.selection or args.config:
                parser.error("MAPPO testing requires --selection and uses its frozen configuration")
            from .mappo import ActorController
            with args.selection.open(encoding="utf-8") as handle:
                selection_record = json.load(handle)
            if selection_record.get("kind") != "selection" or selection_record.get("status") != "complete":
                parser.error("invalid checkpoint selection record")
            spec = selection_record["configuration"]
            controller = ActorController(selection_record["selected"]["checkpoint"], spec)
            if controller.checkpoint_hash != selection_record["selected"]["actor_sha256"]:
                raise ValueError("selection checkpoint hash mismatch")
            from .config import file_hash
            controller.selection_sha256 = file_hash(args.selection)
            seed, checkpoint_hash = selection_record["training_seed"], controller.checkpoint_hash
        else:
            if args.selection:
                parser.error("scripted controllers use --config, not --selection")
            spec = configuration(args.config)
            controller = (RandomController() if args.controller == "random" else WaitController()
                          if args.controller == "wait" else ClassicalController(args.controller, None))
        tasks = [task for suite in args.suite for task in read_tasks(args.tasks, spec, suite)]
        evaluate(spec, tasks, args.output, controller, seed, checkpoint_hash)
    else:
        from .scoring import score
        score(args.inputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
