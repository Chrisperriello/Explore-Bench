"""Fixed-task evaluation that never constructs a learning algorithm."""

import copy
import json
import random
import traceback
from pathlib import Path
from types import SimpleNamespace

from .artifacts import new_run, now, write_json
from .config import EpisodeTask, SEMANTICS, digest, evaluation_tasks, file_hash
from .recording import EpisodeLedger, Recorder
from .session import Session


def prepare(spec, output):
    tasks = {suite: [task.to_dict() for task in evaluation_tasks(spec, suite)]
             for suite in ("validation", "familiar", "unseen")}
    output = Path(output)
    if output.exists():
        raise FileExistsError("task table is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"configuration": spec, "configuration_sha256": digest(spec),
                        "environment_semantics": SEMANTICS, "tasks_sha256": digest(tasks), "tasks": tasks})


def read_tasks(path, spec, suite):
    with Path(path).open(encoding="utf-8") as handle:
        record = json.load(handle)
    if (record["configuration_sha256"] != digest(spec) or record["tasks_sha256"] != digest(record["tasks"])
            or record["environment_semantics"] != SEMANTICS):
        raise ValueError("task table configuration, semantics or hash mismatch")
    tasks = [EpisodeTask(**item) for item in record["tasks"][suite]]
    expected = [task.to_dict() for task in evaluation_tasks(spec, suite)]
    if [task.to_dict() for task in tasks] != expected:
        raise ValueError("task table does not match the frozen map/spawn design")
    return tasks


def baseline_view(session):
    """Only delivered beliefs and robot positions required by old selectors.

    The existing Cost/MMPF selectors only consult agent i's position during
    its selection. No ground-truth map is supplied to this adapter.
    """
    return {"num_agents": session.num_agents, "width": session.width, "height": session.height,
            "resolution": session.resolution, "minDis2Frontier": 2 * session.resolution,
            "visualization": False, "agent_pos": copy.deepcopy(session.agent_pos),
            "built_map": [grid.copy() for grid in session.communication_broker.belief_maps],
            "communication_broker": SimpleNamespace(belief_maps=[grid.copy() for grid in session.communication_broker.belief_maps]),
            "_random": random.Random(session.task.seed)}


def evaluate(spec, tasks, output, controller, training_seed="", checkpoint_hash=""):
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("evaluation requires nonempty unique tasks")
    record = new_run(output, spec, "evaluation", controller=controller.name,
                     training_seed=training_seed, checkpoint_sha256=checkpoint_hash,
                     selection_sha256=getattr(controller, "selection_sha256", None),
                     tasks=[task.to_dict() for task in tasks],
                     tasks_sha256=digest([task.to_dict() for task in tasks]), episodes_completed=0)
    recorder = None
    session = Session(spec)
    # Classical adapters receive a restricted view; all other controllers only
    # receive actor observations and action masks.
    if controller.name in ("cost", "mmpf"):
        controller.bind_view = lambda: baseline_view(session)
    try:
        recorder = Recorder(output, spec["thresholds"], full=True)
        for task in tasks:
            actor, _, available = session.reset_task(task)
            controller.reset(task)
            state = None
            ledger = EpisodeLedger(session, controller.name, training_seed, checkpoint_hash)
            ledger.observe(session, recorder)
            while not (session.terminated or session.truncated):
                request, state = controller.act(actor, state, available, deterministic=True)
                (actor, _, available), _, _, _, _ = session.step_request(request)
                ledger.observe(session, recorder)
            ledger.finish(session, recorder)
            record["episodes_completed"] += 1
            write_json(Path(output) / "manifest.json", record)
        record.update(status="complete", completed_at=now())
    except BaseException:
        record.update(status="failed", completed_at=now(), error=traceback.format_exc())
        raise
    finally:
        if recorder:
            recorder.close()
        record["output_sha256"] = {name: file_hash(Path(output) / name)
                                   for name in ("episodes.csv", "steps.csv", "agents.csv")
                                   if (Path(output) / name).is_file()}
        write_json(Path(output) / "manifest.json", record)
    return record


def checkpoint_lineage(training_run):
    """Return all immutable snapshots across a chain of interrupted attempts."""
    found, visited = {}, set()
    current = Path(training_run).resolve()
    while True:
        if current in visited:
            raise ValueError("cyclic checkpoint lineage")
        visited.add(current)
        with (current / "manifest.json").open(encoding="utf-8") as handle:
            record = json.load(handle)
        for path in sorted((current / "checkpoints").glob("step_*")):
            with (path / "manifest.json").open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            step = metadata["environment_steps"]
            if step in found and found[step][1]["actor_sha256"] != metadata["actor_sha256"]:
                raise ValueError("conflicting checkpoints at the same training step")
            found.setdefault(step, (path, metadata))
        if not record.get("resumed_from"):
            break
        current = Path(record["resumed_from"]).parent.parent
    return [found[key] for key in sorted(found)]


def validate(training_run, tasks_file, output):
    from .mappo import ActorController
    output = Path(output)
    with (Path(training_run) / "manifest.json").open(encoding="utf-8") as handle:
        training = json.load(handle)
    spec = training["configuration"]
    tasks = read_tasks(tasks_file, spec, "validation")
    snapshots = checkpoint_lineage(training_run)
    record = new_run(output, spec, "validation", training_run=str(Path(training_run).resolve()),
                     training_seed=training["training_seed"], checkpoint_count=len(snapshots), evaluations=[])
    try:
        for checkpoint, metadata in snapshots:
            destination = output / checkpoint.name
            controller = ActorController(checkpoint, spec)
            evaluate(spec, tasks, destination, controller, metadata["training_seed"], controller.checkpoint_hash)
            record["evaluations"].append({"directory": str(destination.resolve()),
                "checkpoint": str(checkpoint.resolve()), "environment_steps": metadata["environment_steps"],
                "actor_sha256": metadata["actor_sha256"]})
            write_json(output / "manifest.json", record)
        record.update(status="complete", completed_at=now())
    except BaseException:
        record.update(status="failed", completed_at=now(), error=traceback.format_exc())
        raise
    finally:
        write_json(output / "manifest.json", record)
    return record
