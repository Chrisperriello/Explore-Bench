"""Immutable training checkpoints and independent actor-only inference exports."""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .artifacts import now, write_json
from .config import OBSERVATIONS, SEMANTICS, digest, file_hash


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(run, spec, seed, steps, policy, trainer, worlds, provenance_record):
    parent = Path(run) / "checkpoints"
    parent.mkdir(exist_ok=True)
    target = parent / "step_{:012d}".format(steps)
    if target.exists():
        raise FileExistsError("checkpoint is immutable: " + str(target))
    staging = Path(tempfile.mkdtemp(prefix=".pending_", dir=str(parent)))
    torch.save(policy.actor.state_dict(), str(staging / "actor.pt"))
    state = {"actor": policy.actor.state_dict(), "critic": policy.critic.state_dict(),
             "actor_optimizer": policy.actor_optimizer.state_dict(),
             "critic_optimizer": policy.critic_optimizer.state_dict(),
             "value_normalizer": trainer.value_normalizer.state_dict(),
             "worlds": worlds.snapshot(), "steps": steps, "random_state": random_state(),
             "configuration": spec, "training_seed": seed,
             "source_hashes": provenance_record["source_hashes"]}
    torch.save(state, str(staging / "training.pt"))
    write_json(staging / "manifest.json", {
        "status": "complete", "created_at": now(), "environment_steps": steps,
        "training_seed": seed, "configuration": spec, "configuration_sha256": digest(spec),
        "environment_semantics": SEMANTICS, "observation_version": OBSERVATIONS,
        "actor_sha256": file_hash(staging / "actor.pt"),
        "training_sha256": file_hash(staging / "training.pt"),
        "training_run": str(Path(run).resolve()),
        "source_hashes": provenance_record["source_hashes"],
        "policy_arguments": provenance_record["policy_arguments"],
    })
    os.rename(str(staging), str(target))
    return target


def restore_checkpoint(path, spec, policy, trainer, source_hashes, device):
    import json
    path = Path(path)
    with (path / "manifest.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata["training_sha256"] != file_hash(path / "training.pt"):
        raise ValueError("training checkpoint hash mismatch")
    # Training state contains trusted local Python session objects. Never load
    # a training.pt received from an untrusted source.
    state = torch.load(str(path / "training.pt"), map_location=device, weights_only=False)
    if state["configuration"] != spec or state["source_hashes"] != source_hashes:
        raise ValueError("resume requires the same configuration and source hashes")
    policy.actor.load_state_dict(state["actor"])
    policy.critic.load_state_dict(state["critic"])
    policy.actor_optimizer.load_state_dict(state["actor_optimizer"])
    policy.critic_optimizer.load_state_dict(state["critic_optimizer"])
    trainer.value_normalizer.load_state_dict(state["value_normalizer"])
    return state
