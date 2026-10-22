"""MAPPO orchestration over reusable sessions, records and checkpoints."""

import random
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.shared_buffer import SharedReplayBuffer

from .artifacts import Table, new_run, now, write_json
from .config import file_hash
from .checkpoints import restore_checkpoint, restore_random, save_checkpoint
from .mappo import arguments, flatten, requests, stack_observations
from .recording import Recorder
from .session import Session
from .worlds import WorldBatch


UPDATE_FIELDS = ["environment_steps", "update", "elapsed_seconds", "value_loss", "policy_loss",
                 "dist_entropy", "actor_grad_norm", "critic_grad_norm", "ratio", "approximate_kl", "clip_fraction"]


def train(spec, output, seed, device="cpu", resume=None, stop_after_updates=None, allow_dirty=False):
    output = Path(output)
    if seed not in spec["training_seeds"]:
        raise ValueError("training seed must be declared in the experiment")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    record = new_run(output, spec, "training", training_seed=seed,
                     torch_version=torch.__version__, device=device,
                     device_name=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                     resumed_from=str(Path(resume).resolve()) if resume else None)
    worlds = recorder = updates = None
    previous_handlers = {}
    stopped = [False]
    steps = 0
    try:
        if record["repository_status"] and not allow_dirty:
            raise ValueError("tracked worktree is dirty; commit first or explicitly use --allow-dirty for development")
        torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        args = arguments(spec)
        args.seed = seed
        record["policy_arguments"] = vars(args).copy()
        template = Session(spec)
        policy = R_MAPPOPolicy(args, template.observation_space[0], template.share_observation_space[0],
                               template.action_space[0], device=torch.device(device))
        trainer = R_MAPPO(args, policy, device=torch.device(device))
        buffer = SharedReplayBuffer(args, spec["num_agents"], template.observation_space[0],
                                    template.share_observation_space[0], template.action_space[0])
        state = None
        if resume:
            state = restore_checkpoint(resume, spec, policy, trainer, record["source_hashes"], device)
            if state["training_seed"] != seed:
                raise ValueError("resume training seed mismatch")
            steps = state["steps"]
        worlds = WorldBatch(spec, seed, state["worlds"] if state else None)
        if state:
            restore_random(state["random_state"])
        recorder = Recorder(output, spec["thresholds"], full=False)
        updates = Table(output / "updates.csv", UPDATE_FIELDS)
        write_json(output / "resolved_arguments.json", vars(args))
        for key, value in stack_observations(worlds.observations, 0).items():
            buffer.obs[key][0] = value
        for key, value in stack_observations(worlds.observations, 1).items():
            buffer.share_obs[key][0] = value
        buffer.available_actions[0] = np.stack([observation[2] for observation in worlds.observations])
        count = spec["training"]["num_envs"]
        agents = spec["num_agents"]
        zeros = np.zeros((count * agents, args.recurrent_N, args.hidden_size), np.float32)
        started = time.monotonic()
        initial_steps = steps
        last_checkpoint_steps = None
        if not resume:
            save_checkpoint(output, spec, seed, steps, policy, trainer, worlds, record)
            last_checkpoint_steps = steps

        def request_stop(signum, frame):
            stopped[0] = True
            record["stop_reason"] = "signal_{}".format(signum)

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        rollout_size = count * args.episode_length
        while steps < spec["training"]["total_steps"]:
            trainer.prep_rollout()
            for step in range(args.episode_length):
                with torch.no_grad():
                    value, action, log_probability, rnn, critic_rnn = policy.get_actions(
                        flatten({key: array[step] for key, array in buffer.share_obs.items()}),
                        flatten({key: array[step] for key, array in buffer.obs.items()}),
                        zeros, zeros, buffer.masks[step].reshape(-1, 1),
                        buffer.available_actions[step].reshape(-1, buffer.available_actions.shape[-1]))
                actions = action.cpu().numpy().reshape(count, agents, -1)
                results = worlds.step(requests(actions))
                rewards = np.stack([item["reward"] for item in results])
                dones = np.asarray([item["terminated"] or item["truncated"] for item in results])
                # The external episode cap cuts the trace but is not a terminal
                # state. Bootstrap from the FINAL state, never the reset state.
                truncated_indices = [i for i, item in enumerate(results) if item["truncated"]]
                if truncated_indices:
                    finals = [results[i]["final_observation"] for i in truncated_indices]
                    with torch.no_grad():
                        final_values = policy.get_values(flatten(stack_observations(finals, 1)),
                            zeros[:len(finals) * agents], np.ones((len(finals) * agents, 1), np.float32))
                    final_values = trainer.value_normalizer.denormalize(final_values).reshape(len(finals), agents, 1)
                    for index, final_value in zip(truncated_indices, final_values):
                        rewards[index] += args.gamma * final_value
                masks = np.ones((count, agents, 1), np.float32)
                masks[dones] = 0
                buffer.insert(stack_observations(worlds.observations, 1),
                              stack_observations(worlds.observations, 0),
                              zeros.reshape(count, agents, args.recurrent_N, args.hidden_size),
                              zeros.reshape(count, agents, args.recurrent_N, args.hidden_size),
                              actions, log_probability.cpu().numpy().reshape(count, agents, 1),
                              value.cpu().numpy().reshape(count, agents, 1), rewards, masks,
                              available_actions=np.stack([item[2] for item in worlds.observations]))
                for item in results:
                    for episode in item["episodes"]:
                        recorder.episodes.append(episode)
            with torch.no_grad():
                next_values = policy.get_values(flatten({key: array[-1] for key, array in buffer.share_obs.items()}),
                                                zeros, buffer.masks[-1].reshape(-1, 1))
            buffer.compute_returns(next_values.cpu().numpy().reshape(count, agents, 1), trainer.value_normalizer)
            trainer.prep_training()
            statistics = trainer.train(buffer)
            if not all(np.isfinite(float(value)) for value in statistics.values()):
                raise FloatingPointError("non-finite MAPPO update")
            buffer.after_update()
            steps += rollout_size
            update = steps // rollout_size
            updates.append(dict(environment_steps=steps, update=update,
                                elapsed_seconds=time.monotonic() - started,
                                **{key: float(value) for key, value in statistics.items()}))
            print("update={} environment_steps={} policy_loss={:.6g} value_loss={:.6g}".format(
                update, steps, statistics["policy_loss"], statistics["value_loss"]), flush=True)
            if stop_after_updates is not None and steps - initial_steps >= stop_after_updates * rollout_size:
                stopped[0] = True
                record.setdefault("stop_reason", "requested_update_limit")
            if (steps % spec["training"]["checkpoint_interval"] == 0
                    or steps >= spec["training"]["total_steps"] or stopped[0]):
                checkpoint = save_checkpoint(output, spec, seed, steps, policy, trainer, worlds, record)
                last_checkpoint_steps = steps
                record["latest_checkpoint"] = str(checkpoint.resolve())
                record["environment_steps"] = steps
                write_json(output / "manifest.json", record)
            if stopped[0]:
                break
        record.update(status="complete" if steps >= spec["training"]["total_steps"] else "interrupted",
                      environment_steps=steps, completed_at=now())
    except BaseException:
        record.update(status="failed", environment_steps=steps, completed_at=now(),
                      error=traceback.format_exc())
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if worlds:
            worlds.close()
        if recorder:
            recorder.close()
        if updates:
            updates.close()
        record["output_sha256"] = {name: file_hash(output / name) for name in ("episodes.csv", "updates.csv")
                                   if (output / name).is_file()}
        write_json(output / "manifest.json", record)
    return record
