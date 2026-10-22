"""Thin MAPPO-specific adapters; the evaluator loads only the actor."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

from .config import OBSERVATIONS, SEMANTICS, digest, file_hash
from .session import Session


def arguments(spec):
    args = get_config().parse_args([])
    values = spec["training"]
    overrides = dict(env_name="GridEnv", algorithm_name="mappo", use_recurrent_policy=False,
        use_naive_recurrent_policy=False, use_single_network=False, use_policy_vhead=False,
        use_influence_policy=False, share_policy=True, use_centralized_V=True,
        use_wandb=False, episode_length=values["rollout_length"], n_rollout_threads=values["num_envs"],
        hidden_size=values["hidden_size"], lr=values["learning_rate"], critic_lr=values["learning_rate"],
        ppo_epoch=values["ppo_epochs"], num_mini_batch=values["minibatches"], clip_param=values["clip"],
        gamma=values["gamma"], gae_lambda=values["gae_lambda"], entropy_coef=values["entropy_coefficient"],
        max_grad_norm=values["max_grad_norm"], motion_entropy_weight=1.0, communication_entropy_weight=1.0,
        use_valuenorm=True, use_popart=False, use_proper_time_limits=False)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def stack_observations(observations, index):
    return {key: np.stack([item[index][key] for item in observations])
            for key in observations[0][index]}


def flatten(observation):
    return {key: value.reshape(-1, *value.shape[2:]) for key, value in observation.items()}


def requests(actions):
    coordinates = 1.0 / (1.0 + np.exp(-np.clip(actions[..., :2], -50, 50)))
    return [{"coordinates": coordinate, "comm_actions": action[..., 2].astype(np.int64)}
            for coordinate, action in zip(coordinates, actions)]


class ActorController:
    name = "mappo"

    def __init__(self, checkpoint, spec, device="cpu"):
        checkpoint = Path(checkpoint)
        with (checkpoint / "manifest.json").open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if (self.metadata["environment_semantics"] != SEMANTICS
                or self.metadata["observation_version"] != OBSERVATIONS
                or self.metadata["configuration_sha256"] != digest(spec)):
            raise ValueError("checkpoint/environment contract mismatch")
        self.checkpoint_hash = file_hash(checkpoint / "actor.pt")
        if self.checkpoint_hash != self.metadata["actor_sha256"]:
            raise ValueError("actor checkpoint hash mismatch")
        torch.set_num_threads(1)
        self.args = SimpleNamespace(**self.metadata["policy_arguments"])
        env = Session(spec)
        self.actor = R_Actor(self.args, env.observation_space[0], env.action_space[0], torch.device(device))
        self.actor.load_state_dict(torch.load(str(checkpoint / "actor.pt"), map_location=device, weights_only=True))
        self.actor.eval()
        self.count = spec["num_agents"]

    def reset(self, task):
        pass

    def act(self, observation, state=None, available_actions=None, deterministic=True):
        if state is None:
            state = np.zeros((self.count, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        with torch.no_grad():
            action, _, next_state = self.actor(observation, state, np.ones((self.count, 1), np.float32),
                                               available_actions, deterministic=deterministic)
        return requests(action.cpu().numpy()[None])[0], next_state.cpu().numpy()
