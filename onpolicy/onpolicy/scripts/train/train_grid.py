#!/usr/bin/env python
import sys
import os
# import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config

from onpolicy.envs.GridEnv.GridEnv import GridEnv
from onpolicy.envs.GridEnv.communication import (
    COMMUNICATION_MODES,
    CommunicationConfig,
)
from onpolicy.envs.GridEnv.sensors import sensor_configs_from_values
from onpolicy.envs.env_wrappers import InfoSubprocVecEnv, InfoDummyVecEnv, ChooseInfoSubprocVecEnv, ChooseInfoDummyVecEnv


def make_communication_config(all_args):
    """Translate CLI arguments into one validated broker configuration.

    Returning ``None`` is intentional: it selects the untouched legacy action,
    observation, and ground-truth planning path rather than a silent broker.
    """
    if all_args.communication_mode is None:
        return None
    return CommunicationConfig(
        mode=all_args.communication_mode,
        tile_size=all_args.comm_tile_size,
        candidate_count=all_args.comm_candidate_count,
        radio_range_cells=all_args.comm_range_cells,
        latency_min_steps=all_args.comm_latency_min_steps,
        latency_max_steps=all_args.comm_latency_max_steps,
        packet_loss=all_args.comm_packet_loss,
        cooldown_steps=all_args.comm_cooldown_steps,
        bucket_capacity_bits=all_args.comm_bucket_capacity_bits,
        bucket_refill_bits=all_args.comm_bucket_refill_bits,
        episode_budget_bits=all_args.comm_episode_budget_bits,
        ttl_steps=all_args.comm_ttl_steps,
        timestamp_bits=all_args.comm_timestamp_bits,
        cost_per_patch=all_args.comm_cost_per_patch,
    )


def make_grid_env(all_args, sensor_configs, visualization=False):
    """Construct legacy or communication-aware GridEnv from parsed arguments."""
    communication_config = make_communication_config(all_args)
    if communication_config is None:
        return GridEnv(
            0.1,
            3,
            all_args.num_agents,
            100,
            visualization=visualization,
            sensor_configs=sensor_configs,
        )
    return GridEnv(
        0.1,
        3,
        all_args.num_agents,
        all_args.max_steps,
        use_merge=all_args.use_merge,
        use_same_location=all_args.use_same_location,
        use_complete_reward=all_args.use_complete_reward,
        use_multiroom=all_args.use_multiroom,
        use_time_penalty=all_args.use_time_penalty,
        use_single_reward=all_args.use_single_reward,
        visualization=visualization,
        sensor_configs=sensor_configs,
        communication_config=communication_config,
    )


def make_train_env(all_args):
    """Create deterministically seeded vector environments for training."""
    sensor_configs = sensor_configs_from_values(
        all_args.sensor_types, all_args.sensor_ranges, all_args.num_agents, 3.0
    )
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GridEnv":
                env = make_grid_env(all_args, sensor_configs)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return InfoDummyVecEnv([get_env_fn(0)])
    else:
        return InfoSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """Create separately seeded vector environments for policy evaluation."""
    sensor_configs = sensor_configs_from_values(
        all_args.sensor_types, all_args.sensor_ranges, all_args.num_agents, 3.0
    )
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "GridEnv":
                env = make_grid_env(all_args, sensor_configs, visualization=True)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return InfoDummyVecEnv([get_env_fn(0)])
    else:
        return InfoSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='simple_spread', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")
    parser.add_argument('--num_obstacles', type=int, default=1, help="number of players")
    parser.add_argument('--agent_pos', type=list, default = None, help="agent_pos")
    parser.add_argument('--grid_size', type=int, default=19, help="map size")
    parser.add_argument('--agent_view_size', type=int, default=7, help="depth the agent can view")
    parser.add_argument('--max_steps', type=int, default=100, help="depth the agent can view")
    parser.add_argument('--sensor_types', nargs='+', default=['omnidirectional'],
                        choices=['omnidirectional', 'four_beam'],
                        help="one sensor type or one type per agent")
    parser.add_argument('--sensor_ranges', nargs='+', type=float, default=None,
                        help="one maximum range or one range per agent; accepts inf")
    parser.add_argument('--communication_mode', choices=COMMUNICATION_MODES,
                        default=None,
                        help="enable the Level-0 communication model")
    parser.add_argument('--comm_tile_size', type=int, default=8)
    parser.add_argument('--comm_candidate_count', type=int, default=8)
    parser.add_argument('--comm_range_cells', type=float, default=40.0)
    parser.add_argument('--comm_latency_min_steps', type=int, default=1)
    parser.add_argument('--comm_latency_max_steps', type=int, default=None)
    parser.add_argument('--comm_packet_loss', type=float, default=0.0)
    parser.add_argument('--comm_cooldown_steps', type=int, default=3)
    parser.add_argument('--comm_bucket_capacity_bits', type=int, default=314)
    parser.add_argument('--comm_bucket_refill_bits', type=int, default=53)
    parser.add_argument('--comm_episode_budget_bits', type=int, default=None)
    parser.add_argument('--comm_ttl_steps', type=int, default=8)
    parser.add_argument('--comm_timestamp_bits', type=int, default=16)
    parser.add_argument('--comm_cost_per_patch', type=float, default=0.01)
    parser.add_argument('--local_step_num', type=int, default=3, help="local_goal_step")
    parser.add_argument("--use_same_location", action='store_true', default=False,
                        help="use merge information")
    parser.add_argument("--use_single_reward", action='store_true', default=False,
                        help="use single reward")
    parser.add_argument("--use_complete_reward", action='store_true', default=False,
                        help="use complete reward")            
    parser.add_argument("--use_merge", action='store_true', default=False,
                        help="use merge information")
    parser.add_argument("--use_multiroom", action='store_true', default=False,
                        help="use multiroom")
    parser.add_argument("--use_random_pos", action='store_true', default=False,
                        help="use complete reward")   
    parser.add_argument("--use_time_penalty", action='store_true', default=False,
                        help="use time penalty")    
    parser.add_argument("--use_intrinsic_reward", action='store_true', default=False,
                        help="use intrinsic reward")             
    parser.add_argument("--visualize_input", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    ### debug for env
    # env = GazeboEnv("/home/nics/catkin_ws/small_room.pgm", 0.1, 3, 2, 100, visualization=True)
    # env.reset()
    # import pdb; pdb.set_trace()
    ###
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.communication_mode is not None and not all_args.share_policy:
        parser.error("communication training currently requires shared policies")
    if (
        all_args.communication_mode is not None
        and not all_args.use_centralized_V
    ):
        parser.error(
            "communication training requires the centralized critic"
        )

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents
    if all_args.communication_mode is not None:
        all_args.episode_length = all_args.max_steps
    else:
        all_args.episode_length = all_args.max_steps//all_args.local_step_num

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.grid_runner import GridRunner as Runner
    else:
        from onpolicy.runner.separated.grid_runner import GridRunner as Runner
    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
