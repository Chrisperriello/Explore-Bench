"""Shared actor/critic preprocessing for legacy runners and research sessions."""

import cv2
import numpy as np


def build_communication_observations(context, dict_obs, infos, evaluation=False):
    """Build separate decentralized-actor and centralized-critic inputs.

    Actor observations contain the agent's private sensing, delivered
    belief, last delivered peer poses with freshness, candidate-patch
    features, local resource state, and identity.  They never contain true
    unseen peer state.

    Shared critic observations additionally contain team exploration,
    current team positions/history, every agent belief, and global broker
    progress.  This explicit conversion is the CTDE trust boundary.

    Args:
        dict_obs: Raw map observations returned by vector environments.
        infos: Privileged environment information, one dictionary per env.
        evaluation: Select evaluation rather than training position history.

    Returns:
        ``(raw_obs, obs, share_obs, available_actions)`` where the last
        tensor masks only the categorical communication head.
    """
    env_count = len(dict_obs)
    map_shape = (env_count, context.num_agents, 4, context.full_w, context.full_h)
    input_shape = (env_count, context.num_agents, 4, context.input_w, context.input_h)
    raw_obs = {
        'global_obs': np.zeros(map_shape, dtype=np.float32),
        'global_merge_obs': np.zeros(map_shape, dtype=np.float32),
        'vector': np.zeros(
            (env_count, context.num_agents, context.num_agents), dtype=np.float32
        ),
        'comm_candidates': np.zeros(
            (
                env_count,
                context.num_agents,
                context.all_args.comm_candidate_count,
                5,
            ),
            dtype=np.float32,
        ),
        'comm_status': np.zeros(
            (env_count, context.num_agents, 8), dtype=np.float32
        ),
    }
    obs = {
        'global_obs': np.zeros(input_shape, dtype=np.float32),
        'global_merge_obs': np.zeros(input_shape, dtype=np.float32),
        'vector': np.zeros_like(raw_obs['vector']),
        'comm_candidates': np.zeros_like(raw_obs['comm_candidates']),
        'comm_status': np.zeros_like(raw_obs['comm_status']),
    }
    share_obs = {
        key: np.zeros_like(value) for key, value in obs.items()
    }
    share_obs['agent_beliefs'] = np.zeros(
        (
            env_count,
            context.num_agents,
            2 * context.num_agents,
            context.input_w,
            context.input_h,
        ),
        dtype=np.float32,
    )
    share_obs['global_comm_state'] = np.zeros(
        (env_count, context.num_agents, 3 * context.num_agents + 2),
        dtype=np.float32,
    )
    available_actions = np.zeros(
        (
            env_count,
            context.num_agents,
            context.all_args.comm_candidate_count + 2,
        ),
        dtype=np.float32,
    )
    own_histories = (
        context.eval_all_agent_pos_map if evaluation else context.all_agent_pos_map
    )
    team_histories = (
        context.eval_all_merge_pos_map if evaluation else context.all_merge_pos_map
    )

    for e in range(env_count):
        info = infos[e]
        current_positions = np.zeros((context.full_w, context.full_h), dtype=np.float32)
        for agent_id in range(context.num_agents):
            row, column = info['current_agent_pos'][agent_id]
            marker = (agent_id + 1) * context.augment
            own_histories[e, agent_id, row, column] = marker
            team_histories[e, row, column] = max(
                team_histories[e, row, column], marker
            )
            current_positions[row, column] = marker

        resized_beliefs = np.zeros(
            (2 * context.num_agents, context.input_w, context.input_h),
            dtype=np.float32,
        )
        for belief_agent in range(context.num_agents):
            belief = info['belief_each_map'][belief_agent]
            resized_beliefs[2 * belief_agent] = cv2.resize(
                (belief != 205).astype(np.uint8),
                (context.input_w, context.input_h),
            ).astype(np.float32)
            resized_beliefs[2 * belief_agent + 1] = cv2.resize(
                (belief == 0).astype(np.uint8),
                (context.input_w, context.input_h),
            ).astype(np.float32)

        global_comm_state = np.concatenate(
            [
                info['comm_status'][:, :3].reshape(-1),
                np.array(
                    [
                        min(
                            1.0,
                            info['comm_in_flight_count']
                            / float(max(1, context.num_agents)),
                        ),
                        info['comm_step'] / float(max(1, context.max_steps)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )

        for agent_id in range(context.num_agents):
            row, column = info['current_agent_pos'][agent_id]
            self_position = np.zeros(
                (context.full_w, context.full_h), dtype=np.float32
            )
            self_position[row, column] = (agent_id + 1) * context.augment
            belief = info['belief_each_map'][agent_id]
            peer_positions = np.zeros_like(self_position)
            peer_age = np.zeros_like(self_position)
            for sender in range(context.num_agents):
                if sender == agent_id:
                    continue
                peer_row, peer_column = info['peer_positions'][agent_id, sender]
                sent_step = info['peer_pose_steps'][agent_id, sender]
                if peer_row < 0 or sent_step < 0:
                    continue
                peer_positions[peer_row, peer_column] = (
                    sender + 1
                ) * context.augment
                age = max(0, info['comm_step'] - sent_step)
                peer_age[peer_row, peer_column] = max(
                    0.0,
                    1.0 - age / float(max(1, context.all_args.comm_ttl_steps)),
                )

            raw_obs['global_obs'][e, agent_id, 0] = info[
                'explored_each_map'
            ][agent_id]
            raw_obs['global_obs'][e, agent_id, 1] = info[
                'obstacle_each_map'
            ][agent_id]
            raw_obs['global_obs'][e, agent_id, 2] = self_position / 255.0
            raw_obs['global_obs'][e, agent_id, 3] = (
                own_histories[e, agent_id] / 255.0
            )
            raw_obs['global_merge_obs'][e, agent_id, 0] = belief != 205
            raw_obs['global_merge_obs'][e, agent_id, 1] = belief == 0
            raw_obs['global_merge_obs'][e, agent_id, 2] = (
                peer_positions / 255.0
            )
            raw_obs['global_merge_obs'][e, agent_id, 3] = peer_age
            raw_obs['vector'][e, agent_id] = np.eye(context.num_agents)[agent_id]
            raw_obs['comm_candidates'][e, agent_id] = info[
                'comm_candidate_features'
            ][agent_id]
            raw_obs['comm_status'][e, agent_id] = info['comm_status'][agent_id]
            available_actions[e, agent_id] = info['available_comm_actions'][
                agent_id
            ]

            for key in ('global_obs', 'global_merge_obs'):
                for channel in range(4):
                    obs[key][e, agent_id, channel] = cv2.resize(
                        raw_obs[key][e, agent_id, channel].astype(np.float32),
                        (context.input_w, context.input_h),
                    )
            obs['vector'][e, agent_id] = raw_obs['vector'][e, agent_id]
            obs['comm_candidates'][e, agent_id] = raw_obs[
                'comm_candidates'
            ][e, agent_id]
            obs['comm_status'][e, agent_id] = raw_obs['comm_status'][
                e, agent_id
            ]

            for key in ('global_obs', 'vector', 'comm_candidates', 'comm_status'):
                share_obs[key][e, agent_id] = obs[key][e, agent_id]
            share_obs['global_merge_obs'][e, agent_id, 0] = cv2.resize(
                info['explored_all_map'].astype(np.uint8),
                (context.input_w, context.input_h),
            ).astype(np.float32)
            share_obs['global_merge_obs'][e, agent_id, 1] = cv2.resize(
                info['obstacle_all_map'].astype(np.uint8),
                (context.input_w, context.input_h),
            ).astype(np.float32)
            share_obs['global_merge_obs'][e, agent_id, 2] = cv2.resize(
                (current_positions > 0).astype(np.uint8),
                (context.input_w, context.input_h),
            ).astype(np.float32)
            share_obs['global_merge_obs'][e, agent_id, 3] = cv2.resize(
                (team_histories[e] > 0).astype(np.uint8),
                (context.input_w, context.input_h),
            ).astype(np.float32)
            share_obs['agent_beliefs'][e, agent_id] = resized_beliefs
            share_obs['global_comm_state'][e, agent_id] = global_comm_state

    return raw_obs, obs, share_obs, available_actions
