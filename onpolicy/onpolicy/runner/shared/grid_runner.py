    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio
from icecream import ic
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict, deque
from onpolicy.utils.util import update_linear_schedule, get_shape_from_act_space
from onpolicy.runner.shared.base_runner import Runner
import torch.nn as nn

def _t2n(x):
    return x.detach().cpu().numpy()

class GridRunner(Runner):
    def __init__(self, config):
        super(GridRunner, self).__init__(config)
        self.init_hyperparameters()
        self.init_map_variables() 

    def run(self):
        self.warmup()   

        episodes = int(self.num_env_steps) // self.max_steps // self.n_rollout_threads
        start = time.time()
        for episode in range(episodes):
            self.env_infos = defaultdict(list)

            self.init_map_variables() 
            global_step = 0
            local_step = 0
            # period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.max_steps):        
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(global_step)
                global_step += 1
                inputs = [{} for _ in range(self.n_rollout_threads)]
                for e in range(self.n_rollout_threads):
                    inputs[e]['global_goal'] = self.short_term_goal[e]
                    # if self.use_merge:
                    #     inputs[e]['global_obs'] = self.obs['global_merge_obs'][e]
                    # else:
                    #     inputs[e]['global_obs'] = self.obs['global_obs'][e]
                    # we only want the robot to go its own frontiers
                    inputs[e]['global_obs'] = (
                        self.obs['global_merge_obs'][e]
                        if self.communication_enabled
                        else self.obs['global_obs'][e]
                    )
                actions_env = self.envs.get_short_term_goal(inputs)
                if self.communication_enabled:
                    actions_env = [
                        {
                            'motion_goals': actions_env[e],
                            'comm_actions': self.communication_actions[e],
                        }
                        for e in range(self.n_rollout_threads)
                    ]
                
                dict_obs, rewards, dones, infos = self.envs.step(actions_env)

                data = dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                # insert data into buffer
                self.insert(data)
                print(step)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                
                #print(self.buffer.rewards)
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length             
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                print("average episode ratio is {}".format(np.mean(self.env_infos["merge_explored_ratio"])))
                                
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(0)

    def _eval_convert(self, dict_obs, infos):
        obs = {}
        obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h, 3), dtype=np.float32)
        obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
            #obs['global_merge_goal'] = np.zeros((len(dict_obs), self.num_agents, 2, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        else:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
            #global_merge_goal = np.zeros((len(dict_obs), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        
        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                self.eval_all_agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                if self.use_merge:
                    '''a = int(self.short_term_goal[e][agent_id][0])
                    b = int(self.short_term_goal[e][agent_id][1])
                   
                    if eval_global_merge_goal[e, a, b] != (agent_id + 1) * self.augment and\
                    eval_global_merge_goal[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        eval_global_merge_goal[e, a, b] += (agent_id + 1) * self.augment
                    
                    if self.eval_global_merge_goal_trace[e, a, b] != (agent_id + 1) * self.augment and\
                    self.eval_global_merge_goal_trace[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.eval_global_merge_goal_trace[e, a, b] += (agent_id + 1) * self.augment'''
                    
                    merge_pos_map[e , infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment
                    if self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != (agent_id + 1) * self.augment and\
                    self.eval_all_merge_pos_map[e,  infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment

        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id]
                obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id]
                obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id] / 255.0
                obs['global_obs'][e, agent_id, 3] = self.eval_all_agent_pos_map[e, agent_id] / 255.0
                #self.agent_local_map[e, agent_id] = infos[e]['agent_local_map'][agent_id]
                #obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    #obs['global_merge_goal'][e, agent_id, 0] = global_merge_goal[e]
                    #obs['global_merge_goal'][e, agent_id, 1] = self.global_merge_goal_trace[e]
                    obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map']
                    obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map']
                    obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e] / 255.0
                    obs['global_merge_obs'][e, agent_id, 3] = self.eval_all_merge_pos_map[e] / 255.0

                obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]

                i = 0
                obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
                if self.use_merge:
                    for l in range(self.num_agents):
                        if l!= agent_id: 
                            i += 1 
                            obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]                  

        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return obs

    def _convert(self, dict_obs, infos):
        obs = {}
        obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h, 3), dtype=np.float32)
        obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
            #obs['global_merge_goal'] = np.zeros((len(dict_obs), self.num_agents, 2, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        else:
            obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
            #global_merge_goal = np.zeros((len(dict_obs), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        
        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                self.all_agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                if self.use_merge:
                    merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment
                    
                    '''a = int(self.short_term_goal[e][agent_id][0])
                    b = int(self.short_term_goal[e][agent_id][1])
                    if global_merge_goal[e, a, b] != (agent_id + 1) * self.augment and\
                    global_merge_goal[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        global_merge_goal[e, a, b] += (agent_id + 1) * self.augment
                    
                    if self.global_merge_goal_trace[e, a, b] != (agent_id + 1) * self.augment and\
                    self.global_merge_goal_trace[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.global_merge_goal_trace[e, a, b] += (agent_id + 1) * self.augment'''
                    
                    if self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != (agent_id + 1) * self.augment and\
                    self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment

        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id]
                obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id]
                obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id] / 255.0
                obs['global_obs'][e, agent_id, 3] = self.all_agent_pos_map[e, agent_id] / 255.0
                #self.agent_local_map[e, agent_id] = infos[e]['agent_local_map'][agent_id]
                #obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    #obs['global_merge_goal'][e, agent_id, 0] = global_merge_goal[e]
                    #obs['global_merge_goal'][e, agent_id, 1] = self.global_merge_goal_trace[e]
                    obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map']
                    obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map']
                    obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e] / 255.0
                    obs['global_merge_obs'][e, agent_id, 3] = self.all_merge_pos_map[e] / 255.0

                obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]

                i = 0
                obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
                if self.use_merge:
                    for l in range(self.num_agents):
                        if l!= agent_id: 
                            i += 1 
                            obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]               
        
        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return obs

    def _resize_eval_convert(self, dict_obs, infos):
        raw_obs = {}
        obs = {}

        # raw_obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h, 3), dtype=np.float32)
        # raw_obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        raw_obs['vector'] = np.zeros((len(dict_obs), self.num_agents, 2), dtype=np.float32)
        raw_obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)

        # obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.input_w, self.input_h, 3), dtype=np.float32)
        # obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        obs['vector'] = np.zeros((len(dict_obs), self.num_agents, 2), dtype=np.float32)
        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.input_w, self.input_h), dtype=np.float32)

        # if self.use_merge:
        #     raw_obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
        #     obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
        #     #obs['global_merge_goal'] = np.zeros((len(dict_obs), self.num_agents, 2, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        # else:
        #     raw_obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        #     obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            raw_obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.input_w, self.input_h), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
            #global_merge_goal = np.zeros((len(dict_obs), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        
        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                self.eval_all_agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                if self.use_merge:
                    merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment
                    
                    '''a = int(self.short_term_goal[e][agent_id][0])
                    b = int(self.short_term_goal[e][agent_id][1])
                    if global_merge_goal[e, a, b] != (agent_id + 1) * self.augment and\
                    global_merge_goal[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        global_merge_goal[e, a, b] += (agent_id + 1) * self.augment
                    
                    if self.global_merge_goal_trace[e, a, b] != (agent_id + 1) * self.augment and\
                    self.global_merge_goal_trace[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.global_merge_goal_trace[e, a, b] += (agent_id + 1) * self.augment'''
                    
                    if self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != (agent_id + 1) * self.augment and\
                    self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment

        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                raw_obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id]
                raw_obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id]
                raw_obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id] / 255.0
                raw_obs['global_obs'][e, agent_id, 3] = self.eval_all_agent_pos_map[e, agent_id] / 255.0

                obs['global_obs'][e, agent_id, 0] = cv2.resize(infos[e]['explored_each_map'][agent_id].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                obs['global_obs'][e, agent_id, 1] = cv2.resize(infos[e]['obstacle_each_map'][agent_id].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                obs['global_obs'][e, agent_id, 2] = cv2.resize((agent_pos_map[e, agent_id] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                obs['global_obs'][e, agent_id, 3] = cv2.resize((self.all_agent_pos_map[e, agent_id] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                #self.agent_local_map[e, agent_id] = infos[e]['agent_local_map'][agent_id]
                #obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    #obs['global_merge_goal'][e, agent_id, 0] = global_merge_goal[e]
                    #obs['global_merge_goal'][e, agent_id, 1] = self.global_merge_goal_trace[e]
                    raw_obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map']
                    raw_obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map']
                    raw_obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e] / 255.0
                    raw_obs['global_merge_obs'][e, agent_id, 3] = self.eval_all_merge_pos_map[e] / 255.0

                    obs['global_merge_obs'][e, agent_id, 0] = cv2.resize(infos[e]['explored_all_map'].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    obs['global_merge_obs'][e, agent_id, 1] = cv2.resize(infos[e]['obstacle_all_map'].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    obs['global_merge_obs'][e, agent_id, 2] = cv2.resize((merge_pos_map[e] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    obs['global_merge_obs'][e, agent_id, 3] = cv2.resize((self.all_merge_pos_map[e] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    
                    # obs['global_merge_obs'][e, agent_id, 0] = cv2.resize(infos[e]['explored_all_map'].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    # obs['global_merge_obs'][e, agent_id, 1] = cv2.resize(infos[e]['obstacle_all_map'].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    # obs['global_merge_obs'][e, agent_id, 2] = cv2.resize((merge_pos_map[e] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    # obs['global_merge_obs'][e, agent_id, 3] = cv2.resize((self.eval_all_merge_pos_map[e] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                # raw_obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]
                # obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]
                raw_obs['vector'][0, agent_id] = np.eye(2)[0]
                obs['vector'][0, agent_id] = np.eye(2)[0] 

        #         i = 0
        #         raw_obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
        #         obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
        #         if self.use_merge:
        #             for l in range(self.num_agents):
        #                 if l!= agent_id: 
        #                     i += 1 
        #                     raw_obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]    
        #                     obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]               
        # import pdb; pdb.set_trace()
        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return raw_obs, obs

    def _resize_convert(self, dict_obs, infos):
        raw_obs = {}
        obs = {}

        #raw_obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h, 3), dtype=np.float32)
        raw_obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        raw_obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)

        #obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.input_w, self.input_h, 3), dtype=np.float32)
        obs['vector'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents), dtype=np.float32)
        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.input_w, self.input_h), dtype=np.float32)

        # if self.use_merge:
        #     raw_obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
        #     obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
        #     #obs['global_merge_goal'] = np.zeros((len(dict_obs), self.num_agents, 2, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        # else:
        #     raw_obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        #     obs['global_direction'] = np.zeros((len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            raw_obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w, self.full_h), dtype=np.float32)
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.input_w, self.input_h), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
            #global_merge_goal = np.zeros((len(dict_obs), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        
        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                agent_pos_map[e , agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                self.all_agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment
                if self.use_merge:
                    merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment
                    
                    '''a = int(self.short_term_goal[e][agent_id][0])
                    b = int(self.short_term_goal[e][agent_id][1])
                    if global_merge_goal[e, a, b] != (agent_id + 1) * self.augment and\
                    global_merge_goal[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        global_merge_goal[e, a, b] += (agent_id + 1) * self.augment
                    
                    if self.global_merge_goal_trace[e, a, b] != (agent_id + 1) * self.augment and\
                    self.global_merge_goal_trace[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.global_merge_goal_trace[e, a, b] += (agent_id + 1) * self.augment'''
                    
                    if self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != (agent_id + 1) * self.augment and\
                    self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment

        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                raw_obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id]
                raw_obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id]
                raw_obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id] / 255.0
                raw_obs['global_obs'][e, agent_id, 3] = self.all_agent_pos_map[e, agent_id] / 255.0

                obs['global_obs'][e, agent_id, 0] = cv2.resize(infos[e]['explored_each_map'][agent_id].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                obs['global_obs'][e, agent_id, 1] = cv2.resize(infos[e]['obstacle_each_map'][agent_id].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                obs['global_obs'][e, agent_id, 2] = cv2.resize((agent_pos_map[e, agent_id] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                obs['global_obs'][e, agent_id, 3] = cv2.resize((self.all_agent_pos_map[e, agent_id] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                #self.agent_local_map[e, agent_id] = infos[e]['agent_local_map'][agent_id]
                #obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    #obs['global_merge_goal'][e, agent_id, 0] = global_merge_goal[e]
                    #obs['global_merge_goal'][e, agent_id, 1] = self.global_merge_goal_trace[e]
                    raw_obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map']
                    raw_obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map']
                    raw_obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e] / 255.0
                    raw_obs['global_merge_obs'][e, agent_id, 3] = self.all_merge_pos_map[e] / 255.0

                    obs['global_merge_obs'][e, agent_id, 0] = cv2.resize(infos[e]['explored_all_map'].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    obs['global_merge_obs'][e, agent_id, 1] = cv2.resize(infos[e]['obstacle_all_map'].astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    obs['global_merge_obs'][e, agent_id, 2] = cv2.resize((merge_pos_map[e] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)
                    obs['global_merge_obs'][e, agent_id, 3] = cv2.resize((self.all_merge_pos_map[e] > 0).astype(np.uint8), (self.input_w, self.input_h)).astype(np.float32)

                raw_obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]
                obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]

                # i = 0
                # raw_obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
                # obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][agent_id]]
                # if self.use_merge:
                #     for l in range(self.num_agents):
                #         if l!= agent_id: 
                #             i += 1 
                #             raw_obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]    
                #             obs['global_direction'][e, agent_id, i] = np.eye(4)[infos[e]['agent_direction'][l]]               
        
        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return raw_obs, obs

    def _resize_communication_convert(self, dict_obs, infos, evaluation=False):
        env_count = len(dict_obs)
        map_shape = (env_count, self.num_agents, 4, self.full_w, self.full_h)
        input_shape = (env_count, self.num_agents, 4, self.input_w, self.input_h)
        raw_obs = {
            'global_obs': np.zeros(map_shape, dtype=np.float32),
            'global_merge_obs': np.zeros(map_shape, dtype=np.float32),
            'vector': np.zeros(
                (env_count, self.num_agents, self.num_agents), dtype=np.float32
            ),
            'comm_candidates': np.zeros(
                (
                    env_count,
                    self.num_agents,
                    self.all_args.comm_candidate_count,
                    5,
                ),
                dtype=np.float32,
            ),
            'comm_status': np.zeros(
                (env_count, self.num_agents, 8), dtype=np.float32
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
                self.num_agents,
                2 * self.num_agents,
                self.input_w,
                self.input_h,
            ),
            dtype=np.float32,
        )
        share_obs['global_comm_state'] = np.zeros(
            (env_count, self.num_agents, 3 * self.num_agents + 2),
            dtype=np.float32,
        )
        available_actions = np.zeros(
            (
                env_count,
                self.num_agents,
                self.all_args.comm_candidate_count + 2,
            ),
            dtype=np.float32,
        )
        own_histories = (
            self.eval_all_agent_pos_map if evaluation else self.all_agent_pos_map
        )
        team_histories = (
            self.eval_all_merge_pos_map if evaluation else self.all_merge_pos_map
        )

        for e in range(env_count):
            info = infos[e]
            current_positions = np.zeros((self.full_w, self.full_h), dtype=np.float32)
            for agent_id in range(self.num_agents):
                row, column = info['current_agent_pos'][agent_id]
                marker = (agent_id + 1) * self.augment
                own_histories[e, agent_id, row, column] = marker
                team_histories[e, row, column] = max(
                    team_histories[e, row, column], marker
                )
                current_positions[row, column] = marker

            resized_beliefs = np.zeros(
                (2 * self.num_agents, self.input_w, self.input_h),
                dtype=np.float32,
            )
            for belief_agent in range(self.num_agents):
                belief = info['belief_each_map'][belief_agent]
                resized_beliefs[2 * belief_agent] = cv2.resize(
                    (belief != 205).astype(np.uint8),
                    (self.input_w, self.input_h),
                ).astype(np.float32)
                resized_beliefs[2 * belief_agent + 1] = cv2.resize(
                    (belief == 0).astype(np.uint8),
                    (self.input_w, self.input_h),
                ).astype(np.float32)

            global_comm_state = np.concatenate(
                [
                    info['comm_status'][:, :3].reshape(-1),
                    np.array(
                        [
                            min(
                                1.0,
                                info['comm_in_flight_count']
                                / float(max(1, self.num_agents)),
                            ),
                            info['comm_step'] / float(max(1, self.max_steps)),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )

            for agent_id in range(self.num_agents):
                row, column = info['current_agent_pos'][agent_id]
                self_position = np.zeros(
                    (self.full_w, self.full_h), dtype=np.float32
                )
                self_position[row, column] = (agent_id + 1) * self.augment
                belief = info['belief_each_map'][agent_id]
                peer_positions = np.zeros_like(self_position)
                peer_age = np.zeros_like(self_position)
                for sender in range(self.num_agents):
                    if sender == agent_id:
                        continue
                    peer_row, peer_column = info['peer_positions'][agent_id, sender]
                    sent_step = info['peer_pose_steps'][agent_id, sender]
                    if peer_row < 0 or sent_step < 0:
                        continue
                    peer_positions[peer_row, peer_column] = (
                        sender + 1
                    ) * self.augment
                    age = max(0, info['comm_step'] - sent_step)
                    peer_age[peer_row, peer_column] = max(
                        0.0,
                        1.0 - age / float(max(1, self.all_args.comm_ttl_steps)),
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
                raw_obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]
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
                            (self.input_w, self.input_h),
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
                    (self.input_w, self.input_h),
                ).astype(np.float32)
                share_obs['global_merge_obs'][e, agent_id, 1] = cv2.resize(
                    info['obstacle_all_map'].astype(np.uint8),
                    (self.input_w, self.input_h),
                ).astype(np.float32)
                share_obs['global_merge_obs'][e, agent_id, 2] = cv2.resize(
                    (current_positions > 0).astype(np.uint8),
                    (self.input_w, self.input_h),
                ).astype(np.float32)
                share_obs['global_merge_obs'][e, agent_id, 3] = cv2.resize(
                    (team_histories[e] > 0).astype(np.uint8),
                    (self.input_w, self.input_h),
                ).astype(np.float32)
                share_obs['agent_beliefs'][e, agent_id] = resized_beliefs
                share_obs['global_comm_state'][e, agent_id] = global_comm_state

        return raw_obs, obs, share_obs, available_actions

    def warmup(self):
        # reset env
        dict_obs, info = self.envs.reset()  
        if self.communication_enabled:
            raw_obs, obs, share_obs, available_actions = (
                self._resize_communication_convert(dict_obs, info)
            )
            self.buffer.available_actions[0] = available_actions.copy()
        else:
            raw_obs, obs = self._resize_convert(dict_obs, info)
            raw_share_obs, share_obs = self._resize_convert(dict_obs, info)
        self.obs = raw_obs

        for key in obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()

        for key in share_obs.keys():
            self.buffer.share_obs[key][0] = share_obs[key].copy()

    def init_hyperparameters(self):
        # Calculating full and local map sizes
        # map_size = self.all_args.map_size
        map_size = 250
        input_map_size = 64
        self.max_steps = self.all_args.max_steps
        # self.agent_view_size = self.all_args.agent_view_size
        self.full_w, self.full_h = map_size, map_size
        self.input_w, self.input_h = input_map_size, input_map_size
        self.use_merge = self.all_args.use_merge
        self.communication_enabled = (
            getattr(self.all_args, 'communication_mode', None) is not None
        )
        self.use_intrinsic_reward= self.all_args.use_intrinsic_reward
        self.visualize_input = self.all_args.visualize_input
        self.local_step_num = self.all_args.local_step_num
        self.augment = 255 // (np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum())
        if self.visualize_input:
            plt.ion()
            self.fig, self.ax = plt.subplots(self.num_agents*3, 4, figsize=(10, 2.5), facecolor="whitesmoke")
 
    def init_map_variables(self):
        # Initializing full, merge and local map
        #self.global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32) 
        self.short_term_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32) 
        self.all_agent_pos_map = np.zeros((self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge or self.communication_enabled:
            #self.global_merge_goal_trace = np.zeros((self.n_rollout_threads, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            self.all_merge_pos_map = np.zeros((self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)

    def init_eval_map_variables(self):
        # Initializing full, merge and local map
        self.eval_all_agent_pos_map = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge or self.communication_enabled:
            #self.eval_global_merge_goal_trace = np.zeros((self.n_eval_rollout_threads, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            self.eval_all_merge_pos_map = np.zeros((self.n_eval_rollout_threads, self.full_w, self.full_h), dtype=np.float32)

    @torch.no_grad()
    def compute_global_goal(self, step):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        concat_obs = {}
        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][step])
        for key in self.buffer.obs.keys():
            concat_obs[key] = np.concatenate(self.buffer.obs[key][step])

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(concat_share_obs,
                            concat_obs,
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            np.concatenate(self.buffer.available_actions[step])
                            if self.buffer.available_actions is not None
                            else None)
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        
        if self.communication_enabled:
            self.short_term_goal = np.array(
                np.split(_t2n(nn.Sigmoid()(action[:, :2])), self.n_rollout_threads)
            )
            self.communication_actions = actions[:, :, 2].astype(np.int64)
        else:
            self.short_term_goal = np.array(np.split(_t2n(nn.Sigmoid()(action)), self.n_rollout_threads))
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][-1])

        next_values = self.trainer.policy.get_values(concat_share_obs,
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def insert(self, data):
        dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        dones_env = np.all(dones, axis=-1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.all_agent_pos_map[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge or self.communication_enabled:
            self.all_merge_pos_map[dones_env == True] = np.zeros(((dones_env == True).sum(), self.full_w, self.full_h), dtype=np.float32)

        if self.communication_enabled:
            raw_obs, obs, share_obs, available_actions = (
                self._resize_communication_convert(dict_obs, infos)
            )
        else:
            raw_obs, obs = self._resize_convert(dict_obs, infos)
            raw_share_obs, share_obs = self._resize_convert(dict_obs, infos)
            available_actions = None
        self.obs = raw_obs
        # if self.use_intrinsic_reward:
        #     for e in range(self.n_rollout_threads):
        #         for agent_id in range(self.num_agents):
        #             if np.any(np.array(self.map_goal[e][agent_id])<0) | np.any(np.array(self.map_goal[e][agent_id])>=self.full_w-2*self.agent_view_size):
        #                 rewards[e,agent_id,0]-=0.01
        #             else:
        #                 a=int(self.map_goal[e][agent_id][0])
        #                 b=int(self.map_goal[e][agent_id][1])
        #                 if infos[e]['explored_all_map'][self.agent_view_size:self.full_w-self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size][a,b]!=0:
        #                     rewards[e,agent_id,0]-=0.01
            
        for done_env, info in zip(dones_env, infos):
            metric_info = info.get('terminal_info', info)
            if self.communication_enabled:
                step_metrics = metric_info.get('comm_step_metrics')
                if step_metrics is not None:
                    for metric in (
                        'attempted_messages',
                        'delivered_messages',
                        'collision_messages',
                        'lost_messages',
                        'expired_messages',
                        'in_range_recipients',
                        'attempted_bits',
                        'delivered_bits',
                        'mean_latency',
                    ):
                        self.env_infos[
                            'comm_' + metric + '_per_step'
                        ].append(step_metrics[metric])
                self.env_infos['comm_reward_cost_per_step'].append(
                    float(np.sum(metric_info['comm_penalties']))
                )
            if done_env:
                self.env_infos['merge_explored_ratio_step'].append(metric_info['merge_ratio_step'])
                self.env_infos['merge_explored_ratio'].append(metric_info['merge_explored_ratio'])
                if self.communication_enabled:
                    metrics = metric_info['comm_metrics']
                    for metric in (
                        'attempted_messages',
                        'delivered_messages',
                        'collision_messages',
                        'lost_messages',
                        'expired_messages',
                        'in_range_recipients',
                        'attempted_bits',
                        'delivered_bits',
                        'mean_latency',
                    ):
                        self.env_infos[
                            'comm_episode_' + metric
                        ].append(metrics[metric])
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_ratio_step".format(agent_id)
                    self.env_infos[agent_k].append(metric_info[agent_k])

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            available_actions=available_actions,
        )
    
    def visualize_obs(self, fig, ax, obs):
        # individual
        for agent_id in range(self.num_agents * 3):
            sub_ax = ax[agent_id]
            for i in range(4):
                sub_ax[i].clear()
                sub_ax[i].set_yticks([])
                sub_ax[i].set_xticks([])
                sub_ax[i].set_yticklabels([])
                sub_ax[i].set_xticklabels([])
                if agent_id < self.num_agents:
                    sub_ax[i].imshow(obs['global_obs'][0, agent_id, i])
                elif agent_id < self.num_agents*2 and self.use_merge:
                    sub_ax[i].imshow(obs['global_merge_obs'][0, agent_id-self.num_agents, i])
                #elif i<2: sub_ax[i].imshow(obs['global_merge_goal'][0, agent_id-self.num_agents*2, i])
                #elif i < 5:
                    #sub_ax[i].imshow(obs['global_merge_goal'][0, agent_id-self.num_agents, i-4])
                    #sub_ax[i].imshow(obs['gt_map'][0, agent_id - self.num_agents, i-4])
        plt.gcf().canvas.flush_events()
        # plt.pause(0.1)
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    @torch.no_grad()
    def eval(self, total_num_steps):
        
        action_shape = get_shape_from_act_space(self.eval_envs.action_space[0])
        eval_episode_rewards = []
        eval_env_infos = defaultdict(list)

        # reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
        self.init_eval_map_variables()
        eval_dict_obs, eval_infos = self.eval_envs.reset()
        if self.communication_enabled:
            raw_eval_obs, eval_obs, _, eval_available_actions = (
                self._resize_communication_convert(
                    eval_dict_obs, eval_infos, evaluation=True
                )
            )
        else:
            raw_eval_obs, eval_obs = self._resize_eval_convert(
                eval_dict_obs, eval_infos
            )
            eval_available_actions = None

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_dones_env = np.zeros(self.n_eval_rollout_threads, dtype=bool)
        local_step = 0
        while True:
            eval_choose = (eval_dones_env==False)            
            if ~np.any(eval_choose):
                break
            eval_actions = np.full(
                (self.n_eval_rollout_threads, self.num_agents, action_shape),
                -1.0,
                dtype=np.float32,
            )
            self.short_term_goal = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 2),
                dtype=np.float32,
            )
            
            self.trainer.prep_rollout()

            concat_eval_obs = {}
            for key in eval_obs.keys():
                concat_eval_obs[key] = np.concatenate(eval_obs[key][eval_choose])          
            
            # if local_step >= self.local_step_num :
            #     local_step = 0 
            # if local_step == 0:
            # import pickle
            # output_file = open('testbed.pkl', 'wb')
            # pickle.dump(concat_eval_obs, output_file)
            # output_file.close()
            # import pickle
            #     # output_file = open('gazebo.pkl', 'wb')
            #     # pickle.dump(concat_eval_obs, output_file)
            # input_file = open('testbed.pkl', 'rb')
            # concat_eval_obs_load = pickle.load(input_file)

            # concat_eval_obs_load = pickle.load('testbed.pkl')
            eval_action, eval_rnn_state = self.trainer.policy.act(concat_eval_obs,
                                            np.concatenate(eval_rnn_states[eval_choose]),
                                            np.concatenate(eval_masks[eval_choose]),
                                            np.concatenate(eval_available_actions[eval_choose])
                                            if eval_available_actions is not None
                                            else None,
                                            deterministic=True)
            print('RL raw output: ', eval_action)
            eval_actions[eval_choose] = np.array(np.split(_t2n(eval_action), (eval_choose == True).sum()))
            eval_rnn_states[eval_choose] = np.array(np.split(_t2n(eval_rnn_state), (eval_choose == True).sum()))
        # Obser reward and next obs
            motion_action = eval_action[:, :2] if self.communication_enabled else eval_action
            self.short_term_goal[eval_choose] = np.array(
                np.split(
                    _t2n(nn.Sigmoid()(motion_action)),
                    (eval_choose == True).sum(),
                )
            )
            inputs = [{} for _ in range(self.n_eval_rollout_threads)]
            for e in range(self.n_eval_rollout_threads):
                inputs[e]['global_goal'] = self.short_term_goal[e]
                print('RL output: ', inputs[e]['global_goal'])
                # if self.use_merge:
                #     inputs[e]['global_obs'] = raw_eval_obs['global_merge_obs'][e]
                # else:
                #     inputs[e]['global_obs'] = raw_eval_obs['global_obs'][e]
                inputs[e]['global_obs'] = (
                    raw_eval_obs['global_merge_obs'][e]
                    if self.communication_enabled
                    else raw_eval_obs['global_obs'][e]
                )
            #print(self.global_goal)
            #if np.array(local_step_num).min()-1 < self.local_step_num :
                #self.local_step_num  = np.array(local_step_num).min() - 1
            eval_actions_env = self.eval_envs.get_short_term_goal(inputs)
            if self.communication_enabled:
                communication_actions = eval_actions[:, :, 2].astype(np.int64)
                eval_actions_env = [
                    {
                        'motion_goals': eval_actions_env[e],
                        'comm_actions': communication_actions[e],
                    }
                    for e in range(self.n_eval_rollout_threads)
                ]
            eval_dict_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            # eval_dict_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step_each_grid(eval_actions_env)
            
            eval_dones_env = np.all(eval_dones, axis=-1)

            self.eval_all_agent_pos_map[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            if self.use_merge or self.communication_enabled:
                self.eval_all_merge_pos_map[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.full_w, self.full_h), dtype=np.float32)

            if self.communication_enabled:
                raw_eval_obs, eval_obs, _, eval_available_actions = (
                    self._resize_communication_convert(
                        eval_dict_obs, eval_infos, evaluation=True
                    )
                )
            else:
                raw_eval_obs, eval_obs = self._resize_eval_convert(
                    eval_dict_obs, eval_infos
                )

            eval_episode_rewards.append(eval_rewards)

            for eval_info, eval_done_env in zip(eval_infos, eval_dones_env):
                metric_info = eval_info.get('terminal_info', eval_info)
                if eval_done_env:
                    eval_env_infos['eval_merge_explored_ratio_step'].append(metric_info['merge_ratio_step'])
                    eval_env_infos['eval_merge_explored_ratio'].append(metric_info['merge_explored_ratio'])
                    if self.communication_enabled:
                        metrics = metric_info['comm_metrics']
                        for metric in (
                            'attempted_messages',
                            'delivered_messages',
                            'collision_messages',
                            'lost_messages',
                            'expired_messages',
                            'in_range_recipients',
                            'attempted_bits',
                            'delivered_bits',
                            'mean_latency',
                        ):
                            eval_env_infos[
                                'eval_comm_episode_' + metric
                            ].append(metrics[metric])
                    for agent_id in range(self.num_agents):
                        agent_k = "agent{}_ratio_step".format(agent_id)
                        eval_env_infos["eval_" + agent_k].append(metric_info[agent_k])

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
     
        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
 
        print("eval average merge explored ratio is: " + str(np.mean(eval_env_infos['eval_merge_explored_ratio'])))
        print("eval average episode rewards of agent: " + str(np.mean(eval_env_infos['eval_average_episode_rewards'])))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        env_infos = defaultdict(list)

        envs = self.envs
        
        all_frames = []
        all_local_frames = []
        for episode in range(self.all_args.render_episodes):
            ic(episode)
            self.init_map_variables()
            reset_choose = np.ones(self.n_rollout_threads) == 1.0
            dict_obs, infos = envs.reset(reset_choose)
            obs = self._convert(dict_obs, infos)
            self.obs = obs

            '''if self.all_args.save_gifs:
                image, local_image = envs.render('rgb_array', self.global_goal)[0]
                all_frames.append(image)
                all_local_frames.append(local_image)
            else:
                envs.render('human', self.global_goal)'''

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            local_step = 0
            
            episode_rewards = []
            
            for step in range(self.max_steps):
                #ic(step)
                calc_start = time.time()

                self.trainer.prep_rollout()

                concat_obs = {}
                for key in obs.keys():
                    concat_obs[key] = np.concatenate(obs[key])

                

                #ic(self.global_goal*self.all_args.grid_size)                
                if local_step >= self.local_step_num :
                    local_step = 0 
                if local_step == 0:
                    action, rnn_states = self.trainer.policy.act(concat_obs,
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                    #actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                    rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                    # Obser reward and next obs
                    self.short_term_goal = np.array(np.split(_t2n(nn.Sigmoid()(action)), self.n_rollout_threads))
                     
                    inputs = [{} for _ in range(self.n_rollout_threads)]
                    for e in range(self.n_rollout_threads):
                        inputs[e]['global_goal'] = self.short_term_goal[e]
                        if self.use_merge:
                            inputs[e]['global_obs'] = self.obs['global_merge_obs'][e]
                        else:
                            inputs[e]['global_obs'] = self.obs['global_obs'][e]
                    self.map_goal = envs.get_short_term_path(inputs)#self.global_goal, self.obs['global_merge_obs'])
                    #if np.array(local_step_num).min()-1 < self.local_step_num :
                        #self.local_step_num  = np.array(local_step_num).min()-1
                
                actions_env = envs.get_short_term_action()
                local_step += 1
                
                # Obser reward and next obs            
                dict_obs, rewards, dones, infos = envs.step(actions_env)

                obs = self._convert(dict_obs, infos)
                self.obs = obs
                episode_rewards.append(rewards)
                
                for done, info in zip(dones, infos):
                    if np.all(done):
                        env_infos['merge_explored_ratio_step'].append(info['merge_ratio_step'])
                        env_infos['merge_explored_ratio'].append(info['merge_explored_ratio'])
                        for agent_id in range(self.num_agents):
                            agent_k = "agent{}_ratio_step".format(agent_id)
                            env_infos[agent_k].append(info[agent_k])
                
                dones_env = np.all(dones, axis=-1)

                rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image, local_image = envs.render('rgb_array', self.map_goal)[0]
                    all_frames.append(image)
                    all_local_frames.append(local_image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human', self.map_goal)

                if np.all(dones[0]):
                    ic("end")
                    break

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            print("average merge explored ratio is: " + str(np.mean(env_infos['merge_explored_ratio'])))
            print("average merge explored step is: " + str(np.mean(env_infos['merge_explored_ratio_step'])))

        if self.all_args.save_gifs:
            ic("rendering....")
            imageio.mimsave(str(self.gif_dir) + '/merge.gif', all_frames, duration=self.all_args.ifi)
            imageio.mimsave(str(self.gif_dir) + '/local.gif', all_local_frames, duration=self.all_args.ifi)
            ic("done")
