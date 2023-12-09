
#import tracemalloc

#tracemalloc.start()

import ray
import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import numpy as np
import core.ctree.cytree as cytree

import gc
import copy

import os
import shutil

from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


@ray.remote(max_restarts=-1, max_task_retries=-1)
class DataWorkerSpawner(object):
    def __init__(self, rank, replay_buffers, storage, config, log=True):
        """Data Worker for collecting data through self-play
        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        """
        self.rank = rank
        self.log=log
        self.config = config
        self.storage = storage
        self.replay_buffers = replay_buffers
        # double buffering when data is sufficient
        self.trajectory_pools = [[] for _ in range(self.config.num_models)]
        self.pool_size = 1
        self.device = self.config.device
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = [-1 for _ in range(self.config.num_models)]
    def run(self):
        while True:
            worker = DataWorker.remote(self.rank, self.replay_buffers, self.storage, self.config, log=self.log)
            not_done_ids = [worker.run.remote()]
            while not_done_ids:
                done_ids, not_done_ids = ray.wait(not_done_ids)
                time.sleep(1)
            del worker


@ray.remote(num_tpus=0.375, max_restarts=-1, max_task_retries=-1)
class DataWorker(object):
    def __init__(self, rank, replay_buffers, storage, config, log=True):
        """Data Worker for collecting data through self-play
        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        """
        self.rank = rank
        self.log=log
        self.config = config
        self.storage = storage
        self.replay_buffers = replay_buffers
        # double buffering when data is sufficient
        self.trajectory_pools = [[] for _ in range(self.config.num_models)]
        self.pool_size = 1
        self.device = self.config.device
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = -1

    def put(self, data, curr_model=0):
        # put a game history into the pool
        self.trajectory_pools[curr_model].append(data)

    def len_pool(self, curr_model=0):
        # current pool size
        return len(self.trajectory_pools[curr_model])

    def free(self, curr_model=0):
        # save the game histories and clear the pool
        if self.len_pool(curr_model) >= self.pool_size: 
            #print('saving for model '+str(curr_model))
            self.replay_buffers[curr_model].save_pools.remote(self.trajectory_pools[curr_model], self.gap_step)
            #print('saving memory: '+str(ray.get(self.replay_buffers[curr_model].size.remote())))
        del self.trajectory_pools[curr_model][:]

    def put_last_trajectory(self, player, last_game_histories, last_game_priorities, game_histories, curr_model=0):
        """put the last game history into the pool if the current game is finished
        Parameters
        ----------
        last_game_histories: list
            list of the last game histories
        last_game_priorities: list
            list of the last game priorities
        game_histories: list
            list of the current game histories
        """
        # pad over last block trajectory
        beg_index = self.config.stacked_observations
        end_index = beg_index + self.config.num_unroll_steps

        pad_obs_lst = game_histories[player].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_histories[player].child_visits[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[player].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[player].root_values[beg_index:end_index]

        # pad over and save
        last_game_histories[player].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        last_game_histories[player].game_over()

        self.put([last_game_histories[player], last_game_priorities[player]], curr_model)
        self.free(curr_model)

        # reset last block
        last_game_histories[player] = None
        last_game_priorities[player] = None

    def get_priorities(self, player, pred_values_lst, search_values_lst):
        # obtain the priorities at index i
        if self.config.use_priority and not self.config.use_max_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[player])).to(self.device).float()
            search_values = torch.from_numpy(np.array(search_values_lst[player])).to(self.device).float()
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities


    def run(self):
        
        # number of parallel mcts
        env_nums = self.config.p_mcts_num
        model = self.config.get_uniform_network() 
        if self.config.resume_training:
            print("Self-Play with Stored Weights")
           
        weights = ray.get(self.storage.get_weights.remote(0))
        weights_copy = copy.deepcopy(weights)
        model.set_weights(weights_copy)
        del weights
        gc.collect()
        model.to(self.device)
        model.eval()

        start_training = False
        envs = [self.config.new_game(seed=self.config.seed, log=False) for _ in range(env_nums)] 
        epsilon = 1 / self.config.action_space_size
        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        mcts = MCTS(self.config)
        learned_agent_actions_start = self.config.learned_agent_actions_start
        value_training_start = learned_agent_actions_start
        #snap_0 = tracemalloc.take_snapshot()
        with torch.no_grad():
            for _ in range(10):
                gc.collect()
                print("self play: new game")
                trained_steps = ray.get(self.storage.get_counter.remote(0))
                # training finished
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    time.sleep(30)
                    break
                
                game_histories = [None for _ in range(env_nums)]
                last_game_histories = [None for _ in range(env_nums)]
                last_game_priorities = [None for _ in range(env_nums)]
                stack_obs_windows = [None for _ in range(env_nums)]
                taking_actions = [None for _ in range(env_nums)]
                action_masks = [None for _ in range(env_nums)]
                search_values_lst = [None for _ in range(env_nums)]
                pred_values_lst = [None for _ in range(env_nums)]
                
                def reset_env(idx, env):
                    if self.log:
                        subdir = "./logs"
                        prev = [int(x.split(".")[0]) for x in os.listdir(subdir) if x.endswith("txt")]
                        save_idx = 0
                        if prev:
                            save_idx = max(prev) + 1
                        save_idx = str(save_idx).zfill(6)
                        log = subdir + "/{}.txt".format(save_idx)
                    handle = env.reset.remote(log=log)
                    init_obses, acting, action_mask = ray.get(handle)
                    taking_actions[idx] = acting
                    action_masks[idx] = action_mask
                    env_live_agents = ray.get(env.live_agents.remote())
                    game_histories[idx] = {p: GameHistory(ray.get(env.action_space_size.remote()), ray.get(env.obs_shape.remote()), max_length=self.config.history_length,
                                                  config=self.config) for p in env_live_agents}
                    last_game_histories[idx] = {p: None for p in env_live_agents}
                    last_game_priorities[idx] = {p: None for p in env_live_agents}

                    sw = {}

                    for p in env_live_agents:
                        sw[p] = [init_obses[p] for _ in range(self.config.stacked_observations)]
                        game_histories[i][p].init(sw[p])
                    stack_obs_windows[idx] = sw
                    
                    # for priorities in self-play
                    search_values_lst[idx] = {p: [] for p in env_live_agents}
                    pred_values_lst[idx] = {p: [] for p in env_live_agents}
                
                for i, env in enumerate(envs):
                    reset_env(i, env)

                num_agents = [len(ray.get(env.live_agents.remote())) for env in envs]
                step_counter = 0
                env_done_cnt = [0 for _ in range(env_nums)]
                #dones = [False for _ in range(env_nums)]
                # play games until max moves
                while True:

                    if not start_training:
                        start_training = ray.get(self.storage.get_start_signal.remote(0))
                    
                    # get model
                    trained_steps = ray.get(self.storage.get_counter.remote(0))
                    # training finished
                    if trained_steps >= self.config.training_steps + self.config.last_steps:
                        time.sleep(30)
                        break
                    new_model_index = trained_steps // self.config.checkpoint_interval
                    if new_model_index > self.last_model_index:
                        self.last_model_index = new_model_index
                        # update model
                        weights = ray.get(self.storage.get_weights.remote(0))
                        weights_copy = copy.deepcopy(weights)
                        model.set_weights(weights_copy)
                        model.to(self.device)
                        model.eval()
                        del weights
                        del weights_copy
                        gc.collect()
                    step_counter += 1
                    
                    step_acting_agents = []
                    for i, env in enumerate(envs):
                        env_live_agents = ray.get(env.live_agents.remote())
                        step_acting_agents.append([p for p in env_live_agents if taking_actions[i][p]])
                        
                    tokens, temperatures = [{} for _ in range(env_nums)], []
                    # stack obs for model inference
                    obs_to_stack = []
                    for i, env in enumerate(envs):
                        if step_acting_agents[i]:
                            # set temperature for distributions
                            temperatures.append({p: self.config.visit_softmax_temperature_fn(num_moves=0, 
                                                                                             trained_steps=trained_steps) for p in step_acting_agents[i]})
                            obs_to_stack += [np.concatenate(game_histories[i][p].step_obs(), 0) for p in step_acting_agents[i]]
                        else:
                            temperatures.append({})
                    roots = None
                    if obs_to_stack and (start_training or self.config.resume_training) and trained_steps >= value_training_start:
                        stack_obs = torch.from_numpy(np.array(obs_to_stack)).float().to(self.device)
                        if self.config.amp_type == 'torch_amp':
                            with autocast():
                                network_output = model.initial_inference(stack_obs)
                        else:                        
                            network_output = model.initial_inference(stack_obs.float())
                                                                
                        hidden_state_roots = network_output.hidden_state
                        #reward_hidden_roots = network_output.reward_hidden
                        value_prefix_pool = network_output.value_prefix
                        value_pool = network_output.value.reshape(-1).tolist()
                        chance_token_pool = network_output.chance_token_onehot.detach().cpu()
                        tree_nodes = 1
                        if (start_training or self.config.resume_training) and trained_steps >= value_training_start:
                            tree_nodes = self.config.num_simulations
                        pool_size = self.config.action_space_size * (tree_nodes + 2)
                        roots = cytree.Roots(len(obs_to_stack), pool_size)
                        policy_logits_pool = []
                        #noises = []
                        action_mappings = []
                        idx = 0
                        for i, env in enumerate(envs):
                            for p in step_acting_agents[i]:
                                tokens[i][p] = chance_token_pool[idx] # for logging
                                a_mask = np.flatnonzero(action_masks[i][p])
                                #noises.append(np.random.dirichlet([self.config.root_dirichlet_alpha] * np.sum(action_masks[p]).astype(int)).astype(np.float32).tolist())
                                policy_logits_pool.append(network_output.policy_logits[idx, a_mask].astype(np.float32).tolist())
                                action_mappings.append(a_mask.tolist())
                                idx += 1
                        #roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, value_pool, policy_logits_pool, action_mappings)
                        roots.prepare_no_noise(value_prefix_pool, value_pool, policy_logits_pool, action_mappings)
    
                        # do MCTS for a policy
                        mcts.search(roots, model, hidden_state_roots, training_start=(start_training or self.config.resume_training) and trained_steps >= value_training_start)
    
                        #roots_distributions = roots.get_distributions()
                        roots_values = roots.get_values()
                        roots_completed_values = roots.get_children_values(self.config.discount)
                        roots_improved_policy_probs = roots.get_policies(self.config.discount) # new policy constructed with completed Q in gumbel muzero
                    #print({a: (round(v, 2), round(p, 2)) for a, v, p in zip(action_mappings[0], roots_completed_values[0], roots_improved_policy_probs[0])})
                    idx = 0
                    handles = []
                    actions = []
                    dists_list = []
                    vals_list = []
                    for i, env in enumerate(envs):                          
                        action, dists, vals = {}, {}, {}
                        for p in step_acting_agents[i]:    
                            if self.config.use_priority and not self.config.use_max_priority and (start_training or self.config.resume_training):
                                pred_values_lst[i][p].append(network_output.value[idx].item())
                                search_values_lst[i][p].append(roots_values[idx])                              
                            if (start_training or self.config.resume_training) and trained_steps >= learned_agent_actions_start:
                                search_stats, value, temperature = roots_improved_policy_probs[idx], float(roots_values[idx]), float(temperatures[i][p])
                                distributions = np.zeros(self.config.action_space_size)
                                distributions[np.flatnonzero(action_masks[i][p])] = search_stats
                                #action[p] = np.argmax(distributions)
                                action[p], _ = select_action(distributions, temperature=temperature, deterministic=False)
                            else:
                                # before starting training, use random policy
                                if (start_training or self.config.resume_training) and trained_steps >= value_training_start:
                                    value, temperature = float(roots_values[idx]), float(temperatures[i][p])
                                else:
                                    value, temperature = 0., 0.3
                                distributions = action_masks[i][p].copy()
                                action[p] = np.random.choice(np.flatnonzero(distributions))
                            dists[p] = distributions
                            vals[p] = value
                            idx += 1
                        
                        env_live_agents = ray.get(env.live_agents.remote())
                        for p in env_live_agents:
                            if p not in step_acting_agents[i]:
                                action[p] = 2090

                        env.record_tokens.remote(tokens[i])
                        env.record_best_actions.remote(action, dists)                                       
                        handles.append(env.step.remote(action))
                        actions.append(action)
                        dists_list.append(dists)
                        vals_list.append(vals)
                    for i, env in enumerate(envs):
                        obs, ori_reward, taking_action, done, action_mask = ray.get(handles[i])
                        action = actions[i]
                        dists = dists_list[i]
                        vals = vals_list[i]
                        taking_actions[i] = taking_action
                        action_masks[i] = action_mask
                        for p in step_acting_agents[i]:
                            # clip the reward
                            if self.config.clip_reward:
                                clip_reward = np.sign(ori_reward[p])
                            else:
                                clip_reward = ori_reward[p]
                            game_histories[i][p].store_search_stats(dists[p].copy(), float(vals[p]))
                            if p not in obs:
                                obs[p] = np.zeros(ray.get(env.obs_shape.remote())).astype(int)
                            game_histories[i][p].append(int(action[p]), obs[p].copy(), float(clip_reward))
        
                            # fresh stack windows
                            del stack_obs_windows[i][p][0]
                            stack_obs_windows[i][p].append(obs[p].copy())
                            
                            # if game history is full;
                            # we will save a game history if it is the end of the game or the next game history is finished.
                            if game_histories[i][p].is_full():
                                # pad over last block trajectory
                                if last_game_histories[i][p] is not None:
                                    self.put_last_trajectory(p, last_game_histories[i], last_game_priorities[i], game_histories[i])
    
                                # calculate priority
                                priorities = self.get_priorities(p, pred_values_lst[i], search_values_lst[i])
    
                                # save block trajectory
                                last_game_histories[i][p] = game_histories[i][p]
                                last_game_priorities[i][p] = priorities
    
                                # new block trajectory
                                game_histories[i][p] = GameHistory(ray.get(env.action_space_size.remote()), max_length=self.config.history_length,
                                                                config=self.config)
                                game_histories[i][p].init(stack_obs_windows[i][p])
                        for p in done.keys():
                            # reset env if finished
                            if done[p]:
                                env_done_cnt[i] += 1

                                # pad over last block trajectory
                                if last_game_histories[i][p] is not None:
                                    self.put_last_trajectory(p, last_game_histories[i], last_game_priorities[i], game_histories[i])
                                    
                                # store current block trajectory
                                priorities = self.get_priorities(p, pred_values_lst[i], search_values_lst[i])
                                                      
                                game_histories[i][p].game_over()
    
                                self.put([game_histories[i][p], priorities])
                                self.free()
                                
                                del game_histories[i][p]
                        gc.collect()

                        if env_done_cnt[i] >= num_agents[i]:
                            env_done_cnt[i] = 0
                            reset_env(i, env)
                    if roots:
                        del roots
                print(ray.get(self.replay_buffers[0].size.remote()))
