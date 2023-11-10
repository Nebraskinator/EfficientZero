
#import tracemalloc

#tracemalloc.start()

import ray
import time
import torch

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


@ray.remote(num_gpus=0.125, max_restarts=-1, max_task_retries=-1)
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
        envs = [self.config.new_game(seed=self.config.seed, log=(i==0 and self.log)) for i in range(env_nums)] 
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
                
                game_histories = []
                last_game_histories = []
                last_game_priorities = []
                stack_obs_windows = []
                taking_actions = []
                action_masks = []
                search_values_lst = []
                pred_values_lst = []
                for i, env in enumerate(envs):
                    init_obses, acting, action_mask = env.reset()
                    taking_actions.append(acting)
                    action_masks.append(action_mask)

                    game_histories.append({p: GameHistory(env.action_space_size(), env.obs_shape, max_length=self.config.history_length,
                                                  config=self.config) for p in env.live_agents})
                    last_game_histories.append({p: None for p in env.live_agents})
                    last_game_priorities.append({p: None for p in env.live_agents})

                    sw = {}

                    for p in env.live_agents:
                        sw[p] = [init_obses[p] for _ in range(self.config.stacked_observations)]
                        game_histories[i][p].init(sw[p])
                    stack_obs_windows.append(sw)
                    
                    # for priorities in self-play
                    search_values_lst.append({p: [] for p in env.live_agents})
                    pred_values_lst.append({p: [] for p in env.live_agents})

                num_agents = [len(env.live_agents) for env in envs]
                step_counter = 0
                env_done_cnt = [0 for _ in range(env_nums)]
                dones = [False for _ in range(env_nums)]
                # play games until max moves
                while sum(dones) < env_nums and (step_counter <= self.config.max_moves):

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
                        if not dones[i]:
                            step_acting_agents.append([p for p in env.live_agents if taking_actions[i][p]])
                        else:
                            step_acting_agents.append([])
                        
                    tokens, temperatures = {}, []
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

                    if obs_to_stack:
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
                        if ((start_training or self.config.resume_training) and trained_steps >= value_training_start):
                            tree_nodes = self.config.num_simulations
                        pool_size = self.config.action_space_size * (tree_nodes + 2)
                        roots = cytree.Roots(len(obs_to_stack), pool_size)
                        policy_logits_pool = []
                        #noises = []
                        action_mappings = []
                        idx = 0
                        for i, env in enumerate(envs):
                            for p in step_acting_agents[i]:
                                if i == 0:
                                    tokens[p] = chance_token_pool[idx] # for logging
                                a_mask = np.flatnonzero(action_masks[i][p])
                                #noises.append(np.random.dirichlet([self.config.root_dirichlet_alpha] * np.sum(action_masks[p]).astype(int)).astype(np.float32).tolist())
                                policy_logits_pool.append(network_output.policy_logits[idx, a_mask].astype(np.float32).tolist())
                                action_mappings.append(a_mask.tolist())
                                idx += 1
                        #roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, value_pool, policy_logits_pool, action_mappings)
                        roots.prepare_no_noise(value_prefix_pool, value_pool, policy_logits_pool, action_mappings)
    
                        # do MCTS for a policy
                        mcts.search(roots, model, hidden_state_roots, training_start=((start_training or self.config.resume_training) and trained_steps >= value_training_start))
    
                        #roots_distributions = roots.get_distributions()
                        roots_values = roots.get_values()
                        roots_completed_values = roots.get_children_values(self.config.discount)
                        roots_improved_policy_probs = roots.get_policies(self.config.discount) # new policy constructed with completed Q in gumbel muzero
                    idx = 0
                    for i, env in enumerate(envs):
                        if not dones[i]:                            
                            action, dists, vals = {}, {}, {}
                            for p in step_acting_agents[i]:    
                                if self.config.use_priority and not self.config.use_max_priority and start_training:
                                    pred_values_lst[i][p].append(network_output.value[idx].item())
                                    search_values_lst[i][p].append(roots_values[idx])                              
                                if (start_training or self.config.resume_training) and trained_steps >= learned_agent_actions_start:
                                    search_stats, value, temperature = roots_improved_policy_probs[idx], float(np.max(roots_completed_values[idx])), float(temperatures[i][p])
                                    distributions = np.zeros(self.config.action_space_size)
                                    distributions[np.flatnonzero(action_masks[i][p])] = search_stats
                                    action[p] = np.argmax(distributions)
                                    if step_counter % 4 == 0:
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
                            
    
                            for p in env.live_agents:
                                if p not in step_acting_agents[i]:
                                    action[p] = 2090
                            if i == 0:
                                self.config.record_tokens(tokens, env)
                                self.config.record_best_actions(action, dists, env)                                       
                            obs, ori_reward, taking_action, done, action_mask = env.step(action)
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
                                    obs[p] = np.zeros(env.obs_shape).astype(int)
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
                                    game_histories[i][p] = GameHistory(env.action_space_size(), max_length=self.config.history_length,
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
                                dones[i] = True
                                if i==0 and self.log and (start_training or self.config.resume_training) and trained_steps >= value_training_start and os.path.isfile("log.txt"):
                                    subdir = "./logs"
                                    prev = [int(x.split("_")[0]) for x in os.listdir(subdir) if x.endswith("txt")]
                                    save_idx = 0
                                    if prev:
                                        save_idx = max(prev) + 1
                                    save_idx = str(save_idx).zfill(6)
                                    win = 'W'
                                    if env.PLAYERS['player_0'].health < 1:
                                        win = 'L'
                                    m = 0
                                    c = 'None'
                                    for k, v in env.PLAYERS['player_0'].team_tiers.items():
                                        if v:
                                            if v == m:
                                                c = c + k
                                            if v > m:
                                                m = v
                                                c = k
                                    shutil.copy("log.txt", "./logs/{}_{}_{}.txt".format(save_idx, win, c))
                    del roots
                print(ray.get(self.replay_buffers[0].size.remote()))
