
#import tracemalloc

#tracemalloc.start()

import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

import gc
import copy

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
        self.last_model_index = [-1 for _ in range(self.config.num_models)]

    def put(self, data, curr_model):
        # put a game history into the pool
        self.trajectory_pools[curr_model].append(data)

    def len_pool(self, curr_model):
        # current pool size
        return len(self.trajectory_pools[curr_model])

    def free(self, curr_model):
        # save the game histories and clear the pool
        if curr_model not in self.config.freeze_models and self.len_pool(curr_model) >= self.pool_size: 
            #print('saving for model '+str(curr_model))
            self.replay_buffers[curr_model].save_pools.remote(self.trajectory_pools[curr_model], self.gap_step)
        del self.trajectory_pools[curr_model][:]

    def put_last_trajectory(self, player, last_game_histories, last_game_priorities, game_histories, curr_model):
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
        env_nums = 1 #self.config.p_mcts_num
        models = [self.config.get_uniform_network() for _ in range(self.config.num_models)]
        #prev_models = [self.config.get_uniform_network() for _ in range(self.config.num_prev_models)]
        if self.config.resume_training:
            for i, model in enumerate(models):
                print("Self-Play with Stored Weights")
                weights = ray.get(self.storage.get_weights.remote(i))
                weights_copy = copy.deepcopy(weights)
                model.set_weights(weights_copy)
                del weights
                gc.collect()
            #prev_weights = ray.get(self.storage.get_previous_models_weights.remote())
            #for model, weights in zip(prev_models, prev_weights):
            #    model.set_weights(weights)
            #    model.to(self.device)
            #    model.eval()
        [model.to(self.device) for model in models]
        [model.eval() for model in models]
        #[model.to(self.device) for model in prev_models]
        #[model.eval() for model in prev_models]
        start_training = [False for _ in range(self.config.num_models)]
        env = self.config.new_game(seed=self.config.seed, log=self.log) 
        epsilon = 1 / self.config.action_space_size
        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(self.config.action_space_size)
        # 100k benchmark
        total_transitions = 0
        # max transition to collect for this data worker
        max_transitions = self.config.total_transitions // self.config.num_actors
        mcts = MCTS(self.config)
        #snap_0 = tracemalloc.take_snapshot()
        with torch.no_grad():
            for _ in range(2):
                gc.collect()
                #snap_1 = tracemalloc.take_snapshot()
                #stats = snap_1.compare_to(snap_0, 'lineno')
                #for s in stats[:10]:
                #    print(s)
                print("self play: new game")
                #if prev_models:
                #    prev_weights = ray.get(self.storage.get_previous_models_weights.remote())
                #    for model, weights in zip(prev_models, prev_weights):
                #       model.set_weights(weights)
                #        model.to(self.device)
                #        model.eval()
                trained_steps = [ray.get(self.storage.get_counter.remote(i)) for i in range(self.config.num_models)]
                # training finished
                if all([s >= self.config.training_steps + self.config.last_steps for s in trained_steps]):
                    time.sleep(30)
                    break

                init_obses, taking_actions, action_masks = env.reset()
                             
                num_actors = len(env.live_agents)
                #live_actors = env.live_agents[:-self.config.num_random_actors - self.config.num_prev_models]
                if self.config.num_random_actors:
                    live_actors = env.live_agents[:-self.config.num_random_actors]
                else:
                    live_actors = list(env.live_agents)
                dead_actors = []
                #prev_actors = env.live_agents[-self.config.num_random_actors - self.config.num_prev_models : -self.config.num_random_actors]
                random_actors = env.live_agents[-self.config.num_random_actors:]
                rewards = {p: [] for p in live_actors}
                actor_model_dict = {i: [] for i in range(self.config.num_models)}
                for i, actor in enumerate(live_actors):
                    actor_model_dict[i % self.config.num_models].append(actor)                  
                dones = {p:False for p in live_actors}
                #prev_game_histories = {p: GameHistory(env.action_space_size(), env.obs_shape, max_length=self.config.history_length,
                #                             config=self.config) for p in prev_actors}
                game_histories = {p: GameHistory(env.action_space_size(), env.obs_shape, max_length=self.config.history_length,
                                              config=self.config) for p in live_actors}
                last_game_histories = {p: None for p in live_actors}
                last_game_priorities = {p: None for p in live_actors}

                # stack observation windows in boundary: s398, s399, s400, current s1 -> for not init trajectory
                #stack_obs_windows = {p: [] for p in live_actors + prev_actors}
                stack_obs_windows = {p: [] for p in live_actors}

                for p in live_actors:
                    stack_obs_windows[p] = [init_obses[p] for _ in range(self.config.stacked_observations)]
                    game_histories[p].init(stack_obs_windows[p])
                #for p in prev_actors:
                #    stack_obs_windows[p] = [init_obses[p] for _ in range(self.config.stacked_observations)]
                #    prev_game_histories[p].init(stack_obs_windows[p])

                # for priorities in self-play
                search_values_lst = {p: [] for p in live_actors}
                pred_values_lst = {p: [] for p in live_actors}

                # some logs
                eps_ori_reward_lst, eps_reward_lst, eps_steps_lst, visit_entropies_lst = 0, 0, 0, 0
                step_counter = 0

                self_play_rewards = 0.
                self_play_ori_rewards = 0.
                self_play_moves = 0.
                self_play_episodes = 0.

                self_play_rewards_max = - np.inf
                self_play_moves_max = 0

                self_play_visit_entropy = []
                other_dist = {}
                #prev_pred_values = {}
                #prev_search_values = {}
                done_cnt = 0
                
                # play games until max moves
                while done_cnt < num_actors and (step_counter <= self.config.max_moves):
                    
                    live_actors = [p for p in env.live_agents if p in live_actors and p not in dead_actors]
                    #prev_actors = [p for p in env.live_agents if p in prev_actors]
                    random_actors = [p for p in env.live_agents if p in random_actors and p not in dead_actors]
                    step_acting_agents = [i for i in live_actors if taking_actions[i]]
                    #prev_acting_agents = [i for i in prev_actors if taking_actions[i]]
                    if not all(start_training):
                        for i, start in enumerate(start_training):
                            if not start:
                                start_training[i] = ray.get(self.storage.get_start_signal.remote(i))
                    
                    # get model
                    trained_steps = [ray.get(self.storage.get_counter.remote(i)) for i in range(self.config.num_models)]
                    # training finished
                    if all([s >= self.config.training_steps + self.config.last_steps for s in trained_steps]):
                        time.sleep(30)
                        break
                    '''
                    if start_training and (total_transitions / max_transitions) > (trained_steps / self.config.training_steps):
                        # self-play is faster than training speed or finished
                        print("waiting")
                        time.sleep(1)
                        continue
                    '''


                    # update the models in self-play every checkpoint_interval
                    for i in range(self.config.num_models):
                        new_model_index = trained_steps[i] // self.config.checkpoint_interval
                        if new_model_index > self.last_model_index[i]:
                            self.last_model_index[i] = new_model_index
                            # update model
                            weights = ray.get(self.storage.get_weights.remote(i))
                            weights_copy = copy.deepcopy(weights)
                            models[i].set_weights(weights_copy)
                            models[i].to(self.device)
                            models[i].eval()
                            del weights
                            del weights_copy
                            gc.collect()

                        '''
                        # log if more than 1 env in parallel because env will reset in this loop.
                        if env_nums > 1:
                            if len(self_play_visit_entropy) > 0:
                                visit_entropies = np.array(self_play_visit_entropy).mean()
                                visit_entropies /= max_visit_entropy
                            else:
                                visit_entropies = 0.

                            if self_play_episodes > 0:
                                log_self_play_moves = self_play_moves / self_play_episodes
                                log_self_play_rewards = self_play_rewards / self_play_episodes
                                log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                            else:
                                log_self_play_moves = 0
                                log_self_play_rewards = 0
                                log_self_play_ori_rewards = 0

                            self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                            log_self_play_ori_rewards, log_self_play_rewards,
                                                                            self_play_rewards_max, _temperature.mean(),
                                                                            visit_entropies, 0,
                                                                            other_dist)
                            self_play_rewards_max = - np.inf
                        '''

                    step_counter += 1
                    action, visit_entropy, dists, vals = {}, {}, {}, {}
                    # stack obs for model inference
                    
                    for curr_model, model in enumerate(models):
                        model_acting_agents = [p for p in actor_model_dict[curr_model] if p in step_acting_agents]
                        if model_acting_agents:
                            # set temperature for distributions
                            _temperature = np.array(
                                [self.config.visit_softmax_temperature_fn(num_moves=0, trained_steps=trained_steps[curr_model]) for player in
                                 model_acting_agents])
                            stack_obs = np.array([np.concatenate(game_histories[p].step_obs(), 0) for p in model_acting_agents])
                            stack_obs = np.moveaxis(stack_obs, -1, -3).astype(float) / 255.
                            stack_obs = torch.from_numpy(stack_obs).to(self.device).float()
                            if self.config.amp_type == 'torch_amp':
                                with autocast():
                                    network_output = model.initial_inference(stack_obs.float())
                            else:                        
                                network_output = model.initial_inference(stack_obs.float())
                            hidden_state_roots = network_output.hidden_state
                            reward_hidden_roots = network_output.reward_hidden
                            value_prefix_pool = network_output.value_prefix
                            #policy_logits_pool = np.array([np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum() for x in network_output.policy_logits])
                            #policy_logits_pool = (network_output.policy_logits * np.array([action_masks[p] for p in model_acting_agents])).astype(np.float32).tolist()
                            #policy_logits_pool = (network_output.policy_logits * np.array([action_masks[p] for p in model_acting_agents])).astype(np.float32).tolist()
                            roots = cytree.Roots(len(model_acting_agents), self.config.action_space_size, self.config.num_simulations if (start_training[curr_model] or self.config.load_model) else 1)
                            policy_logits_pool = []
                            noises = []
                            for i, p in enumerate(model_acting_agents):
                                #noise = np.zeros(self.config.action_space_size)
                                #noise[np.flatnonzero(action_masks[p])] = np.random.dirichlet([self.config.root_dirichlet_alpha] * np.sum(action_masks[p]).astype(int))
                                #noises.append(noise.astype(np.float32).tolist())  
                                noises.append(np.random.dirichlet([self.config.root_dirichlet_alpha] * np.sum(action_masks[p]).astype(int)).astype(np.float32).tolist())
                                policy_logits_pool.append(network_output.policy_logits[i, np.flatnonzero(action_masks[p])].astype(np.float32).tolist())
                            #noises = [(np.random.dirichlet([self.config.root_dirichlet_alpha] * np.sum(action_masks[p]))).astype(np.float32).tolist() for p in step_acting_agents]
                            #print((noises, policy_logits_pool))
                            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                            # do MCTS for a policy
                            mcts.search(roots, model, hidden_state_roots, reward_hidden_roots, training_start=(start_training[curr_model] or self.config.load_model))
        
                            roots_distributions = roots.get_distributions()
                            roots_values = roots.get_values()
                                                        
                            for i, p in enumerate(model_acting_agents):
                                if self.config.use_priority and not self.config.use_max_priority and start_training[curr_model]:
                                    pred_values_lst[p].append(network_output.value[i].item())
                                    search_values_lst[p].append(roots_values[i])
                                    #search_values_lst[p].append(0)
                                deterministic = False                                
                                if (start_training[curr_model] or self.config.resume_training) and all([s >= 50000 for s in trained_steps]):
                                    search_stats, value, temperature = roots_distributions[i], float(roots_values[i]), float(_temperature[i])
                                    #distributions, value, temperature = np.ones(self.config.action_space_size), 0., 0.3
                                    distributions = np.zeros(self.config.action_space_size)
                                    distributions[np.flatnonzero(action_masks[p])] = search_stats
                                    distributions = distributions.astype(float)
                                    action[p], visit_entropy[p] = select_action(distributions, temperature=temperature, deterministic=deterministic)
                                else:
                                    # before starting training, use random policy
                                    #value, temperature = float(roots_values[i]), float(_temperature[i])
                                    value, temperature = 0., 0.3
                                    distributions = np.zeros(self.config.action_space_size).astype(float)
                                    #distributions = np.ones(self.config.action_space_size).astype(float)
                                    #distributions *= action_masks[p]
                                    a = env.PLAYERS[p].ai.get_action()
                                    distributions[a] = 1
                                    action[p] = a
                                    _, visit_entropy[p] = select_action(distributions, temperature=temperature, deterministic=deterministic)
                                #if np.sum(distributions) == 0:
                                #    distributions[1976] = 1 
                                dists[p] = distributions
                                vals[p] = value
                                #print((np.min(policy_logits_pool[i]), np.max(policy_logits_pool[i])))
                                #print((np.min(distributions), np.max(distributions)))
                                #prev_pred_values[p] = network_output.value[i].item()
                                #prev_search_values[p] = value
                                
                                #action[p], visit_entropy[p] = select_action(distributions, temperature=temperature, deterministic=deterministic)
                            del roots
                    
                    
                    '''
                    for p, model in zip(prev_actors, prev_models):
                        if taking_actions[p]:
                            # set temperature for distributions
                            _temperature = np.array(
                                [self.config.visit_softmax_temperature_fn(num_moves=0, trained_steps=trained_steps[0])])
                            stack_obs = np.array([np.concatenate(prev_game_histories[p].step_obs(), 0)])
                            stack_obs = np.moveaxis(stack_obs, -1, -3).astype(float) / 255.
                            stack_obs = torch.from_numpy(stack_obs).to(self.device).float()
                            if self.config.amp_type == 'torch_amp':
                                with autocast():
                                    network_output = model.initial_inference(stack_obs.float())
                            else:                        
                                network_output = model.initial_inference(stack_obs.float())
                            hidden_state_roots = network_output.hidden_state
                            reward_hidden_roots = network_output.reward_hidden
                            value_prefix_pool = network_output.value_prefix
                            policy_logits_pool = (network_output.policy_logits * np.array([action_masks[p]])).tolist()
                            roots = cytree.Roots(1, self.config.action_space_size, self.config.num_simulations)
                            noises = []
                            noise = np.zeros(self.config.action_space_size)
                            noise[np.flatnonzero(action_masks[p])] = np.random.dirichlet([self.config.root_dirichlet_alpha] * np.sum(action_masks[p]).astype(int))
                            noises.append(noise.astype(np.float32).tolist())                           
                            #noises = [(np.random.dirichlet([self.config.root_dirichlet_alpha] * np.sum(action_masks[p]))).astype(np.float32).tolist() for p in step_acting_agents]
                            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                            # do MCTS for a policy
                            MCTS(self.config).search(roots, model, hidden_state_roots, reward_hidden_roots, training_start=(start_training or self.config.load_model))
                    
                            roots_distributions = roots.get_distributions()
                            roots_values = roots.get_values()
                            #print((np.array(roots_distributions).shape, np.array(roots_values).shape, np.array(_temperature).shape))
                            if start_training or self.config.resume_training:
                                distributions, value, temperature = roots_distributions[0], roots_values[0], _temperature[0]
                            else:
                                # before starting training, use random policy
                                value, temperature = roots_values[0], _temperature[0]
                                distributions = np.ones(self.config.action_space_size)
                            distributions *= action_masks[p]
                            if np.sum(distributions) == 0:
                                distributions[1976] = 1 
                                
                            action[p], visit_entropy[p] = select_action(distributions, temperature=temperature, deterministic=deterministic)
                        else:
                            action[p] = 1976
                    '''
                    for p in live_actors:
                        if p not in step_acting_agents:
                            action[p] = 1976
                    
                    self.config.record_best_actions(action, dists, env)
                    
                    for p in random_actors:
                        #action[p] = np.random.choice(np.where(action_masks[p])[0].tolist())
                        action[p] = env.PLAYERS[p].ai.get_action()
                    
                    '''
                    for p in live_actors:
                        action[p] = np.random.choice(np.where(action_masks[p])[0].tolist())
                        dists[p] = np.ones(env.action_space_size())
                        vals[p] = 0
                        visit_entropy[p] = 0
                    '''
                        
                    obs, ori_reward, taking_actions, dones, action_masks = env.step(action)

                    for p in live_actors:
                        rewards[p].append(ori_reward[p])

                    for p in step_acting_agents:
                        # clip the reward
                        if self.config.clip_reward:
                            clip_reward = np.sign(ori_reward[p])
                        else:
                            clip_reward = ori_reward[p]

                        # store data
                        game_histories[p].store_search_stats(dists[p].copy(), float(vals[p]))
                        if p not in obs:
                            obs[p] = np.zeros(env.obs_shape).astype(int)
                        game_histories[p].append(int(action[p]), obs[p].copy(), float(clip_reward))

                        eps_reward_lst += clip_reward
                        eps_ori_reward_lst += ori_reward[p]

                        visit_entropies_lst += visit_entropy[p]

                        eps_steps_lst += 1
                        total_transitions += 1

                        # fresh stack windows
                        del stack_obs_windows[p][0]
                        stack_obs_windows[p].append(obs[p].copy())
                        
                        # if game history is full;
                        # we will save a game history if it is the end of the game or the next game history is finished.
                        if game_histories[p].is_full():
                            # pad over last block trajectory
                            if last_game_histories[p] is not None:
                                self.put_last_trajectory(p, last_game_histories, last_game_priorities, game_histories)

                            # calculate priority
                            priorities = self.get_priorities(p, pred_values_lst, search_values_lst)

                            # save block trajectory
                            last_game_histories[p] = game_histories[p]
                            last_game_priorities[p] = priorities

                            # new block trajectory
                            game_histories[p] = GameHistory(env.action_space_size(), max_length=self.config.history_length,
                                                            config=self.config)
                            game_histories[p].init(stack_obs_windows[p])
                    '''
                    for p in prev_acting_agents:
                        # clip the reward
                        if dones[p]:
                            obs[p] = np.zeros(env.obs_shape).astype(int)
                        prev_game_histories[p].append(action[p], obs[p], clip_reward)
                        total_transitions += 1

                        # fresh stack windows
                        del stack_obs_windows[p][0]
                        stack_obs_windows[p].append(obs[p])
                    '''    
                    for player in list(dones.keys()):
                        # reset env if finished
                        if dones[player] and player in live_actors and player not in dead_actors:
                            dead_actors.append(player)
                            done_cnt += 1
                            for m_num, player_list in actor_model_dict.items():
                                if player in player_list:
                                    curr_model = m_num
                                    break
                            # pad over last block trajectory
                            if last_game_histories[player] is not None:
                                self.put_last_trajectory(player, last_game_histories, last_game_priorities, game_histories, curr_model)

                            # store current block trajectory
                            priorities = self.get_priorities(player, pred_values_lst, search_values_lst)
                                                  
                            game_histories[player].game_over()

                            self.put([game_histories[player], priorities], curr_model)
                            #print('rank ' + str(self.rank) + ', saving match from ' + player + ', for model ' + str(curr_model) + ', dones count ' + str(done_cnt))
                            self.free(curr_model)
                            
                            del game_histories[player]
                        #if player in prev_actors and dones[player]:
                        #    del prev_game_histories[player]
                        #    done_cnt += 1
                        if player in random_actors and dones[player] and player not in dead_actors:
                            done_cnt += 1
                    gc.collect()
                if self.config.num_models > 1:
                    print_rewards = ""
                    for m_num, ps in actor_model_dict.items():
                        res = []
                        for p in ps:
                            res.append(np.sum(rewards[p]))
                        print_rewards = print_rewards + "model {}: {}, ".format(m_num, np.mean(res))
                        
                    print("rank: {}, ".format(self.rank)+print_rewards)
                

                '''
                            # reset the finished env and new a env
                            envs[i].close()
                            init_obs = envs[i].reset()
                            game_histories[i] = GameHistory(env.env.action_space, max_length=self.config.history_length,
                                                            config=self.config)
                            
                            last_game_histories[player] = None
                            last_game_priorities[player] = None
                            stack_obs_windows[i] = [init_obs for _ in range(self.config.stacked_observations)]
                            game_histories[i].init(stack_obs_windows[i])
                            
                            # log
                            self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                            self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                            self_play_rewards += eps_reward_lst[i]
                            self_play_ori_rewards += eps_ori_reward_lst[i]
                            self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                            self_play_moves += eps_steps_lst[i]
                            self_play_episodes += 1

                            pred_values_lst[i] = []
                            search_values_lst[i] = []
                            # end_tags[i] = False
                            eps_steps_lst[i] = 0
                            eps_reward_lst[i] = 0
                            eps_ori_reward_lst[i] = 0
                            visit_entropies_lst[i] = 0
                            
                '''

                '''
                for p in step_live_agents:

                    if False: #dones[i]:
                        # pad over last block trajectory
                        if last_game_histories[p] is not None:
                            self.put_last_trajectory(p, last_game_histories, last_game_priorities, game_histories)

                        # store current block trajectory
                        priorities = self.get_priorities(p, pred_values_lst, search_values_lst)
                        game_histories[p].game_over()

                        self.put((game_histories[p], priorities))
                        self.free()

                        
                        self_play_rewards_max = max(self_play_rewards_max, eps_reward_lst[i])
                        self_play_moves_max = max(self_play_moves_max, eps_steps_lst[i])
                        self_play_rewards += eps_reward_lst[i]
                        self_play_ori_rewards += eps_ori_reward_lst[i]
                        self_play_visit_entropy.append(visit_entropies_lst[i] / eps_steps_lst[i])
                        self_play_moves += eps_steps_lst[i]
                        self_play_episodes += 1
                        
                    else:
                        # if the final game history is not finished, we will not save this data.
                        total_transitions -= len(game_histories[i])
                '''
                '''
                # logs
                visit_entropies = np.array(self_play_visit_entropy).mean()
                visit_entropies /= max_visit_entropy

                if self_play_episodes > 0:
                    log_self_play_moves = self_play_moves / self_play_episodes
                    log_self_play_rewards = self_play_rewards / self_play_episodes
                    log_self_play_ori_rewards = self_play_ori_rewards / self_play_episodes
                else:
                    log_self_play_moves = 0
                    log_self_play_rewards = 0
                    log_self_play_ori_rewards = 0

                other_dist = {}
                # send logs
                self.storage.set_data_worker_logs.remote(log_self_play_moves, self_play_moves_max,
                                                                log_self_play_ori_rewards, log_self_play_rewards,
                                                                self_play_rewards_max, _temperature.mean(),
                                                                visit_entropies, 0,
                                                                other_dist)
                '''