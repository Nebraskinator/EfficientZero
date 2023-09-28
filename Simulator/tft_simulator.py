import Simulator.config as config
import functools
import time
import gym
import numpy as np
from gym.spaces import MultiDiscrete, Box, Dict, Tuple
from Simulator import pool
from Simulator.player import Player as player_class
from Simulator.step_function import Step_Function
from Simulator.game_round import Game_Round
#from Simulator.observation import Observation
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env():
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer pettingzoo documentation.
    """
    local_env = TFT_Simulator(env_config=None)

    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    local_env = wrappers.OrderEnforcingWrapper(local_env)
    return local_env


#parallel_env = parallel_wrapper_fn(env)


class TFT_Simulator(AECEnv):
    metadata = {"is_parallelizable": True, "name": "tft-set4-v0"}

    def __init__(self, env_config, log=True):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id, self)
                        for player_id in range(config.NUM_PLAYERS)}
        #self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.render_mode = None
        self.live_agents = list(self.PLAYERS.keys())
        self.NUM_DEAD = 0
        self.num_players = config.NUM_PLAYERS
        self.log = log
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}
        self.image_channel = 52
        self.action_depth = 38
        self.obs_shape = (self.num_players, 14, 4, self.image_channel)
        self.step_function = Step_Function(self.pool_obj)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function, self.log)
        self.actions_taken = 0
        self.actions_taken_this_round = 0
        self.max_actions_per_round = config.ACTION_PER_TURN
        self.game_round.play_game_round()
        for key, p in self.PLAYERS.items():
            self.step_function.generate_shop(key, p)
        #self.step_function.generate_shop_vectors(self.PLAYERS)

        self.possible_agents = ["player_" + str(r) for r in range(config.NUM_PLAYERS)]
        self.agents = self.possible_agents[:]
        self.kill_list = []
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self.possible_agents[0]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"state_empty": False} for agent in self.agents}
        self.state = {agent: {} for agent in self.agents}
        self.public_observations = {player: self.PLAYERS[player].observation() for player in list(self.PLAYERS.keys())}
        self.actions = {agent: {} for agent in self.agents}

        # For MuZero
        # self.observation_spaces: Dict = dict(
        #     zip(self.agents,
        #         [Box(low=(-5.0), high=5.0, shape=(config.NUM_PLAYERS, config.OBSERVATION_SIZE,),
        #              dtype=np.float32) for _ in self.possible_agents])
        # )

        # For PPO
        '''
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Dict({
                        "tensor": Box(low=0, high=10.0, shape=(config.OBSERVATION_SIZE,), dtype=np.float64),
                        "mask": Tuple((MultiDiscrete(np.ones(6) * 2, dtype=np.int8), 
                                       MultiDiscrete(np.ones(5) * 2, dtype=np.int8),
                                       MultiDiscrete(np.ones(28) * 2, dtype=np.int8),
                                       MultiDiscrete(np.ones(9) * 2, dtype=np.int8),
                                       MultiDiscrete(np.ones(10) * 2, dtype=np.int8)))
                    }) for _ in self.agents
                ],
            )
        )
        '''

        # For MuZero
        # self.action_spaces = {agent: MultiDiscrete([config.ACTION_DIM for _ in range(config.NUM_PLAYERS)])
        #                       for agent in self.agents}

        # For PPO
        self.action_spaces = {agent: MultiDiscrete(np.ones(config.ACTION_DIM))
                              for agent in self.agents}
        super().__init__()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        return 14*4*38
    def action_space_size(self):
        return 14*4*38
    def check_dead(self):
        num_alive = 0
        for key, player in self.PLAYERS.items():
            if player:
                if player.health <= 0:
                    self.NUM_DEAD += 1
                    self.game_round.NUM_DEAD = self.NUM_DEAD
                    self.pool_obj.return_hero(player)
                    self.kill_list.append(key)
                else:
                    num_alive += 1
        return num_alive

    def observe(self, player_id):
        return self.observations[player_id]


    
    def generate_random_matchup(self):
        from Simulator import champion
        from Simulator.item_stats import items
        from Simulator.stats import COST
        from Simulator.origin_class_stats import tiers
        
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(2)}
        self.step_function = Step_Function(self.pool_obj)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function, self.log)
        r = np.random.choice([3, 4, 5, 7,
                                9, 10, 11,
                                13, 15, 16,
                                17, 19, 21,
                                22, 23, 25,
                                27, 28, 29,
                                31, 33, 34,
                                35, 37, 39,
                                40, 41, 43])
        self.game_round.current_round = r
        p0_level = np.clip(np.random.randint(np.clip(r // 9 - 2, 1, 9), np.clip(r // 9 + 2, 1, 10)), 1, 9)
        p1_level = np.clip(np.random.randint(np.clip(r // 9 - 2, 1, 9), np.clip(r // 9 + 2, 1, 10)), 1, 9)
        self.PLAYERS['player_0'].level = p0_level
        self.PLAYERS['player_0'].max_units = p0_level
        self.PLAYERS['player_0'].round = r
        self.PLAYERS['player_1'].level = p1_level
        self.PLAYERS['player_1'].max_units = p1_level
        self.PLAYERS['player_1'].round = r
        
        positions = [(x, y) for x in range(7) for y in range(4)]
        
        p0_positions = np.random.choice(np.arange(len(positions)), p0_level, replace=False)
        p1_positions = np.random.choice(np.arange(len(positions)), p1_level, replace=False)
        item_list = list(items.keys())
        
        for i in range(np.random.randint(10)):
            self.PLAYERS['player_0'].item_bench[i] = item_list[np.random.randint(9)]
        for i in range(np.random.randint(10)):
            self.PLAYERS['player_1'].item_bench[i] = item_list[np.random.randint(9)]
        
        for i, champ in enumerate(self.PLAYERS['player_0'].pool_obj.sample(self.PLAYERS['player_0'], p0_level)):
            if champ.endswith("_c"):
                c_shop = champ.split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(champ)
            for _ in range(np.random.randint(4 - a_champion.stars)):
                a_champion.golden()
            x, y = positions[p0_positions[i]]
            a_champion.x, a_champion.y = x, y
            self.PLAYERS['player_0'].board[x][y] = a_champion
        
        for i, champ in enumerate(self.PLAYERS['player_1'].pool_obj.sample(self.PLAYERS['player_1'], p1_level)):
            if champ.endswith("_c"):
                c_shop = champ.split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(champ)
            for _ in range(np.random.randint(4 - a_champion.stars)):
                a_champion.golden()
            x, y = positions[p1_positions[i]]
            a_champion.x, a_champion.y = x, y
            self.PLAYERS['player_1'].board[x][y] = a_champion
            
        for i, item in enumerate(self.PLAYERS['player_0'].item_bench):
            if item:
                x, y = positions[p0_positions[np.random.randint(len(p0_positions))]]
                self.PLAYERS['player_0'].move_item(i, x, y)
                
        for i, item in enumerate(self.PLAYERS['player_1'].item_bench):
            if item:
                x, y = positions[p1_positions[np.random.randint(len(p1_positions))]]
                self.PLAYERS['player_1'].move_item(i, x, y)
              
        def embed_champion(champ):
            embedding = np.zeros(162)                        
            try:
                c_index = list(COST.keys()).index(champ.name)
            except:
                c_index = 0
            embedding[c_index] = 255
            embedding[63 + champ.stars] = 255
            if champ.chosen:
                embedding[67] = 255
                embedding[68 + list(tiers).index(champ.chosen)] += 127
            for ind, item in enumerate(champ.items):
                embedding[94 + list(items).index(item)] += 84        
            if champ.target_dummy:
                embedding[153] = 255 
            for origin in champ.origin:
                embedding[68 + list(tiers).index(origin)] += 127
            
            lookup = {'AD' : [False, 0],
                      'crit_chance' : [False, 1],
                      'armor' : [False, 2],
                      'MR' : [False, 3],
                      'dodge' : [False, 4],
                      'health' : [False, 5],
                      'AS' : [True, 6],
                      'SP' : [True, 7],
                      }
            
            stat_emb = np.zeros(8)
            stat_emb[0] = champ.AD
            stat_emb[1] = champ.crit_chance
            stat_emb[2] = champ.armor
            stat_emb[3] = champ.MR
            stat_emb[4] = champ.dodge
            stat_emb[5] = champ.health
            stat_emb[6] = champ.AS
            stat_emb[7] = champ.SP
            
            for item in champ.items:
                for key, value in items[item].items():
                    if key in lookup:
                        if lookup[key][0]:
                            stat_emb[lookup[key][1]] *= value
                        else:
                            stat_emb[lookup[key][1]] += value
            
            stat_emb[0] = np.clip(stat_emb[0] / 2, 0, 255)
            stat_emb[1] = np.clip(stat_emb[1] * 255, 0, 255)
            stat_emb[2] = np.clip(stat_emb[2], 0, 255)
            stat_emb[3] = np.clip(stat_emb[3], 0, 255)
            stat_emb[4] = np.clip(stat_emb[4] * 400, 0, 255)
            stat_emb[5] = np.clip(stat_emb[5] * (255/6000), 0, 255)
            stat_emb[6] = np.clip(stat_emb[6] * 51, 0, 255)
            stat_emb[7] = np.clip(stat_emb[7] * 128, 0, 255)
            
            embedding[154:] = stat_emb
            
            return embedding

        obs = []
        for p in list(self.PLAYERS.values()):
            ob = np.zeros((7, 4, 162))
            for x in range(7):
                for y in range(4):
                    if p.board[x][y]:
                        ob[x, y, :] = embed_champion(p.board[x][y]) / 255
            obs.append(ob)
        
        r_obs = np.zeros(50)
        r_obs[r] = 1
        
        self.game_round.start_round()
        self.game_round.play_game_round()
        
        hp = self.PLAYERS['player_0'].reward / 20
        return obs[0], obs[1], r_obs, hp
    
    def generate_matchup(self, p1_vector, p2_vector, r, stars=1):
        from Simulator import champion
        from Simulator.item_stats import items
        from Simulator.stats import COST
        from Simulator.origin_class_stats import tiers
        
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id)
                        for player_id in range(2)}
        self.step_function = Step_Function(self.pool_obj)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function, self.log)

        self.game_round.current_round = r
        p0_level = p1_vector[0]
        p1_level = p2_vector[0]
        self.PLAYERS['player_0'].level = p0_level
        self.PLAYERS['player_0'].max_units = p0_level
        self.PLAYERS['player_0'].round = r
        self.PLAYERS['player_1'].level = p1_level
        self.PLAYERS['player_1'].max_units = p1_level
        self.PLAYERS['player_1'].round = r
        
        positions = [(x, y) for x in range(7) for y in range(4)]
        
        p0_positions = p1_vector[1]
        p1_positions = p2_vector[1]
        item_list = list(items.keys())
        
        for i, b in enumerate(p1_vector[2]):
            self.PLAYERS['player_0'].item_bench[i] = item_list[b]
        for i, b in enumerate(p2_vector[2]):
            self.PLAYERS['player_1'].item_bench[i] = item_list[b]
        
        for i, champ in enumerate(p1_vector[3]):
            if champ.endswith("_c"):
                c_shop = champ.split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(champ, stars=stars)
            x, y = positions[p0_positions[i]]
            a_champion.x, a_champion.y = x, y
            self.PLAYERS['player_0'].board[x][y] = a_champion
        
        for i, champ in enumerate(p2_vector[3]):
            if champ.endswith("_c"):
                c_shop = champ.split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
            else:
                a_champion = champion.champion(champ, stars=stars)
            x, y = positions[p1_positions[i]]
            a_champion.x, a_champion.y = x, y
            self.PLAYERS['player_1'].board[x][y] = a_champion
            
        for i, item in enumerate(self.PLAYERS['player_0'].item_bench):
            if item:
                x, y = positions[p1_vector[4][i]]
                self.PLAYERS['player_0'].move_item(i, x, y)
                
        for i, item in enumerate(self.PLAYERS['player_1'].item_bench):
            if item:
                x, y = positions[p2_vector[4][i]]
                self.PLAYERS['player_1'].move_item(i, x, y)
              
        
        self.game_round.start_round()
        self.game_round.play_game_round()
        
        hp = self.PLAYERS['player_0'].reward
        return hp


    def generate_public_observations(self):
        for player in self.PLAYERS:
            pass

    def reset(self, seed=None, options=None):
        self.pool_obj = pool.pool()
        self.PLAYERS = {"player_" + str(player_id): player_class(self.pool_obj, player_id, self)
                        for player_id in range(config.NUM_PLAYERS)}
        #self.game_observations = {"player_" + str(player_id): Observation() for player_id in range(config.NUM_PLAYERS)}
        self.live_agents = list(self.PLAYERS.keys())
        self.NUM_DEAD = 0
        self.previous_rewards = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.step_function = Step_Function(self.pool_obj)
        self.game_round = Game_Round(self.PLAYERS, self.pool_obj, self.step_function, self.log)
        self.actions_taken = 0
        self.game_round.play_game_round()
        for key, p in self.PLAYERS.items():
            self.step_function.generate_shop(key, p)
        #self.step_function.generate_shop_vectors(self.PLAYERS)

        self.agents = self.possible_agents.copy()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.infos = {agent: {"state_empty": False} for agent in self.agents}
        self.actions = {agent: {} for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.observations = {agent: {} for agent in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        super().__init__()
        [p.update_opponent_observations() for p in list(self.PLAYERS.values())]
        self.public_observations = {player: self.PLAYERS[player].observation() for player in list(self.PLAYERS.keys())}
        obs_dict = {player: self.make_observation(player) for player in list(self.PLAYERS.keys()) if self.PLAYERS[player].is_alive}
        taking_actions_dict = {player: self.PLAYERS[player].taking_actions for player in list(self.PLAYERS.keys())}
        action_masks = {player: self.PLAYERS[player].generate_action_mask_single() for player in list(self.PLAYERS.keys())}

        return obs_dict, taking_actions_dict, action_masks

    def render(self):
        ...

    def close(self):
        self.reset()

    def get_opponent_observations(self, player):
        ret = {}
        for p in list(self.PLAYERS.keys()):
            if self.PLAYERS[p] != player:
                if self.PLAYERS[p].is_alive:
                    ret[p] = self.PLAYERS[p].observation(False)
                else:
                    ret[p] = self.PLAYERS[player].empty_observation()
        return ret            

    def make_observation(self, player):
        obs = np.zeros(self.obs_shape).astype('uint8')
        obs[0] = self.PLAYERS[player].observation(True)
        cnt = 1
        for p in list(self.PLAYERS.keys()):
            if p != player:
                if self.PLAYERS[p].is_alive:
                    obs[cnt] = self.PLAYERS[player].opponent_public_obs[p].copy()
                cnt += 1
        return obs

    def step(self, action):
        for player in list(action.keys()):
            if self.PLAYERS[player].is_alive and self.PLAYERS[player].taking_actions:
                self.PLAYERS[player].take_action_single(action[player])
        self.actions_taken_this_round += 1
        dones = {p: False for p in self.live_agents}
        if self.actions_taken_this_round >= self.max_actions_per_round:
            self.actions_taken_this_round = 0
            self.game_round.play_game_round()
            self.public_observations = {player: self.PLAYERS[player].observation() for player in self.live_agents}
            for p in self.live_agents:
                if self.PLAYERS[p].health <= 0:
                    self.PLAYERS[p].is_alive = False
                    self.live_agents.remove(p)
                    self.PLAYERS[p].lost_game(len(self.live_agents))
                    self.PLAYERS[p].taking_actions = False
                    dones[p] = True
                else:
                    self.PLAYERS[p].taking_actions = True
                    self.PLAYERS[p].update_opponent_observations()
            if any(dones.values()):
                self.game_round.update_players({p: agent for p, agent in self.PLAYERS.items() if p in self.live_agents})
            self.game_round.start_round()
        
        obs_dict = {player: self.make_observation(player) for player in self.live_agents}
        if len(self.live_agents) <= 1:
            for p in self.live_agents:
                dones[p] = True
                self.PLAYERS[p].won_game()
        rewards_dict = {player: self.PLAYERS[player].reward for player in list(self.PLAYERS.keys())}
        for p in list(self.PLAYERS.keys()):
            self.PLAYERS[p].reward = 0
        taking_actions_dict = {player: self.PLAYERS[player].taking_actions for player in list(self.PLAYERS.keys())}
        action_masks = {p: self.PLAYERS[p].generate_action_mask_single() for p in list(self.PLAYERS.keys())}
        return obs_dict, rewards_dict, taking_actions_dict, dones, action_masks
            
            
    
    def step_old(self, action):
        # step for dead agents
        if self.terminations[self.agent_selection]:
            self._was_dead_step(action)
            return
        action = np.asarray(action)
        if action.ndim == 0:
            self.step_function.action_controller(action, self.PLAYERS[self.agent_selection], self.PLAYERS,
                                                 self.agent_selection, self.game_observations)
        elif action.ndim == 1:
            self.step_function.batch_2d_controller(action, self.PLAYERS[self.agent_selection], self.PLAYERS,
                                                   self.agent_selection, self.game_observations)

        # if we don't use this line, rewards will compound per step
        # (e.g. if player 1 gets reward in step 1, he will get rewards in steps 2-8)
        self._clear_rewards()
        self.infos[self.agent_selection] = {"state_empty": self.PLAYERS[self.agent_selection].state_empty()}

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        for agent in self.agents:
            self.observations[agent] = self.game_observations[agent].observation(
                agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector)

        # Also called in many environments but the line above this does the same thing but better
        # self._accumulate_rewards()
        if self._agent_selector.is_last():
            self.actions_taken += 1

            # If at the end of the turn
            if self.actions_taken >= config.ACTIONS_PER_TURN:
                # Take a game action and reset actions taken
                self.actions_taken = 0
                self.game_round.play_game_round()

                # Check if the game is over
                if self.check_dead() <= 1 or self.game_round.current_round > 48:
                    # Anyone left alive (should only be 1 player unless time limit) wins the game
                    for player_id in self.agents:
                        if self.PLAYERS[player_id] and self.PLAYERS[player_id].health > 0:
                            self.PLAYERS[player_id].won_game()
                            self.rewards[player_id] = 35 + self.PLAYERS[player_id].reward
                            self._cumulative_rewards[player_id] = self.rewards[player_id]
                            self.PLAYERS[player_id] = None  # Without this the reward is reset

                    self.terminations = {a: True for a in self.agents}

                self.infos = {a: {"state_empty": False} for a in self.agents}

                _live_agents = self.agents[:]
                for k in self.kill_list:
                    self.terminations[k] = True
                    _live_agents.remove(k)
                    self.rewards[k] = (3 - len(_live_agents)) * 10 + 5 + self.PLAYERS[k].reward
                    self._cumulative_rewards[k] = self.rewards[k]
                    self.PLAYERS[k] = None
                    self.game_round.update_players(self.PLAYERS)

                if len(self.kill_list) > 0:
                    self._agent_selector.reinit(_live_agents)
                self.kill_list = []

                if not all(self.terminations.values()):
                    self.game_round.start_round()

                    for agent in _live_agents:
                        self.observations[agent] = self.game_observations[agent].observation(
                            agent, self.PLAYERS[agent], self.PLAYERS[agent].action_vector)

            for player_id in self.PLAYERS:
                if self.PLAYERS[player_id]:
                    self.rewards[player_id] = self.PLAYERS[player_id].reward
                    self._cumulative_rewards[player_id] = self.rewards[player_id]

        # I think this if statement is needed in case all the agents die to the same minion round. a little sad.
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        # Probably not needed but doesn't hurt?
        self._deads_step_first()
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos
    