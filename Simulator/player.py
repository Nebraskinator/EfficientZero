import math
import time
import numpy as np
import random
from Simulator import champion, origin_class
import Simulator.utils as utils
import Simulator.config as config
from Simulator.item_stats import basic_items, item_builds, thieves_gloves_items, \
    starting_items, trait_items, uncraftable_items, items

from Simulator.stats import COST
from Simulator.pool_stats import cost_star_values
from Simulator.origin_class_stats import tiers, fortune_returns
from Simulator.origin_class_stats import origin_class as origin_dict
from math import floor
from itertools import permutations

DEBUG = False

"""
Description - This is the base player class
              Stores all values relevant to an individual player in the game
Inputs      - pool_pointer: Pool object pointer
                pointer to the pool object, used for updating the pool on buy and sell commands
              player_num: Int
                An identifier for the player, used in match_making for combats
"""


class Player:
    # Explanation - We may switch to a class config for the AI side later so separating the two now is highly useful.
    def __init__(self, pool_pointer, player_num, simulator):

        self.simulator = simulator
        self.opponent_public_obs = {}
                
        self.gold = 0
        self.level = 1
        self.exp = 0
        self.health = 100
        self.player_num = player_num
        
        self.ai = PlayerAI(self)

        self.win_streak = 0  # For purposes of gold generation at start of turn
        self.loss_streak = 0  # For purposes of gold generation at start of turn
        self.fortune_loss_streak = 0  # For purposes of gold generation if fortune trait is active

        # array of champions, since order does not matter, can be unordered list
        self.bench = [None for _ in range(9)]
        # Champion array, this is a 7 by 4 array.
        self.board = [[None for _ in range(4)] for _ in range(7)]
        # List of items, there is no object for this so this is a string array
        self.item_bench = [None for _ in range(10)]
        
        self.shop = [None for _ in range(5)]
        
        self.selection = None
        
        self.taking_actions = True
        self.is_alive = True
        
        self.actions_taken_this_round = 0
        self.actions_per_round = config.ACTION_PER_TURN

        # opponent and opponent_board not currently used
        # Leaving here in case we want to add values to the observation that include previous opponent
        self.opponent = None  # Other player, player object
        self.opponent_board = None  # Other player's board for combat, not sure if I will use this.
        self.chosen = False  # Does this player have a chosen unit already
        self.log = []

        self.action_history = []

        # I need to comment how this works.
        self.triple_catalog = []
        self.num_units_in_play = 0
        self.max_units = 1
        self.exp_cost = 4
        self.exp_per_buy = 4
        self.round = 0
        self.action_mask = np.zeros(56)

        # This could be in a config file, but we could implement something that alters the
        # Amount of gold required to level that differs player to player
        self.level_costs = [0, 2, 2, 6, 10, 20, 36, 56, 80, 100]
        self.max_level = 9

        # 2 spot for each item(2 component) 10 slots
        #self.item_vector = np.zeros(config.MAX_BENCH_SPACE * 6)

        # This time we only need 5 bits total
        #self.chosen_vector = np.zeros(5)

        # player related info split between what other players can see and what they can't see
        #self.player_public_vector = np.zeros(7)
        #self.player_private_vector = np.zeros(16)

        # Encoding board as an image, so we can run convolutions on it.
        #self.board_vector = np.zeros(728)  # 26 size on each unit, 28 squares
        #self.bench_vector = np.zeros(config.BENCH_SIZE * config.CHAMP_ENCODING_SIZE)

        #self.decision_mask = np.ones(6, dtype=np.int8)
        #self.shop_mask = np.ones(5, dtype=np.int8)
        # locations of champions on the board, 1 for spot taken 0 for not
        #self.board_mask = np.ones(28, dtype=np.int8)
        # locations of units that are not champions on the board
        #self.dummy_mask = np.ones(28, dtype=np.int8)
        # locations of champions on the bench, 1 for spot taken 0 for not
        #self.bench_mask = np.ones(9, dtype=np.int8)
        #self.item_mask = np.ones(10, dtype=np.int8)
        # random useful masks
        # util_mask[0] = 0 if board is full, 1 if not
        # util_mask[1] = 0 if bench is full, 1 if not
        # util_mask[2] = 0 if item_bench is full, 1 if not
        #self.util_mask = np.ones(3, dtype=np.int8)
        # thieves_glove_mask = 1 if unit has thieves glove, 0 if not
        #self.thieves_glove_mask = np.zeros(37, dtype=np.int8)
        # glove_item_mask = 1 if unit has sparring glove + item, 0 if not
        #self.glove_item_mask = np.zeros(37, dtype=np.int8)
        # glove_mask = 1 if there is a sparring glove in that item slot, 0 if not
        #self.glove_mask = np.zeros(10, dtype=np.int8)
        self.shop_costs = np.ones(5)

        # Using this to track the reward gained by each player for the AI to train.
        self.reward = 0.0

        # cost to refresh
        self.refresh_cost = 2

        # reward levers
        self.refresh_reward = 0
        self.minion_count_reward = 0
        self.mistake_reward = 0
        self.level_reward = 0
        self.item_reward = 0
        self.won_game_reward = 10
        self.prev_rewards = 0
        self.damage_reward = 0.4
        self.excess_gold_reward = 0

        # Everyone shares the pool object.
        # Required for buying champions to and from the pool
        self.pool_obj = pool_pointer

        # Boolean for fought this round or not
        self.combat = False
        # List of team compositions
        self.team_composition = origin_class.game_compositions[self.player_num]
        # List of tiers of each trait.
        self.team_tiers = origin_class.game_comp_tiers[self.player_num]

        # An array to record match history
        self.match_history = []

        self.start_time = time.time_ns()

        # Putting this here to show the next possible opponent
        self.possible_opponents = {"player_" + str(player_id): config.MATCHMAKING_WEIGHTS
                                   for player_id in range(config.NUM_PLAYERS)}
        self.possible_opponents["player_" + str(self.player_num)] = -1
        self.opponent_options = {"player_" + str(player_id): 0 for player_id in range(config.NUM_PLAYERS)}

        self.kayn_turn_count = 0
        self.kayn_transformed = False
        self.kayn_form = None

        self.thieves_gloves_loc = []

        self.action_vector = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        self.current_action = 0
        self.action_complete = False
        self.action_values = []

        # Start with two copies of each item in your item pool
        self.item_pool = []
        self.refill_item_pool()
        self.refill_item_pool()

        # Context For Loot Orbs
        self.orb_history = []

        # Call vector generation methods for first observation
        #self.generate_player_vector()
        #self.generate_board_vector()
        #self.generate_bench_vector()
        #self.generate_item_vector()
        #self.generate_chosen_vector()

    """
    Description - Main method used in buy_champion to add units to the bench
                  Treats buying a unit with a full bench as the same as buying and immediately selling it
    Inputs      - a_champion: Champion object
                    champion to be added to the bench
    Outputs     - True: successfully added to the bench
                  False: Either could not update triple catalog (some error), or the bench was full
    """
    # TODO: Verify that the units received carousel round are not bugging out.
    #       Check the logs in the simulator discord channel
    def add_to_bench(self, a_champion, from_carousel=False):
        # try to triple second
        golden, triple_success = self.update_triple_catalog(a_champion)
        if not triple_success:
            self.print("Could not update triple catalog for champion " + a_champion.name)
            if DEBUG:
                print("Could not update triple catalog for champion " + a_champion.name)
            return False
        if golden:
            return True
        if self.bench_full():
            self.sell_champion(a_champion, field=False)
            if not from_carousel:
                self.reward += self.mistake_reward
                self.print("attempted to buy champion while bench is full")
                if DEBUG:
                    print("Trying to buy a unit with bench full")
                return False
            return True
        bench_loc = self.bench_vacancy()
        self.bench[bench_loc] = a_champion
        a_champion.bench_loc = bench_loc
        if a_champion.chosen:
            self.print("Adding chosen champion {} of type {}".format(a_champion.name, a_champion.chosen))
            self.chosen = a_champion.chosen
        self.print("Adding champion {} with items {} to bench".format(a_champion.name, a_champion.items))
        if self.bench[bench_loc].items and self.bench[bench_loc].items[0] == 'thieves_gloves':
            self.thieves_gloves_loc.append([bench_loc, -1])
            self.thieves_gloves(bench_loc, -1)
        #self.generate_bench_vector()
        #self.generate_item_vector()
        return True

    """
    Description - Deselect an item or champion
    Inputs      - location_clicked: string
                    location where selection was placed (board, bench, item_bench, or shop)
    """

    def clear_selection(self, location_clicked):
        if self.selection['origin'] == 'board':
            move = self.selection['selected'].name
        elif self.selection['origin'] == 'item_bench':
            move = self.selection['selected']
        elif self.selection['origin'] == 'bench':
            move = self.selection['selected'].name
        else:
            move = 'unknown'        
        self.print("clearing selection: {} from {} position {} after clicking {}".format(move, self.selection['origin'], self.selection['coord'], location_clicked))
        self.selection = None
        
    """
    Description - Main method used to select an item or champion
    Inputs      - selection: Champion object or item string
                    champion or item to be selected
                - origin: string
                    where the selected object is (board, bench, or item_bench)
                - origin_coord: int or list(int)
                    coordinates of the selected object
    """
        
    def select_champion_or_item(self, selection, origin, origin_coord):
        self.selection = {'selected' : selection,
                          'origin' : origin,
                          'coord' : origin_coord}
        if origin == 'board':
            move = selection.name
        elif origin == 'bench':
            move = selection.name
        elif origin == 'item_bench':
            move = selection
        else:
            move = 'unknown'
        self.print("selected {} from {} position {}".format(move, origin, origin_coord))
        
    """
    Description - Used to log actions which do not change the game state
    Inputs      - area: string
                    location which was selected (board, bench, item_bench, or shop)
    """

    def actionless_click(self, area):
        self.print("clicked {} but no action was possible".format(area))
        
    """
    Description - Used to flag player as passing turns until after next combat
    """
    # TODO: change self play to ignore passing players during MCTS
    def pass_turn(self):
        self.taking_actions = False
        self.print("passing turn; no actions until next combat")
        
    """
    Description - Used to flag player as passing turns until after next combat
    """
    # TODO: change self play to ignore passing players during MCTS
    def pass_action(self):
        self.update_opponent_observations()
        self.print("no action, updating opponent observations")

    """
    Description - Single player step function to take actions
    Inputs      - action: int
                    0 - 27: board positions
                    28 - 36: bench positions
                    37 - 46: item_bench positions
                    47 - 51: shop positions
                    52: buy exp
                    53: refresh shop
                    54: pass turn
                    >54: ignored
    """

    def take_action(self, action):
        if 1 <= action <= 28:
            t = action - 1
            x, y = t % 7, t // 7
            if self.selection:                                 
                if self.selection['origin'] == 'item_bench':
                    self.move_item_to_board(self.selection['coord'], x, y)
                elif self.selection['origin'] == 'board':
                    if [x, y] != self.selection['coord']:
                        self.move_board_to_board(self.selection['coord'][0], self.selection['coord'][1], x, y)
                        self.print("champions in play: "+str(self.num_units_in_play))
                elif self.selection['origin'] == 'bench':
                    self.move_bench_to_board(self.selection['coord'], x, y)
                    self.print("champions in play: "+str(self.num_units_in_play))
                else:
                    self.reward += self.mistake_reward
                    self.actionless_click('board')
                self.clear_selection('board')
            elif self.board[x][y]:
                self.select_champion_or_item(self.board[x][y], 'board', [x, y])
            else:
                self.reward += self.mistake_reward
                self.actionless_click('board')
        elif 29 <= action <= 37:
            x = action - 29
            if self.selection:
                if self.selection['origin'] == 'item_bench':
                    self.move_item_to_bench(self.selection['coord'], x)
                elif self.selection['origin'] == 'board':
                    self.move_board_to_bench(self.selection['coord'][0], self.selection['coord'][1], x)
                    self.print("champions in play: "+str(self.num_units_in_play))
                elif self.selection['origin'] == 'bench':
                    self.reward += self.mistake_reward
                    self.actionless_click('bench')
                self.clear_selection('bench')
            elif self.bench[x]:
                self.select_champion_or_item(self.bench[x], 'bench', x)
            else:
                self.reward += self.mistake_reward
                self.actionless_click('bench')
        elif 38 <= action <= 47:
            x = action - 38
            if self.selection: 
                self.reward += self.mistake_reward                                
                self.clear_selection('item bench')
            elif self.item_bench[x]:
                self.select_champion_or_item(self.item_bench[x], 'item_bench', x)
            else:
                self.reward += self.mistake_reward
                self.actionless_click('item_bench')
        elif 48 <= action <= 52:
            if self.selection:
                if self.selection['origin'] == 'board':
                    self.sell_champion(self.selection['selected'], field=True)
                    self.print("champions in play: "+str(self.num_units_in_play))
                elif self.selection['origin'] == 'bench':
                    self.sell_from_bench(self.selection['coord'])
                    self.print("champions in play: "+str(self.num_units_in_play))
                elif self.selection['origin'] == 'item_bench':
                    self.reward += self.mistake_reward
                    self.actionless_click('shop')
                self.clear_selection('shop')
            else:
                x = action - 48
                success = False
                if not self.shop[x]:
                    self.reward += self.mistake_reward
                    self.actionless_click('shop')
                if self.shop[x] and self.shop[x].endswith("_c"):
                    c_shop = self.shop[x].split('_')
                    a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
                    success = self.buy_champion(a_champion)
                    self.print("champions in play: "+str(self.num_units_in_play))
                elif self.shop[x]:
                    a_champion = champion.champion(self.shop[x])
                    success = self.buy_champion(a_champion)
                    self.print("champions in play: "+str(self.num_units_in_play))
                if success:
                    self.shop[x] = None
        elif action == 53:
            if self.refresh():
                self.shop = self.pool_obj.sample(self, 5)
                self.printShop(self.shop)
        elif action == 54:
            self.buy_exp()
        elif action == 55:
            exp_to_level = 0
            if self.level < self.max_level:
                exp_to_level = self.level_costs[self.level] - self.exp
                num_buys = math.ceil(exp_to_level / self.exp_per_buy)
                self.buy_exp(num_buys)
        elif action == 0:
            self.pass_action()
        self.actions_taken_this_round += 1

    def take_action_single(self, action):
        self.action_history.append(action)
        x, y, z = action // 38 // 4, action // 38 % 4, action % 38
        self.print("action: {}, x: {}, y: {}, z: {}".format(action, x, y, z))
        if x < 7:
            if self.board[x][y]:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                    if x != dest_x or y != dest_y:
                        self.move_board_to_board(x, y, dest_x, dest_y)
                elif z < 37:
                    x_bench = z - 28
                    self.move_board_to_bench(x, y, x_bench)
                else:
                    self.sell_champion(self.board[x][y], field=True)
        elif 7 <= x <= 9 and (x-7) * 4 + y < 9:
            x_bench = (x-7) * 4 + y
            if x_bench < len(self.bench) and self.bench[x_bench]:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                    self.move_bench_to_board(x_bench, dest_x, dest_y)
                elif z < 37:
                    dest_x = z - 28
                    if x_bench != dest_x:
                        self.move_bench_to_bench(x_bench, dest_x)
                else:
                    self.sell_from_bench(x_bench)      
        elif 9 <= x <= 10 and (x-9) * 4 + y - 1 < 5:
            x_shop = (x-9) * 4 + y - 1
            success = False
            if not self.shop[x_shop]:
                self.reward += self.mistake_reward
                self.actionless_click('shop')
            elif self.shop[x_shop] and self.shop[x_shop].endswith("_c"):
                c_shop = self.shop[x_shop].split('_')
                a_champion = champion.champion(c_shop[0], chosen=c_shop[1], itemlist=[])
                success = self.buy_champion(a_champion)
                self.print("champions in play: "+str(self.num_units_in_play))
            elif self.shop[x_shop]:
                a_champion = champion.champion(self.shop[x_shop])
                success = self.buy_champion(a_champion)
                self.print("champions in play: "+str(self.num_units_in_play))
            if success:
                self.shop[x_shop] = None
        elif 10 <= x <= 12:
            x_itembench = (x-10) * 4 + y - 2
            if self.item_bench[x_itembench]:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                    self.move_item_to_board(x_itembench, dest_x, dest_y)
                elif z < 37:
                    dest_x = z - 28
                    self.move_item_to_bench(x_itembench, dest_x)

        elif x == 13 and y == 0 and self.round > 1:
            if self.refresh():
                self.shop = self.pool_obj.sample(self, 5)
                self.printShop(self.shop)
        elif x == 13 and y == 1 and self.round > 1:
            self.buy_exp()

        elif x == 13 and y == 3:
            self.pass_action() 
        self.actions_taken_this_round += 1
        '''
        elif x == 5 and y == 6 and self.round > 1:
            exp_to_level = 0
            if self.level < self.max_level:
                exp_to_level = self.level_costs[self.level] - self.exp
                num_buys = math.ceil(exp_to_level / self.exp_per_buy)
                self.buy_exp(num_buys) 
        '''
            
    def generate_action_mask_single(self):
        mask = np.zeros((14, 4, 38))
        for y in range(4):
            for x in range(7):
                if self.board[x][y]:
                    mask[x, y, :] = 1
                    mask[x, y, x*4 + y] = 0
        for x_bench in range(len(self.bench)):
            if self.bench[x_bench]:
                x, y = x_bench // 4 + 7, x_bench % 4
                if self.num_units_in_play < self.max_units:
                    mask[x, y, :] = 1                    
                else:
                    mask[x, y, 28:] = 1                    
                    for board_y in range(4):
                        for board_x in range(7):
                            if self.board[board_x][board_y]:
                                mask[x, y, board_x*4 + board_y] = 1
                mask[x, y, 28 + x_bench] = 0
        for x_itembench, item in enumerate(self.item_bench):
            if item:
                x, y = (x_itembench + 2) // 4 + 10, (x_itembench + 2) % 4 
                for board_y in range(4):
                    for board_x in range(7):
                        if self.board[board_x][board_y] and len(self.board[board_x][board_y].items) < 3:
                            mask[x, y, board_x*4 + board_y] = 1
                for x_bench in range(len(self.bench)):
                    if self.bench[x_bench] and len(self.bench[x_bench].items) < 3:
                        mask[x, y, 28 + x_bench] = 1
        for x_shop in range(5):
            if self.shop[x_shop]:
                x, y = (x_shop + 1) // 4 + 9, (x_shop + 1) % 4
                champ_name = self.shop[x_shop].split('_')[0]
                level = 1
                cost = COST[champ_name]
                if self.shop[x_shop].endswith('_c'):
                    cost = cost_star_values[cost - 1][1]
                    level = 2
                if cost <= self.gold:
                    if self.bench_full():
                        for trip in self.triple_catalog:
                            if trip['name'] == champ_name and trip['num'] == 2 and trip['level'] == level:
                                mask[x, y, 28] = 1
                    else:
                        mask[x, y, 28] = 1
        if self.gold >= 4:
            mask[-1, 1, 0] = 1
        if self.gold >= 2:
            mask[-1, 0, 0] = 1
        '''
        exp_to_level = 0
        if self.level < self.max_level:
            exp_to_level = self.level_costs[self.level] - self.exp
            num_buys = math.ceil(exp_to_level / self.exp_per_buy)
            if self.gold >= num_buys * self.exp_cost:
                mask[5, 6, 0] = 1
        '''
        mask[-1, 3, 0] = 1
        return np.reshape(mask, 14*4*38)
        

    def generate_action_mask(self):
        mask = np.zeros(56)
        for i in range(1, 29):
            t = i-1
            x, y = t % 7, t // 7            
            if self.selection:
                if self.selection['origin'] == 'board' and [x, y] != self.selection['coord']:
                    mask[i] = 1
                if self.selection['origin'] == 'bench':
                    if self.board[x][y] and not self.board[x][y].target_dummy:
                        mask[i] = 1
                    elif self.num_units_in_play < self.max_units:
                        mask[i] = 1
                if self.selection['origin'] == 'item_bench':
                    if self.board[x][y] and not self.board[x][y].target_dummy:
                        if self.selection['selected'] == 'champion_duplicator':
                            mask[i] = 1
                        elif self.selection['selected'] in ['magnetic_remover','reforger']:
                            if len(self.board[x][y].items) > 0:
                                mask[i] = 1
                        elif self.selection['selected'] == 'thieves_gloves':
                            if len(self.board[x][y].items) == 0:
                                mask[i] = 1
                        elif self.selection['selected'] in ['kayn_shadowassassin','kayn_rhast']:
                            if self.board[x][y].name == 'kayn':
                                mask[i] = 1
                        elif len(self.board[x][y].items) < 3:
                            mask[i] = 1
            elif self.board[x][y]:
                mask[i] = 1
        for i in range(29, 38):
            x_bench = i - 29
            if self.selection:
                if self.selection['origin'] == 'board' and not self.selection['selected'].target_dummy:
                    mask[i] = 1
                if self.selection['origin'] == 'bench':
                    continue
                if self.selection['origin'] == 'item_bench':
                    if self.bench[x_bench]:
                        if self.selection['selected'] == 'champion_duplicator':
                            mask[i] = 1
                        elif self.selection['selected'] in ['magnetic_remover','reforger']:
                            if len(self.bench[x_bench].items) > 0:
                                mask[i] = 1
                        elif self.selection['selected'] == 'thieves_gloves':
                            if len(self.bench[x_bench].items) == 0:
                                mask[i] = 1
                        elif self.selection['selected'] in ['kayn_shadowassassin','kayn_rhast']:
                            if self.bench[x_bench].name == 'kayn':
                                mask[i] = 1
                        elif len(self.bench[x_bench].items) < 3:
                            mask[i] = 1
            elif self.bench[x_bench]:
                mask[i] = 1
        for i in range(38, 48):
            if not self.selection:
                x = i - 38
                if self.item_bench[x]:
                    mask[i] = 1
        for i in range(48, 53):
            if self.selection:
                if self.selection['origin'] in ['board', 'bench'] and not self.selection['selected'].target_dummy:
                    mask[i] = 1
            else:
                x = i - 48
                if self.shop[x]:
                    cost = COST[self.shop[x].split('_')[0]]
                    if self.shop[x].endswith('_c'):
                        cost = cost_star_values[cost - 1][1]
                    if cost <= self.gold:
                        mask[i] = 1
        if self.gold >= 4:
            mask[54] = 1
        if self.gold >= 2:
            mask[53] = 1
        exp_to_level = 0
        if self.level < self.max_level:
            exp_to_level = self.level_costs[self.level] - self.exp
            num_buys = math.ceil(exp_to_level / self.exp_per_buy)
            if self.gold >= num_buys * self.exp_cost:
                mask[55] = 1
        mask[0] = 1
        self.action_mask = mask
                
            
                    

    """
    Description - Adds an item to the item_bench. 
    Inputs      - item: String
                    name of the item to add
    Outputs     - False if unsuccessful. Will happen if item_bench is full
    """
    # TODO: Create unit tests for when the item_bench is full.
    # TODO: Verify that the loot orbs are dropping a sufficient amount of items.
    #       Should be enough for 2 units to have full items in a game at least
    def add_to_item_bench(self, item):
        if self.item_bench_full(1):
            self.reward += self.mistake_reward
            if DEBUG:
                print("Failed to add item to item bench")
            return False
        bench_loc = self.item_bench_vacancy()
        self.item_bench[bench_loc] = item
        #if item == "sparring_gloves":
        #    self.glove_mask[bench_loc] = 1
        #else:
        #    self.glove_mask[bench_loc] = 0
        #self.generate_item_vector()

    """
    Description - Checks if the bench is full, updates util mask as well
    Outputs     - True: Bench is full
                  False: Bench is not full
    """
    def bench_full(self):
        for u in self.bench:
            if not u:
                #self.util_mask[1] = 1
                return False
        #self.util_mask[1] = 0
        return True

    """
    Description - Returns the spot on the champion bench where there is a vacancy
    Outputs     - Int or bool: location on bench where there is a vacancy, False otherwise.
    """
    # TODO: Unit test anywhere this gets called with a full bench to ensure correct behavior
    def bench_vacancy(self):
        for free_slot, u in enumerate(self.bench):
            if not u:
                return free_slot
        return False

    """
    Description - Buys a champion, deals with gold and various champion checks related to buying of the unit
    Inputs      - a_champion: Champion object
                    champion to be added to the bench
    Outputs     - True: Champion purchase successful
                  False: not enough gold to buy champion
    """
    def buy_champion(self, a_champion):
        if cost_star_values[a_champion.cost - 1][a_champion.stars - 1] > self.gold or a_champion.cost == 0:
            self.reward += self.mistake_reward
            if DEBUG:
                print("No gold to buy champion")
            return False
        self.gold -= cost_star_values[a_champion.cost - 1][a_champion.stars - 1]
        if a_champion.name == 'kayn':
            a_champion.kayn_form = self.kayn_form
        success = self.add_to_bench(a_champion)
        # Putting this outside success because when the bench is full. It auto sells the champion.
        # Which adds another to the pool and need this here to remove the fake copy from the pool
        self.pool_obj.update_pool(a_champion, -1)
        if success:
            # Leaving this out because the agent will learn to simply buy everything and sell everything
            # I want it to just buy what it needs to win rounds.
            # self.reward += 0.005 * cost_star_values[a_champion.cost - 1][a_champion.stars - 1]
            self.print("Spending gold on champion {}".format(a_champion.name) + " with cost = " +
                       str(cost_star_values[a_champion.cost - 1][a_champion.stars - 1])
                       + ", remaining gold " + str(self.gold) + " and chosen = " + str(a_champion.chosen))
            #self.generate_player_vector()
        # else:
        #     if self.player_num == 0:
        #         print("Did not buy champion successfully")
        return success

    """
    Description - Checks to make sure conditions are met to be able to buy exp then buys exp if met
    Outputs     - True: exp purchase successful
                  False: Not enough gold or already max level
    """
    def buy_exp(self, num=1):
        # if the player doesn't have enough gold to buy exp or is max level, give bad reward
        if self.gold < self.exp_cost * num or self.level == self.max_level:
            self.reward += self.mistake_reward
            #self.decision_mask[4] = 0
            if DEBUG:
                print("Did not have gold to buy_exp")
            return False
        self.gold -= num * self.exp_cost
        # self.reward += 0.02        
        self.exp += num * self.exp_per_buy
        self.print("exp to {} on level {}".format(self.exp, self.level))
        self.level_up()
        #self.generate_player_vector()
        return True

    """
    Description - Method used to optimize code by only calling generate vector on the vector needed
    Inputs      -
    """
    def decide_vector_generation(self, x):
        if x:
            self.generate_board_vector()
        else:
            self.generate_bench_vector()

    """
    Description - Handles end of turn actions like moving units from bench to free slots until max_unit_in_play hit.
                  This method also calls the print board, comp, and items
    """
    # TODO: Move the print board / comp / items / bench to the start round function so it will be more accurate when
    # TODO: Combat starts to change the player state. (for example, urgot in set 8)
    def end_turn_actions(self, fill_board=False):
        # autofill the board.
        if fill_board:
            num_units_to_move = self.max_units - self.num_units_in_play
            position_found = -1
            
            for _ in range(num_units_to_move):
                found_position = False
                for i in range(position_found + 1, len(self.bench)):
                    if self.bench[i]:
                        for x in range(len(self.board)):
                            for y in range(len(self.board[0])):
                                if self.board[x][y] is None:
                                    self.move_bench_to_board(i, x, y)
                                    found_position = True
                                    position_found = i
                                    break
                            if found_position:
                                break
                    if found_position:
                        break
        
        # update board to survive combat = False, will update after combat if they survived
        # update board to participated in combat
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                if self.board[x][y]:
                    self.board[x][y].participated_in_combat = True
                    self.board[x][y].survive_combat = False
        # update bench to did not participate in combat
        for x in range(len(self.bench)):
            if self.bench[x]:
                self.bench[x].participated_in_combat = False
                self.bench[x].survive_combat = False
        #self.generate_bench_vector()
        #self.generate_board_vector()
        #self.generate_player_vector()

    """
    Description - Finds locations to put azir's sandguards when he first gets put on the field
    Inputs      - Azir's x and y coordinates
    Outputs     - Two sets of coordinates
    """
    def find_azir_sandguards(self, azir_x, azir_y):
        coords_candidates = self.find_free_squares(azir_x, azir_y)
        x = 6
        y = 3
        while len(coords_candidates) < 2:
            hexes = self.find_free_squares(x, y)
            for hex in hexes:
                if hex not in coords_candidates:
                    coords_candidates.append(hex)
            x -= 1
            if x == -1:
                x = 6
                y -= 1

        for x in range(2):
            self.board[coords_candidates[x][0]][coords_candidates[x][1]] = champion.champion('sandguard',
                                    kayn_form=self.kayn_form, target_dummy=True)
        coords = [coords_candidates[0], coords_candidates[1]]
        return coords

    """
    Description - Finds free squares around a coordinate
    Inputs      - Coordinate
    Outputs     - Free squares surrounding the coordinate
    """
    def find_free_squares(self, x, y):
        if x < 0 or x > 6 or y < 0 or y > 3:
            return []
        directions = [
            [[+1, 0], [+1, +1], [0, -1],
             [0, +1], [-1, 0], [-1, +1]],
            [[+1, -1], [+1, 0], [0, -1],
             [0, +1], [-1, -1], [-1, 0]],
        ]

        parity = y & 1
        neighbors = []
        for c in directions[parity]:
            nY = c[0] + y
            nX = c[1] + x
            if (0 <= nY < 4 and 0 <= nX < 7) and not self.board[x][y]:
                neighbors.append([nX, nY])
        return neighbors

    """
    Description - Generates board vector. We use binary encoding. Each square on the board gets a 26 value encoding.
                  If there is no unit on the square, 0s will fill that position. Stars and cost are not binary
    """
    def generate_board_vector(self):
        for y in range(0, 4):
            # IMPORTANT TO HAVE THE X INSIDE -- Silver is not sure why but ok.
            for x in range(0, 7):
                # when using binary encoding (6 champ  + stars + chosen + 3 * 6 item) = 26
                champion_info_array = np.zeros(6 * 4 + 2)
                if self.board[x][y]:
                    # Check for target_dummy or azir sandguard
                    if self.board[x][y].target_dummy or self.board[x][y].overlord_coordinates is not None:
                        self.dummy_mask[7 * y + x] = 1

                    self.board_mask[7 * y + x] = 1
                    curr_champ = self.board[x][y]
                    c_index = list(COST.keys()).index(curr_champ.name)
                    champion_info_array[0:6] = utils.champ_binary_encode(c_index)
                    champion_info_array[6] = curr_champ.stars / 3
                    champion_info_array[7] = curr_champ.cost / 5
                    for ind, item in enumerate(curr_champ.items):
                        start = (ind * 6) + 7
                        finish = start + 6
                        i_index = []
                        if item in uncraftable_items:
                            i_index = list(uncraftable_items).index(item) + 1
                        elif item in item_builds.keys():
                            i_index = list(item_builds.keys()).index(item) + 1 + len(uncraftable_items)
                        champion_info_array[start:finish] = utils.item_binary_encode(i_index)
                else:
                    # Different from the board vector because it needs to match the MCTS encoder
                    self.board_mask[7 * y + x] = 0

                # Fit the area into the designated spot in the vector
                self.board_vector[x * 4 + y:x * 4 + y + 26] = champion_info_array

        if self.num_units_in_play == self.max_units:
            self.util_mask[0] = 0
        else:
            self.util_mask[0] = 1

    """
    Description - Generates the bench vector. The same encoding style for the board is used for the bench.
    """
    def generate_bench_vector(self):
        space = 0
        bench = np.zeros(config.BENCH_SIZE * config.CHAMP_ENCODING_SIZE)
        for x_bench in range(len(self.bench)):
            # when using binary encoding (6 champ  + stars + chosen + 3 * 6 item) = 26
            champion_info_array = np.zeros(6 * 4 + 2)
            if self.bench[x_bench]:
                curr_champ = self.bench[x_bench]
                c_index = list(COST.keys()).index(curr_champ.name)
                champion_info_array[0:6] = utils.champ_binary_encode(c_index)
                champion_info_array[6] = curr_champ.stars / 3
                champion_info_array[7] = curr_champ.cost / 5
                for ind, item in enumerate(curr_champ.items):
                    start = (ind * 6) + 7
                    finish = start + 6
                    i_index = []
                    if item in uncraftable_items:
                        i_index = list(uncraftable_items).index(item) + 1
                    elif item in item_builds.keys():
                        i_index = list(item_builds.keys()).index(item) + 1 + len(uncraftable_items)
                    champion_info_array[start:finish] = utils.item_binary_encode(i_index)
                #self.bench_mask[x_bench] = 1
            else:
                if x_bench == 9:
                    print("length of bench = {}".format(len(self.bench)))
                #self.bench_mask[x_bench] = 0
                space = 1
            bench[x_bench * config.CHAMP_ENCODING_SIZE:
                  x_bench * config.CHAMP_ENCODING_SIZE + config.CHAMP_ENCODING_SIZE] = champion_info_array
        self.bench_vector = bench
        #self.util_mask[1] = space

    """
    Description - Generates the chosen vector, this uses binary encoding of the index in possible chosen traits. 
    """
    def generate_chosen_vector(self):
        output_array = np.zeros(5)
        if self.chosen:
            i_index = list(self.team_composition.keys()).index(self.chosen)
            # This should update the item name section of the vector
            for z in range(5, 0, -1):
                if i_index > 2 * z:
                    output_array[5 - z] = 1
                    i_index -= 2 * z
        self.chosen_vector = output_array

    """
    Description - Generates the observation space of the player. 
                    Observation space is 12x12x164
                    See "TFT AI Notes" Google sheet for full description
    Inputs      - private: bool
                    indicates if the observation is public or private                    
    """
    
    
    def update_opponent_observations(self):
        self.opponent_public_obs = self.simulator.get_opponent_observations(self)
    
    def observation(self, private=False):
        
        # 161 length 
        item_encoding_size = 10
        champ_encoding_size = 22
        def encode_item(item):
            encoding = np.zeros(10).astype('uint8')
            encoding[0] = list(items).index(item) + 1
            
            mods = {'AD':0,
                    'crit_chance':0,
                    'armor':0,
                    'MR':0,
                    'dodge':0,
                    'health':0,
                    'mana':0,
                    'AS':0,
                    'SP':0}
                        
            for key, value in items[item].items():
                if key == 'AD':
                    encoding[1] = np.clip(value / 2, 0, 255)
                    mods[key] += value
                elif key == 'crit_chance':
                    encoding[2] = np.clip(value * 255, 0, 255)
                    mods[key] += value
                elif key == 'armor':
                    encoding[3] = np.clip(value, 0, 255)
                    mods[key] += value
                elif key == 'MR':
                    encoding[4] = np.clip(value, 0, 255)
                    mods[key] += value
                elif key == 'dodge':
                    encoding[5] = np.clip(value * 400, 0, 255)
                    mods[key] += value
                elif key == 'health':
                    encoding[6] = np.clip((value**0.5) * 3, 0, 255)
                    mods[key] += value
                elif key == 'mana':
                    encoding[7] = np.clip(value * 2.5, 0, 255)
                    mods[key] += value
                elif key == 'AS':
                    encoding[8] = np.clip((value-1) * 2.5, 0, 255)
                    mods[key] += value - 1
                elif key == 'SP':
                    encoding[9] = np.clip(value * 3, 0, 255)
                    mods[key] += value
                           
            return encoding, mods
        
        def encode_champion(champ, locationID):
            # items are 0:30, then champ stats
            es = item_encoding_size * 3
            encoding = np.zeros(es+champ_encoding_size).astype('uint8')         
            
            origins = []
            mods = []
            for i, item in enumerate(champ.items):
                if i >= 3:
                    break
                s = i*item_encoding_size
                e, m = encode_item(item)
                encoding[s:s+item_encoding_size] = e
                mods.append(m)
                if item in trait_items.values():
                    for o, t in trait_items.items():
                        if t == item:
                            origins.append(o)
                            break
            e_idx = es
            if champ.chosen:
                origins.append(champ.chosen)
            
            try:
                c_index = list(COST.keys()).index(champ.name)
            except:
                c_index = 0
                
            encoding[e_idx] = c_index
            e_idx += 1
            # max of 7 
            for i, origin in enumerate(champ.origin + origins):
                encoding[e_idx+i] = list(tiers).index(origin) + 1
            e_idx += 7
            
            encoding[e_idx] = champ.stars - 1
            e_idx += 1
            encoding[e_idx] = COST[champ.name]   
            e_idx += 1
            encoding[e_idx] = locationID
            e_idx += 1
            
            AD = champ.AD
            crit_chance = champ.crit_chance
            armor = champ.armor
            MR = champ.MR
            dodge = champ.dodge
            health = champ.health
            mana = champ.mana
            AS = 1
            SP = 1
            
            for mod in mods:
                AD += mod['AD']
                crit_chance += mod['crit_chance']
                armor += mod['armor']
                MR += mod['MR']
                dodge += mod['dodge']
                health += mod['health']
                mana += mod['mana']
                AS += mod['AS']
                SP += mod['SP']
                
            maxmana = champ.maxmana
            encoding[e_idx] = np.clip(AD / 2, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip(crit_chance * 255, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip(armor, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip(MR, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip(dodge * 255, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip((health**0.5) * 3, 0, 255)
            e_idx += 1
            if maxmana:
                encoding[e_idx] = np.clip(mana / maxmana * 255, 0, 255)
            else:
                encoding[e_idx] = 0
            e_idx += 1
            encoding[e_idx] = np.clip(maxmana * 1.45, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip(champ.AS * AS * 100, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip(SP * 100, 0, 255)
            e_idx += 1
            encoding[e_idx] = np.clip(champ.range * 32, 0, 255)
            e_idx += 1
            
            return encoding
                        
        obs = np.zeros((14, 4, item_encoding_size*3 + champ_encoding_size)).astype('uint8')
        for y in range(0, 4):
            # IMPORTANT TO HAVE THE X INSIDE -- Silver is not sure why but ok.
            for x in range(0, 7):
                if self.board[x][y]:                    
                    curr_champ = self.board[x][y]                        
                    obs[x, y, :] = encode_champion(curr_champ, 1)

        for x_bench in range(len(self.bench)):
            x, y = x_bench // 4 + 7, x_bench % 4
            if self.bench[x_bench]:
                curr_champ = self.bench[x_bench]                        
                obs[x, y, :] = encode_champion(curr_champ, 2)

        for ind, item in enumerate(self.item_bench):
            x, y = (ind + 2) // 4 + 10, (ind + 2) % 4
            if item and list(items).index(item):
                temp, _ = encode_item(item)
                obs[x, y, :len(temp)] = temp
        if private:
            for x_shop in range(len(self.shop)):
                x, y = (x_shop + 1) // 4 + 9, (x_shop + 1) % 4
                if self.shop[x_shop]:
                    curr_champ = self.shop[x_shop]  
                    origin = False                      
                    if curr_champ.endswith('_c'):
                        curr_champ, origin = curr_champ.split('_')[:2]
                    curr_champ = champion.champion(curr_champ, chosen=origin)
                    obs[x, y, :] = encode_champion(curr_champ, 3)     
        
        v_idx = 0
        max_o = 10
        vector = np.zeros(31).astype('uint8')
        for i, origin in enumerate(tiers):
            if self.team_tiers[origin]:
                vector[v_idx] = list(tiers).index(origin) + 1
                vector[v_idx + 1] = self.team_tiers[origin]
                v_idx += 2
        
        v_idx = max_o*2
        
        vector[v_idx] = np.clip(self.level - 1 / 8 * 255, 0, 255)
        v_idx += 1
        vector[v_idx] = np.clip(self.max_units - 1 / 9 * 255, 0, 255)
        v_idx += 1
        vector[v_idx] = np.clip(self.max_units - self.num_units_in_play / 10 * 255, 0, 255)
        v_idx += 1
        vector[v_idx] = np.clip(self.num_units_in_play / 10 * 255, 0, 255)
        v_idx += 1
        vector[v_idx] = np.clip(self.round * 5, 0, 255)
        v_idx += 1
        vector[v_idx] = np.clip(self.health * 2.1, 0, 255)
        v_idx += 1
        vector[v_idx] = np.clip(self.win_streak * 10, 0, 255)
        v_idx += 1
        vector[v_idx] = np.clip(self.loss_streak * 10, 0, 255)
        v_idx += 1
        if private:
            vector[v_idx] = np.clip(self.gold * 3, 0, 255)
            v_idx += 1
            exp_to_level = 0
            if self.level < self.max_level:
                exp_to_level = self.level_costs[self.level] - self.exp
            vector[v_idx] = np.clip(exp_to_level * 4, 0, 255)
            v_idx += 1
        else:
            gold = np.clip(self.gold // 10, 0, 5) * 10
            vector[v_idx] = np.clip(gold * 3, 0, 255)
            v_idx += 1
            vector[v_idx] = 0
            v_idx += 1
        vector[v_idx] = np.clip((self.actions_per_round - self.actions_taken_this_round) / self.actions_per_round * 255, 0, 255)
        v_idx += 1


        # vector size is 576
        for i in range(len(vector) // obs.shape[-1] + (1 if len(vector) % obs.shape[-1] else 0)):
            temp = vector[i*obs.shape[-1] : (i+1)*obs.shape[-1]]
            obs[-1, i, :len(temp)] = temp 
        
        action_obs = np.zeros((14*4, 2)).astype('uint8')
        if private and self.action_history:
            a = self.action_history[-1]
            x, y, z = a // 38 // 4, a // 38 % 4, a % 38
            action_obs[x*4 + y, 0] = 255
            if x >= 4 and y >= 4:
                dest_x, dest_y = x, y
            else:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                elif z < 37:
                    x_bench = z - 28
                    dest_x, dest_y = x_bench // 4 + 7, x_bench % 4                   
                else:
                    dest_x, dest_y = -1, 2
            action_obs[dest_x*4 + dest_y, 1] = 255
        action_obs = np.reshape(action_obs, (14, 4, 2))
            
        obs = np.concatenate([obs, action_obs], axis=-1)
        
        return obs
        

    """
    Description - Generates the item vector. This is done using binary encoding.
    """
    # return output_array
    # TODO: Make champion_duplicator work when bench is full and can upgrade rank
    def generate_item_vector(self):
        item_arr = np.zeros(config.MAX_BENCH_SPACE * 6)
        for ind, item in enumerate(self.item_bench):
            item_info = np.zeros(6)
            if item == 'champion_duplicator' and self.bench_full():
                item_info = utils.item_binary_encode(list(uncraftable_items).index(item) + 1)
                self.item_mask[ind] = 0
            elif item in uncraftable_items:
                item_info = utils.item_binary_encode(list(uncraftable_items).index(item) + 1)
                self.item_mask[ind] = 1
            elif item in item_builds.keys():
                item_info = utils.item_binary_encode(list(item_builds.keys()).index(item) + 1 + len(uncraftable_items))
                self.item_mask[ind] = 1
            else:
                self.item_mask[ind] = 0
            item_arr[ind*6:ind*6 + 6] = item_info
        self.item_vector = item_arr

    """
    Description - All information that other players do not have access to is stored in this vector
    """
    def generate_private_player_vector(self):
        self.player_private_vector[0] = self.gold / 100
        self.player_private_vector[1] = self.exp / 100
        self.player_private_vector[2] = self.round / 30

        exp_to_level = 0
        if self.level < self.max_level:
            exp_to_level = self.level_costs[self.level] - self.exp
        self.player_private_vector[4] = exp_to_level
        self.player_private_vector[5] = max(self.win_streak, self.loss_streak)
        if len(self.match_history) > 2:
            self.player_private_vector[6] = self.match_history[-3]
            self.player_private_vector[7] = self.match_history[-2]
            self.player_private_vector[8] = self.match_history[-1]
        # Who we can play against in the next round. / 20 to keep numbers between 0 and 1.
        # TODO: Figure out a better way to get around this nested if statement that doesn't involve iterating over
        # TODO: The entire list
        for x in range(9, 17):
            if (x - 9) < self.player_num:
                if ("player_" + str(x - 9)) in self.opponent_options:
                    self.player_private_vector[x] = self.opponent_options["player_" + str(x - 9)] / 20
                else:
                    self.player_private_vector[x] = -1
            elif (x - 0) > self.player_num:
                if ("player_" + str(x - 9)) in self.opponent_options:
                    self.player_private_vector[x - 1] = self.opponent_options["player_" + str(x - 9)] / 20
                else:
                    self.player_private_vector[x - 1] = -1

        # Decision mask parameters
        if self.level == self.max_level:
            self.decision_mask[4] = 0
        # if gold < 4, do not allow to level
        if self.gold < 4:
            self.decision_mask[4] = 0
            # Can not roll down to 0 gold
            self.decision_mask[5] = 0
        else:
            self.decision_mask[4] = 1
            self.decision_mask[5] = 1

        for idx, cost in enumerate(self.shop_costs):
            if self.gold < cost or cost == 0:
                self.shop_mask[idx] = 0
            elif self.gold >= cost:
                self.shop_mask[idx] = 1

    """
    Description - All game state information that other players have access to is stored here..
    """
    def generate_public_player_vector(self):
        self.player_private_vector[0] = self.health / 100
        self.player_public_vector[1] = self.level / 10
        self.player_public_vector[2] = self.max_units / 10
        self.player_private_vector[3] = self.max_units / 10
        self.player_public_vector[4] = self.num_units_in_play / self.max_units
        self.player_public_vector[5] = min(floor(self.gold / 10), 5) / 5.0
        streak_lvl = 0
        if self.win_streak == 4:
            streak_lvl = 0.5
        elif self.win_streak >= 5:
            streak_lvl = 1
        self.player_private_vector[6] = streak_lvl

    """
    Description - So we can call one method instead of 2 in the dozen or so places where these vectors get updated.
    """
    def generate_player_vector(self):
        self.generate_public_player_vector()
        self.generate_private_player_vector()


    """
    Description - This takes every occurrence of a champion at a given level and returns 1 of a higher level.
                  Transfers items over. The way I have it would mean it would require bench space.
    Inputs      - a_champion: Champion object
                    The third champion in the triple of 3.
    Outputs     - b_champion: Champion object
                    The goldened champion in whichever spot was decided to for it to be.             
    """
    # TODO: Verify if multiple units have full items, that it does not do weird behavior.
    def golden(self, a_champion) -> champion:
        x = -1
        y = -1
        chosen = False
        b_champion = champion.champion(a_champion.name, stars=a_champion.stars,
                                       itemlist=a_champion.items, chosen=a_champion.chosen)
        for i in range(0, len(self.bench)):
            if self.bench[i]:
                if self.bench[i].name == a_champion.name and self.bench[i].stars == a_champion.stars:
                    x = i
                    if self.bench[i].chosen:
                        chosen = self.bench[i].chosen
                    self.sell_from_bench(i, golden=True)
        for i in range(0, 7):
            for j in range(0, 4):
                if self.board[i][j]:
                    if self.board[i][j].name == a_champion.name and self.board[i][j].stars == a_champion.stars:
                        x = i
                        y = j
                        if self.board[i][j].chosen:
                            chosen = self.board[i][j].chosen
                        self.sell_champion(self.board[i][j], golden=True, field=True)
        b_champion.chosen = chosen
        b_champion.golden()
        if chosen:
            b_champion.new_chosen()

        # Leaving this code here in case I want to give rewards for triple
        # if b_champion.stars == 2:
        #     self.reward += 0.05
        #     self.print("+0.05 reward for making a level 2 champion")
        # if b_champion.stars == 3:
        #     self.reward += 1.0
        #     self.print("+1.0 reward for making a level 3 champion")
        temp = None
        if self.bench_full():
            temp = self.bench[0]
            self.bench[0] = None
        self.add_to_bench(b_champion)
        if y != -1:
            self.move_bench_to_board(b_champion.bench_loc, x, y)
        if temp:
            self.bench[0] = temp
        self.print("champion {} was made golden".format(b_champion.name))
        return b_champion

    """
    Description - Including base_exp income here. This gets called before any of the neural nets happen. 
                  This is the start of the round
    Inputs      - t_round: Int
                    Current game round
    """
    # TODO: FORTUNE TRAIT - HUGE EDGE CASE - GOOGLE FOR MORE INFO - FORTUNE - TFT SET 4
    def gold_income(self, t_round):
        self.exp += 2
        self.level_up()
        if t_round <= 4:
            starting_round_gold = [0, 2, 2, 3, 4]
            self.gold += floor(self.gold / 10)
            self.gold += starting_round_gold[t_round]
            #self.generate_player_vector()
            return
        interest = min(floor(self.gold / 10), 5)
        self.gold += interest
        self.gold += 5
        if self.win_streak == 2 or self.win_streak == 3 or self.loss_streak == 2 or self.loss_streak == 3:
            self.gold += 1
        elif self.win_streak == 4 or self.loss_streak == 4:
            self.gold += 2
        elif self.win_streak >= 5 or self.loss_streak >= 5:
            self.gold += 3
        #self.generate_player_vector()

    """
    Description - Checks if there's enough room on the item bench for a given number of items
    Inputs      - number of items to add to item bench
    Outputs     - boolean whether there's enough room
    """
    # num of items to be added to bench, set 0 if not adding.
    # I need to redo this to see how many slots within the length of the array are currently full.
    def item_bench_full(self, num_of_items=0) -> bool:
        counter = 0
        for i in self.item_bench:
            if i:
                counter += 1
        if counter + num_of_items > len(self.item_bench):
            #self.util_mask[2] = 1
            return True
        else:
            #self.util_mask[2] = 0
            return False

    """
    Description - Finds a free slot on the item bench
    """

    def item_bench_vacancy(self) -> int or False:
        for free_slot, u in enumerate(self.item_bench):
            if not u:
                return free_slot
        return False

    """
    Description - Checks if there's a kayn on the board
    """
    def kayn_check(self) -> bool:
        for x in range(0, 7):
            for y in range(0, 4):
                if self.board[x][y]:
                    if self.board[x][y].name == "kayn":
                        return True
        return False

    """
    Description - Gives the player the kayn items that allow the player to select kayn's form
    Outputs     - Sets kayn_transformed to true so it only happens once
    """
    def kayn_transform(self):
        if not self.kayn_transformed:
            if not self.item_bench_full(2):
                self.add_to_item_bench('kayn_shadowassassin')
                self.add_to_item_bench('kayn_rhast')
                self.kayn_transformed = True

    """
    Description - logic around leveling up. Also handles reward and max_unit amounts
    """
    def level_up(self):
        if self.level < self.max_level and self.exp >= self.level_costs[self.level]:
            self.exp -= self.level_costs[self.level]
            self.level += 1
            self.max_units += 1
            self.reward += self.level_reward
            self.print(f"leveled to {self.level}")
            # Only needed if it's possible to level more than once in one transaction
            self.level_up()

        if self.level == self.max_level:
            self.exp = 0

    """
    Description - Handles all variables related to losing rounds
    Inputs      - The amount of damage resulting from the loss (calculated in game_round)
    """
    # TODO: Separate losing a combat round and a minion round. They have differences related to win_streaks and classes
    def loss_round(self, damage):
        if not self.combat:
            self.loss_streak += 1
            self.win_streak = 0
            self.reward -= self.damage_reward * damage
            self.print(str(-self.damage_reward * damage) + " reward for losing round against player " + str(self.opponent.player_num))
            self.match_history.append(0)

            if self.team_tiers['fortune'] > 0:
                self.fortune_loss_streak += 1
                if self.team_tiers['fortune'] > 1:
                    self.fortune_loss_streak += 1
                    
    def move_bench_to_bench(self, bench_x, bench_x2):
        if 0 <= bench_x < 9 and self.bench[bench_x] and 0 <= bench_x2 < 9:
            m_champion = self.bench[bench_x]
            added_log = ""
            if self.bench[bench_x2]:
                self.bench[bench_x] = self.bench[bench_x2]
                self.bench[bench_x].x = bench_x
                self.bench[bench_x].y = -1
                added_log = ", and moved {} from bench {} to bench {}".format(self.bench[bench_x].name, 
                                                                                    bench_x2, bench_x)
            else:
                self.bench[bench_x] = None
            self.bench[bench_x2] = m_champion
            self.bench[bench_x2].x = bench_x2
            self.bench[bench_x2].x = -1
            self.print("moved {} from bench {} to bench {}".format(self.bench[bench_x2].name,
                                                                         bench_x, bench_x2) + added_log)
            return True
        self.reward += self.mistake_reward
        if DEBUG:
            print(f"Outside bench range, bench: {self.bench[bench_x]}, bench: {self.board[bench_x2]}, bench_x: {bench_x}, bench_x2: {bench_x2}")
        return False

    """
    Description - Moves a unit from bench to board if possible. Will switch if max units on board and board slot is used
    Inputs      - dcord: Int
                    For example, 27 -> 6 for x and 3 for y
    Outputs     - x: Int
                    x_coord
                  y: Int
                    y_coord
    """
    def move_bench_to_board(self, bench_x, board_x, board_y):
        if 0 <= bench_x < 9 and self.bench[bench_x] and 7 > board_x >= 0 and 4 > board_y >= 0:
            if self.num_units_in_play < self.max_units or self.board[board_x][board_y] is not None:
                # TO DO - IMPLEMENT AZIR TURRET SPAWNS
                m_champion = self.bench[bench_x]
                added_log = ""
                if self.board[board_x][board_y]:
                    if not self.board[board_x][board_y].target_dummy:
                        self.bench[bench_x] = self.board[board_x][board_y]
                        self.bench[bench_x].x = bench_x
                        self.bench[bench_x].y = -1
                        if len(self.bench[bench_x].items) > 0:
                            if self.bench[bench_x].items[0] == 'thieves_gloves':
                                self.thieves_gloves_loc_update(board_x, board_y, bench_x, -1)
                        if self.bench[bench_x].name == 'azir':
                            coords = self.bench[bench_x].sandguard_overlord_coordinates
                            self.bench[bench_x].overlord = False
                            for coord in coords:
                                self.board[coord[0]][coord[1]] = None
                        added_log = ", and moved {} from board [{}, {}] to bench {}".format(self.bench[bench_x].name, 
                                                                                            board_x, board_y, bench_x)
                    else:
                        added_log = ", but was unable to complete move because the board position was occupied by a non-champion"
                        m_champion = self.board[board_x][board_y]
                else:
                    self.bench[bench_x] = None
                    self.num_units_in_play += 1
                self.board[board_x][board_y] = m_champion
                self.board[board_x][board_y].x = board_x
                self.board[board_x][board_y].y = board_y
                # tracking thiefs gloves location
                if len(m_champion.items) > 0:
                    if m_champion.items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(bench_x, -1, board_x, board_y)
                if m_champion.name == 'azir':
                    # There should never be a situation where the board is to fill to fit the sand guards.
                    sand_coords = self.find_azir_sandguards(board_x, board_y)
                    self.board[board_x][board_y].overlord = True
                    self.board[board_x][board_y].sandguard_overlord_coordinates = sand_coords                
                #self.generate_bench_vector()
                #self.generate_board_vector()
                self.print("moved {} from bench {} to board [{}, {}]".format(self.board[board_x][board_y].name,
                                                                             bench_x, board_x, board_y) + added_log)
                self.update_team_tiers()
            else:
                self.reward += self.mistake_reward
            return True
        self.reward += self.mistake_reward
        self.print("Outside board range, bench: {}, board: {}, {}".format(bench_x, board_x, board_y))

        if DEBUG:
            print(f"Outside board range, bench: {self.bench[bench_x]}, board: {self.board[board_x][board_y]}, bench_x: {bench_x}, board_x: {board_x}, board_y: {board_y}")
        return False

    """
    Description - Moves a champion to the first open bench slot available
    Inputs      - x, y: Int
                    coords on the board to move to the board
    Outputs     - True if successful
                  False if coords are outside allowable range or could not sell unit
    """
    def move_board_to_bench(self, x, y, x_bench) -> bool:
        if 0 <= x < 7 and 0 <= y < 4 and 0 <= x_bench < 9:
            if self.bench[x_bench] and self.board[x][y] and not self.board[x][y].target_dummy:
                s_champion = self.bench[x_bench]
                if self.board[x][y]:
                    self.bench[x_bench] = self.board[x][y]
                    if self.board[x][y].name == 'azir':
                        coords = self.board[x][y].sandguard_overlord_coordinates
                        self.board[x][y].overlord = False
                        for coord in coords:
                            self.board[coord[0]][coord[1]] = None
                    self.board[x][y] = s_champion
                    self.board[x][y].x = x
                    self.board[x][y].y = y
                    self.bench[x_bench].x = x_bench
                    self.bench[x_bench].y = -1
                    if self.bench[x_bench].items and self.bench[x_bench].items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(x_bench, -1, x, y)
                    if self.board[x][y].items and self.board[x][y].items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(x, y, x_bench, -1)
                    self.update_team_tiers()
                    self.print("moved {} from board [{}, {}] to bench {}".format(self.bench[x_bench].name, x, y, x_bench) + \
                               ", and moved {} from bench {} to board [{}, {}]".format(self.board[x][y].name, x_bench, x, y))
                    return True
                else:                    
                    self.reward += self.mistake_reward
                if DEBUG:
                    print("Unit not on board slot")
                return False
            else:
                if self.board[x][y] and not self.board[x][y].target_dummy:
                    # Dealing with edge case of azir
                    if self.board[x][y].name == 'azir':
                        coords = self.board[x][y].sandguard_overlord_coordinates
                        self.board[x][y].overlord = False
                        for coord in coords:
                            self.board[coord[0]][coord[1]] = None
                    self.bench[x_bench] = self.board[x][y]
                    if self.board[x][y]:
                        self.print("moved {} from board [{}, {}] to bench".format(self.board[x][y].name, x, y))
                    self.board[x][y] = None
                    self.bench[x_bench].x = x_bench
                    self.bench[x_bench].y = -1
                    self.num_units_in_play -= 1
                    if self.bench[x_bench].items and self.bench[x_bench].items[0] == 'thieves_gloves':
                        self.thieves_gloves_loc_update(x_bench, -1, x, y)
                    #self.generate_bench_vector()
                    #self.generate_board_vector()
                    #self.generate_item_vector()
                    self.update_team_tiers()
                    return True
        self.reward += self.mistake_reward
        if DEBUG:
            print(f"Move board to bench outside board limits: {x}, {y}")
        return False

    """
    Description - Switches the contents of two squares on the board
    Inputs      - x1, y1: Int
                    coords on the board for square 1
                  x2, y2: Int
                    coords on the board for square 2
    Outputs     - True if successful
                  False if outside board limit or nothing on either square
    """
    def move_board_to_board(self, x1, y1, x2, y2) -> bool:
        if 0 <= x1 < 7 and 0 <= y1 < 4 and 0 <= x2 < 7 and 0 <= y2 < 4:
            if self.board[x1][y1] and self.board[x2][y2]:
                temp_champ = self.board[x2][y2]
                if (self.board[x1][y1].items and self.board[x1][y1].items[0] == 'thieves_gloves') or \
                   (self.board[x2][y2].items and self.board[x2][y2].items[0] == 'thieves_gloves'):
                    self.thieves_gloves_loc_update(x1, y1, x2, y2)
                self.board[x2][y2] = self.board[x1][y1]
                self.board[x1][y1] = temp_champ
                self.board[x1][y1].x = x1
                self.board[x1][y1].y = y1
                self.board[x2][y2].x = x2
                self.board[x2][y2].y = y2
                if self.board[x1][y1].name == 'sandguard' or self.board[x2][y2].name == 'sandguard':
                    for x in range(7):
                        for y in range(4):
                            if self.board[x][y] and self.board[x][y].name == 'azir':
                                if [x1, y1] in self.board[x][y].sandguard_overlord_coordinates and \
                                        [x2, y2] not in self.board[x][y].sandguard_overlord_coordinates:
                                    self.board[x][y].sandguard_overlord_coordinates.remove([x1, y1])
                                    self.board[x][y].sandguard_overlord_coordinates.append([x2, y2])
                                elif [x2, y2] in self.board[x][y].sandguard_overlord_coordinates and \
                                        [x1, y1] not in self.board[x][y].sandguard_overlord_coordinates:
                                    self.board[x][y].sandguard_overlord_coordinates.remove([x2, y2])
                                    self.board[x][y].sandguard_overlord_coordinates.append([x1, y1])
                self.print("moved {} and {} from board [{}, {}] to board [{}, {}]"
                           .format(self.board[x1][y1].name, self.board[x2][y2].name, x1, y1, x2, y2))
                #self.generate_board_vector()
                return True
            elif self.board[x1][y1]:
                if self.board[x1][y1].items and self.board[x1][y1].items[0] == 'thieves_gloves':
                    self.thieves_gloves_loc_update(x1, y1, x2, y2)
                self.board[x2][y2] = self.board[x1][y1]
                self.board[x1][y1] = None
                self.board[x2][y2].x = x2
                self.board[x2][y2].y = y2
                if self.board[x2][y2].name == 'sandguard':
                    for x in range(7):
                        for y in range(4):
                            if self.board[x][y] and self.board[x][y].name == 'azir':
                                if [x1, y1] in self.board[x][y].sandguard_overlord_coordinates:
                                    self.board[x][y].sandguard_overlord_coordinates.remove([x1, y1])
                                    self.board[x][y].sandguard_overlord_coordinates.append([x2, y2])
                self.print("moved {} from board [{}, {}] to board [{}, {}]".format(self.board[x2][y2].name, x1, y1, x2, y2))
                #self.generate_board_vector()
                return True
        self.reward += self.mistake_reward
        if DEBUG:
            print("Outside board limits")
        return False

    """
    Description - Moves an item or consumable from the item bench to either the player bench or player board
    Inputs      - item bench slot to move, [x, y] location to be moved to (y = -1 for the bench)
    Outputs     - boolean whether or not the move was successful
    """
    # TODO : Item combinations.
    # TODO : Documentation and setting up correct masks.
    def move_item(self, xBench, x, y) -> bool:
        board = False
        if y >= 0:
            champ = self.board[x][y]
            board = True
        if y == -1:
            champ = self.bench[x]
        if self.item_bench[xBench] and champ and not champ.target_dummy:
            # thiefs glove exception
            self.print("moving {} to {} with items {}".format(self.item_bench[xBench], champ.name, champ.items))
            # kayn item support
            if self.item_bench[xBench] == 'kayn_shadowassassin' or \
                    self.item_bench[xBench] == 'kayn_rhast':
                if champ.name == 'kayn':
                    self.transform_kayn(self.item_bench[xBench])
                    #self.generate_item_vector()
                    #self.decide_vector_generation(board)
                    return True
                if DEBUG:
                    print("Applying kayn item on not kayn")
                return False
            if self.item_bench[xBench] == 'champion_duplicator':
                if COST[champ.name] != 0:
                    if not self.bench_full():
                        self.add_to_bench(champion.champion(champ.name, chosen=champ.chosen, kayn_form=champ.kayn_form))
                        self.item_bench[xBench] = None
                        #self.generate_item_vector()
                        return True
                return False
            if self.item_bench[xBench] == 'magnetic_remover':
                if len(champ.items) > 0:
                    if not self.item_bench_full(len(champ.items)):
                        while len(champ.items) > 0:
                            self.item_bench[self.item_bench_vacancy()] = champ.items[0]
                            if champ.items[0] in trait_items.values():
                                champ.origin.pop(-1)
                                self.update_team_tiers()
                            champ.items.pop(0)
                        self.item_bench[xBench] = None
                        #self.generate_item_vector()
                        #self.decide_vector_generation(board)
                        return True
                if DEBUG:
                    print("Applying magnetic remover to a champion with no items")
                return False
            if self.item_bench[xBench] == 'reforger':
                return self.use_reforge(xBench, x, y)
            if self.item_bench[xBench] == 'thieves_gloves':
                if len(champ.items) < 1:
                    champ.items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                    self.thieves_gloves_loc.append([x, y])
                    self.thieves_gloves(x, y)
                    #self.generate_item_vector()
                    #self.decide_vector_generation(board)
                    return True
                if DEBUG:
                    print("Trying to add thieves gloves to unit with a separate item")
                return False
            # TODO: Clean up this code, we already checked for thieves_glove by this point
            if ((len(champ.items) < 3 and self.item_bench[xBench] != "thieves_gloves") or
                    (champ.items and champ.items[-1] in basic_items and self.item_bench[xBench]
                     in basic_items and len(champ.items) == 3)):
                for trait, name in enumerate(trait_items.values()):
                    if self.item_bench[xBench] == name:
                        item_trait = list(trait_items.keys())[trait]
                        if item_trait in champ.origin:
                            if DEBUG:
                                print("Trying to add item to unit with that trait")
                            return False
                        else:
                            champ.origin.append(item_trait)
                            self.update_team_tiers()
                # only execute if you have items
                if len(champ.items) > 0:
                    # implement the item combinations here. Make exception with thieves gloves
                    if champ.items[-1] in basic_items and self.item_bench[xBench] in basic_items:
                        coord = utils.x_y_to_1d_coord(champ.x, champ.y)
                        #self.glove_item_mask[coord] = 0
                        item_build_values = item_builds.values()
                        item_index = 0
                        item_names = list(item_builds.keys())
                        for index, items in enumerate(item_build_values):
                            if ((champ.items[-1] == items[0] and self.item_bench[xBench] == items[1]) or
                                    (champ.items[-1] == items[1] and self.item_bench[xBench] == items[0])):
                                item_index = index
                                break
                        for trait, names in enumerate(list(trait_items.values())):
                            if item_names[item_index] == names:
                                item_trait = list(trait_items.keys())[trait]
                                if item_trait in champ.origin:
                                    if DEBUG:
                                        print("trying to add trait item to unit with that trait")
                                    return False
                                else:
                                    champ.origin.append(item_trait)
                                    self.update_team_tiers()
                        if item_names[item_index] == "thieves_gloves":
                            if champ.num_items != 1:
                                if DEBUG:
                                    print("Trying to add thieves gloves to unit with a separate item")
                                return False
                            else:
                                self.thieves_gloves_loc.append([x, y])
                        self.item_bench[xBench] = None
                        champ.items.pop()
                        champ.items.append(item_names[item_index])
                        if champ.items[0] == 'thieves_gloves':
                            self.thieves_gloves(x, y)
                        self.reward += .2 * self.item_reward
                        if self.item_reward:
                            self.print(
                                ".2 reward for combining two basic items into a {}".format(item_names[item_index]))
                    elif champ.items[-1] in basic_items and self.item_bench[xBench] not in basic_items:
                        basic_piece = champ.items.pop()
                        champ.items.append(self.item_bench[xBench])
                        champ.items.append(basic_piece)
                        self.item_bench[xBench] = None
                    else:
                        if self.item_bench[xBench] == "sparring_gloves":
                            coord = utils.x_y_to_1d_coord(champ.x, champ.y)
                            #self.glove_item_mask[coord] = 1
                        champ.items.append(self.item_bench[xBench])
                        self.item_bench[xBench] = None
                else:
                    champ.items.append(self.item_bench[xBench])
                    self.item_bench[xBench] = None
                self.print("After Move {} to {} with items {}".format(self.item_bench[xBench], champ.name,
                                                                      champ.items))
                #self.generate_item_vector()
                #self.decide_vector_generation(board)
                return True
            elif len(champ.items) < 1 and self.item_bench[xBench] == "thieves_gloves":
                champ.items.append(self.item_bench[xBench])
                self.item_bench[xBench] = None
                #self.generate_item_vector()
                #self.decide_vector_generation(board)
                self.print("After Move {} to {} with items {}".format(self.item_bench[xBench], champ.name, champ.items))
                self.thieves_gloves_loc.append([x, -1])
                return True
        # last case where 3 items but the last item is a basic item and the item to input is also a basic item
        self.reward += self.mistake_reward
        if DEBUG:
            print(f"Failed to add item {self.item_bench[xBench]}")
        return False

    """
    Description - Uses the move_item method to move an item or consumable from the item bench to the player bench
    Inputs      - see move_item
    Outputs     - see move_item
    """
    def move_item_to_bench(self, xBench, x):
        self.move_item(xBench, x, -1)

    """
    Description - Uses the move_item method to move an item or consumable from the item bench to the board
    Inputs      - see move_item
    Outputs     - see move_item
    """
    def move_item_to_board(self, xBench, x, y):
        self.move_item(xBench, x, y)

    """
    Description - 
    Inputs      -
    Outputs     - 
    """
    def num_in_triple_catelog(self, a_champion):
        num = 0
        for entry in self.triple_catalog:
            if entry["name"] == a_champion.name and entry["level"] == a_champion.stars:
                # print("champion name: " + a_champion.name + " and their level is : " + str(a_champion.stars))
                num += 1
        return num

    """
    Description -
    Inputs      -
    Outputs     - 
    """
    def print(self, msg):
        self.printt('{:<120}'.format('{:<8}'.format(self.player_num)
                                     + '{:<20}'.format(str((time.time_ns() - self.start_time)/1000)) + msg))

    """
    Description -
    Inputs      -
    Outputs     - 
    """
    def printBench(self, log=True):
        for i in range(len(self.bench)):
            if self.bench[i]:
                if log:
                    self.print(str(i) + ": " + self.bench[i].name)
                else:
                    print(self.bench[i].name + ", ")

    """
    Description -
    Inputs      -
    Outputs     - 
    """
    def printComp(self, log=True):
        keys = list(self.team_composition.keys())
        values = list(self.team_composition.values())
        tier_values = list(self.team_tiers.values())
        self.prev_rewards = self.reward
        for i in range(len(self.team_composition)):
            if values[i] != 0:
                if log:
                    self.print("{}: {}, tier: {}".format(keys[i], values[i], tier_values[i]))
        for x in range(7):
            for y in range(4):
                if self.board[x][y]:
                    self.print("at ({}, {}), champion {}, with level = {}, items = {}, and chosen = {}".format(x, y,
                               self.board[x][y].name, self.board[x][y].stars,
                               self.board[x][y].items, self.board[x][y].chosen))
        self.print("Player level {} with gold {}, max_units = {}, ".format(self.level, self.gold, self.max_units) +
                   "num_units_in_play = {}, health = {}".format(self.num_units_in_play, self.health))


    """
    Description -
    Inputs      -
    Outputs     - 
    """
    def printItemBench(self, log=True):
        for i, item in enumerate(self.item_bench):
            if item:
                if log:
                    self.print(str(i) + ": " + item)
                else:
                    print(str(i) + ": " + item)

    """
    Description -
    Inputs      -
    Outputs     - 
    """
    def printShop(self, shop):
        self.print("Shop with level " + str(self.level) + ": " +
                   shop[0] + ", " + shop[1] + ", " + shop[2] + ", " + shop[3] + ", " + shop[4])

    """
    Description -
    Inputs      -
    Outputs     - 
    """
    def printt(self, msg):
        if config.PRINTMESSAGES:
            self.log.append(msg)


    """
    Description - Item pool mechanic utilities
    """
    def refill_item_pool(self):
        self.item_pool.extend(starting_items)

    """
    Description - Removes item given to player from pool to allow for a more diverse set of items received each game
    Inputs      - item: String
                    item to remove from pool
    """
    def remove_from_pool(self, item):
        self.item_pool.remove(item)

    """
    Description - Picks and returns item to player
    Outputs     - item: String
                    item to be returned.
    """
    def random_item_from_pool(self):
        item = random.choice(self.item_pool)
        self.remove_from_pool(item)
        return item

    """
    Description - Handles the gold cost and possible reward for refreshing a shop
    Outputs     - True: Refresh is allowed
                  False: Refresh is not possible
    """
    def refresh(self):
        if self.gold >= self.refresh_cost:
            self.gold -= self.refresh_cost
            self.reward += self.refresh_reward * self.refresh_cost
            self.print("Refreshing shop")
            #self.generate_player_vector()
            return True
        self.reward += self.mistake_reward
        if DEBUG:
            print("Could not refresh")
        return False

    """
    Description - Returns the items on a given champion to the item bench.
    Inputs      - x: Int
                    Bench location
    Outputs     - True: Item returned from bench
                  False: Item was not able to be returned. No unit at specified bench location
    """
    # TODO: Handle case where item_bench if full
    # TODO: Thieves_gloves bug appeared again on self.thieves_gloves_loc.remove([x, -1])
    def return_item_from_bench(self, x) -> bool:
        # if the unit exists
        if self.bench[x]:
            # skip if there are no items, trying to save a little processing time.
            if self.bench[x].items:

                # thieves_gloves_loc_always needs to be cleared even if there's not enough room on bench
                if self.bench[x].items[0] == 'thieves_gloves':
                    self.thieves_gloves_loc.remove([x, -1])
                    self.bench[x].items = ['thieves_gloves']

                # if I have enough space on the item bench for the number of items needed
                if not self.item_bench_full(len(self.bench[x].items)):
                    # Each item in possession
                    for i in self.bench[x].items:
                        # thieves glove exception
                        self.item_bench[self.item_bench_vacancy()] = i
                        self.print("returning " + i + " to the item bench")
                # if there is only one or two spots left on the item_bench and thieves_gloves is removed
                elif not self.item_bench_full(1) and self.bench[x].items[0] == "thieves_gloves":
                    self.item_bench[self.item_bench_cvacancy()] = self.bench[x].items[0]
                    self.print("returning " + str(self.bench[x].items[0]) + " to the item bench")
                self.bench[x].items = []
            #self.generate_item_vector()
            return True
        if DEBUG:
            print("No units at bench location {}".format(x))
        self.print("No units at bench location {}".format(x))
        return False

    """
    Description - Returns item from a given champion. Only used when bench is full and trying to add unit from carousel
                  or minion round. Do not use this method for selling a champion from board or bench
    Inputs      - a_champion: Champion object
                    champion to be sold.
    Outputs     - True: Able to return the item to the item bench.
                  False: Unable to return the item or method called with a NULL champion
    """
    def return_item(self, a_champion) -> bool:
        # if the unit exists
        if a_champion:
            # skip if there are no items, trying to save a little processing time.
            if a_champion.items:
                # thieves_gloves_location needs to be removed whether there's room on the bench or not
                if a_champion.items[0] == 'thieves_gloves':
                    if [a_champion.x, a_champion.y] in self.thieves_gloves_loc:
                        self.thieves_gloves_loc.remove([a_champion.x, a_champion.y])
                    a_champion.items = ['thieves_gloves']
                # if I have enough space on the item bench for the number of items needed
                if not self.item_bench_full(len(a_champion.items)):
                    # Each item in possession
                    for item in a_champion.items:
                        if item in trait_items.values():
                            a_champion.origin.pop(-1)
                            self.update_team_tiers()
                        self.item_bench[self.item_bench_vacancy()] = item
                        self.print("returning " + item + " to the item bench")

                # if there is only one or two spots left on the item_bench and thieves_gloves is removed
                elif not self.item_bench_full(1) and a_champion.items[0] == "thieves_gloves":
                    self.item_bench[self.item_bench_vacancy()] = a_champion.items[0]
                    self.print("returning " + str(a_champion.items[0]) + " to the item bench")
                else:
                    self.print("Could not remove item {} from champion {}".format(a_champion.items, a_champion.name))
                    if DEBUG:
                        print("Could not remove item {} from champion {}".format(a_champion.items, a_champion.name))
                    return False
                a_champion.items = []
                #self.generate_item_vector()

            return True
        if DEBUG:
            print("Null champion")
        return False

    """
    Description - Called when selling a unit to remove it from being tracked for tripling
    Inputs      - a_champion: Champion object
                    the champion we want to remove
                  golden: Boolean
                    True: We remove additional copies of the base unit beyond the single copy of champion
                    False: Remove only the one copy of the unit of choice.
    Outputs     - 
    """
    def remove_triple_catalog(self, a_champion, golden=False) -> bool:
        gold = False
        if golden:
            for unit in self.triple_catalog:
                if unit["name"] == a_champion.name and unit["level"] == a_champion.stars:
                    self.triple_catalog.remove(unit)
                    return True
            return True
        for idx, i in enumerate(self.triple_catalog):
            if i["name"] == a_champion.name and i["level"] == a_champion.stars:
                i["num"] -= 1
                if i["num"] == 0:
                    self.triple_catalog.pop(idx)
                return True
            if i["name"] == a_champion.name and i["level"] - 1 == a_champion.stars:
                gold = True
        if gold:
            self.print("Trying to fix bug for {} with level {}".format(a_champion.name, a_champion.stars))
            return True
        if a_champion.stars >= 2:
            self.print("Failed to remove " + str(a_champion.name) +
                       " from triple catalog with stars = " + str(a_champion.stars))
        self.print("could not find champion " + a_champion.name + " with star = "
                   + str(a_champion.stars) + " in the triple catelog")
        self.print("{}".format(self.triple_catalog))
        if DEBUG:
            print("could not find champion " + a_champion.name + " with star = "
                  + str(a_champion.stars) + " in the triple catelog")
        return False

    """
    Description - Used in unit tests to allow for a cleaner state.
    """
    def reset_state(self):
        self.bench = [None for _ in range(9)]
        self.board = [[None for _ in range(4)] for _ in range(7)]
        self.item_bench = [None for _ in range(10)]
        self.gold = 0
        self.level = 1
        self.exp = 0
        self.health = 100
        self.max_units = 1
        self.num_units_in_play = 0
        #self.generate_board_vector()
        #self.generate_bench_vector()
        #self.generate_item_vector()
        #self.generate_player_vector()
        #self.generate_chosen_vector()

    """
    Description - This should only be called when trying to sell a champion from the field and the bench is full
                  This can occur after a carousel round where you get a free champion and it can enter the field
                  Even if you already have too many units in play. The default behavior will be sell that champion.
    Inputs      - a_champion: Champion object
                    the champion we want to remove
                  golden: Boolean
                    True: Don't update the pool or grant gold
                    False: Update pool with sold champion and grant gold.
                  field: Boolean
                    True: unit is being sold from the field so decrement units in play
                    False: don't decrement units in play
    Outputs     - True: Unit successful sold
                  False: Was unable to sell unit due to remove from triple catalog, return item or target dummy.
    """
    # TODO: Varify that the bench / board vectors are being updated somewhere in the same operation as this method call.
    def sell_champion(self, s_champion, golden=False, field=True) -> bool:
        # Need to add the behavior that on carousel when bench is full, add to board.
        if not (self.remove_triple_catalog(s_champion, golden=golden) and self.return_item(s_champion) and not
                s_champion.target_dummy):
            self.reward += self.mistake_reward
            self.print("Could not sell champion " + s_champion.name)
            if DEBUG:
                print("Could not sell champion " + s_champion.name)
            return False
        if not golden:
            self.gold += cost_star_values[s_champion.cost - 1][s_champion.stars - 1]
            self.pool_obj.update_pool(s_champion, 1)
            #self.generate_player_vector()
        if s_champion.chosen:
            self.chosen = False
        if s_champion.x != -1 and s_champion.y != -1:
            self.board[s_champion.x][s_champion.y] = None
            #self.generate_board_vector()
        if field:
            self.num_units_in_play -= 1
        self.print("selling champion " + s_champion.name + " with stars = " + str(s_champion.stars) + " from position ["
                   + str(s_champion.x) + ", " + str(s_champion.y) + "]")
        return True

    """
    Description - Selling unit from the bench
    Inputs      - location: Int
                    Which location on the bench to sell from
                  golden: Boolean
                    True: Don't update the pool or grant gold
                    False: Update pool with sold champion and grant gold.
    Outputs     - True: Unit successful sold
                  False: Was unable to sell unit due to remove from triple catalog, return item or target dummy.
    """
    def sell_from_bench(self, location, golden=False) -> bool:
        if self.bench[location]:
            if not (self.remove_triple_catalog(self.bench[location], golden=golden) and
                    self.return_item_from_bench(location)):
                self.print("Mistake in sell from bench with {} and level {}".format(self.bench[location],
                                                                                    self.bench[location].stars))
                self.reward += self.mistake_reward
                if DEBUG:
                    print("Could not remove from triple catalog or return item")
                return False
            if not golden:
                self.gold += cost_star_values[self.bench[location].cost - 1][self.bench[location].stars - 1]
                self.pool_obj.update_pool(self.bench[location], 1)
            if self.bench[location].chosen:
                self.chosen = False
            return_champ = self.bench[location]
            self.print("selling champion " + self.bench[location].name + " with stars = " +
                       str(self.bench[location].stars) + " from bench_location " + str(location))
            self.bench[location] = None
            #self.generate_bench_vector()
            #self.generate_item_vector()
            #self.generate_player_vector()
            return return_champ
        if DEBUG:
            print("Nothing at bench location")
        return False

    """
    Description - Returns true if there are no possible actions in the state
    Outputs     - True: No possible actions
                  False: There are actions possible
    """
    def spill_reward(self, damage):
        self.reward += self.damage_reward * damage
        self.print("Spill reward of {} received".format(self.damage_reward * damage))

    """
    Description - Does all operations that happen at the start of the round. 
                  This includes gold, reward, kayn updates and thieves gloves
    Inputs      - t_round: Int
                    current game round
    """
    # TODO Organize methods to be alphabetical so people can find what they are looking for in this file. This file only
    def start_round(self, t_round, log=True):
        self.start_time = time.time_ns()
        self.round = t_round
        self.reward += self.num_units_in_play * self.minion_count_reward
        if self.gold > 70:
            self.reward += self.excess_gold_reward
        self.gold_income(self.round)
        #self.generate_player_vector()
        if self.kayn_check():
            self.kayn_turn_count += 1
        if self.kayn_turn_count >= 3:
            self.kayn_transform()
        for x in self.thieves_gloves_loc:
            if (x[1] != -1 and self.board[x[0]][x[1]]) or self.bench[x[0]]:
                self.thieves_gloves(x[0], x[1])
        self.actions_taken_this_round = 0
        self.ai.reset_phase()
        self.selection = None
        if log:
            self.printComp()
            self.printBench()
            self.printItemBench()
            self.printShop(self.shop)

    """
    Description - Returns true if there are no possible actions in the state
    Outputs     - True: No possible actions
                  False: There are actions possible
    """
    def state_empty(self):
        # Need both in case of an empty shop.
        if self.gold == 0 or self.gold < min(self.shop_costs):
            for xbench in self.bench:
                if xbench:
                    return False
            for x_board in range(len(self.board)):
                for y_board in range(len(self.board[0])):
                    if self.board[x_board][y_board]:
                        return False
            return True
        else:
            return False

    """
    Description - Gives new thieves gloves items to a champion in thieves_gloves_loc
    Inputs      - thieves_gloves_loc coordinates
    Outputs     - 2 completed items
    """
    def thieves_gloves(self, x, y) -> bool:
        r1 = random.randint(0, len(thieves_gloves_items) - 1)
        r2 = random.randint(0, len(thieves_gloves_items) - 1)
        while r1 == r2:
            r2 = random.randint(0, len(thieves_gloves_items) - 1)
        self.print("thieves_gloves: {} and {}".format(thieves_gloves_items[r1], thieves_gloves_items[r2]))
        if y >= 0:
            self.board[x][y].items = ['thieves_gloves']
            self.board[x][y].items.append(thieves_gloves_items[r1])
            self.board[x][y].items.append(thieves_gloves_items[r2])
            return True
        elif y == -1:
            self.bench[x].items = ['thieves_gloves']
            self.bench[x].items.append(thieves_gloves_items[r1])
            self.bench[x].items.append(thieves_gloves_items[r2])
            return True
        else:
            if DEBUG:
                print("Could not assign thieves glove items")
            return False

    """
    Description - Checks if either of 2 coordinates is in thieves_gloves_loc and swaps it for the one that isn't
    Inputs      - 2 x, y coordinates
    Outputs     - modified thieves_gloves_loc list
    """
    def thieves_gloves_loc_update(self, x1, y1, x2, y2):
        if [x1, y1] in self.thieves_gloves_loc and [x2, y2] in self.thieves_gloves_loc:
            return True
        elif [x1, y1] in self.thieves_gloves_loc:
            self.thieves_gloves_loc.remove([x1, y1])
            self.thieves_gloves_loc.append([x2, y2])
            #self.thieves_mask_update(x1, y1, x2, y2)
        elif [x2, y2] in self.thieves_gloves_loc:
            self.thieves_gloves_loc.remove([x2, y2])
            self.thieves_gloves_loc.append([x1, y1])
            #self.thieves_mask_update(x2, y2, x1, y1)

    """
    Description - Updating the thieves glove mask
    Inputs      - x1, y1 - coords of the unit to remove, y=-1 for bench
                - x2, y2 - coords of unit to add, y=-1 for bench
    """
    def thieves_mask_update(self, x1, y1, x2, y2):
        coord_remove = utils.x_y_to_1d_coord(x1, y1)
        #self.thieves_glove_mask[coord_remove] = 0

        coord_add = utils.x_y_to_1d_coord(x2, y2)
        #self.thieves_glove_mask[coord_add] = 1

    """
    Description - Transforms Kayn into either shadowassassin or rhast based on which item the player used
    Inputs      - kayn item (either shadowassassin or rhast)
    Outputs     - Transforms all Kayns (board, bench and shop)
    """
    def transform_kayn(self, kayn_item):
        self.kayn_form = kayn_item
        for x in range(len(self.item_bench)):
            if self.item_bench[x] == 'kayn_shadowassassin' or self.item_bench[x] == 'kayn_rhast':
                self.item_bench[x] = None
        for x in range(7):
            for y in range(4):
                if self.board[x][y]:
                    if self.board[x][y].name == 'kayn':
                        self.board[x][y].kayn_form = kayn_item
        for x in range(9):
            if self.bench[x]:
                if self.bench[x].name == 'kayn':
                    self.bench[x].kaynform = kayn_item

    """
    Description - Updates shop costs to use for the shop mask
    Inputs      - shop_costs: List
                    list of ints of size [shop_len] with the cost of each shop.
    """
    def update_shop_costs(self, shop_costs):
        self.shop_costs = shop_costs

    """
    Description - Updates the traits that this player's comp has. Connected to the team_origin_class in origin_class.py
                  This is used when looking for specific traits as well as part of the observation.
    """
    def update_team_tiers(self):
        self.team_composition = origin_class.team_origin_class(self)
        if self.chosen in self.team_composition.keys():
            self.team_composition[self.chosen] += 1
        for trait in self.team_composition:
            counter = 0
            while self.team_composition[trait] >= tiers[trait][counter]:
                counter += 1
                if counter >= len(tiers[trait]):
                    break
            self.team_tiers[trait] = counter
        origin_class.game_comp_tiers[self.player_num] = self.team_tiers

    """
    Description - Method for keeping track of which units are golden
                  It calls golden which then calls add_to_bench which calls this again with the goldened unit
    Inputs      - a_champion: Champion object
                    champion to be added to the catalog
    Outputs     - Boolean if the unit was goldened, 
                  Boolean for successful operation.
    """
    def update_triple_catalog(self, a_champion):
        for entry in self.triple_catalog:
            if entry["name"] == a_champion.name and entry["level"] == a_champion.stars:
                entry["num"] += 1
                if entry["num"] == 3:
                    self.golden(a_champion)
                    return True, True
                return False, True
        if a_champion.stars > 3:
            return False, False
        self.triple_catalog.append({"name": a_champion.name, "level": a_champion.stars, "num": 1})
        return False, True

    """
    Description - Handles all reforger functions
    Inputs      - reforger's slot on the item bench, champion coordinates (y = -1 for bench)
    Outputs     - Reforges the items if there's room on item bench, otherwise returns false
    """
    def use_reforge(self, xBench, x, y) -> bool:
        board = False
        trait_item_list = list(trait_items.values())
        if y >= 0:
            champ = self.board[x][y]
            board = True
        elif y == -1:
            champ = self.bench[x]
        if len(champ.items) > 0 and not self.item_bench_full(len(champ.items)):
            for item in champ.items:
                if item == 'spatula':
                    self.item_bench[self.item_bench_vacancy()] = 'spatula'
                elif item in starting_items:
                    r = random.randint(0, 7)
                    while starting_items[r] == item:
                        r = random.randint(0, 7)
                    self.item_bench[self.item_bench_vacancy()] = starting_items[r]
                elif item in trait_item_list:
                    r = random.randint(0, 7)
                    while trait_item_list[r] == item:
                        r = random.randint(0, 7)
                    self.item_bench[self.item_bench_vacancy()] = trait_item_list[r]
                elif item in thieves_gloves_items:
                    r = random.randint(0, len(thieves_gloves_items) - 1)
                    while thieves_gloves_items[r] == item:
                        r = random.randint(0, len(thieves_gloves_items) - 1)
                    self.item_bench[self.item_bench_vacancy()] = thieves_gloves_items[r]
                else:   # this will only ever be thieves gloves
                    r = random.randint(0, len(thieves_gloves_items) - 1)
                    self.item_bench[self.item_bench_vacancy()] = thieves_gloves_items[r]
            champ.items = []
            self.item_bench[xBench] = None
            #self.generate_item_vector()
            #self.decide_vector_generation(board)
            return True
        if DEBUG:
            print("could not use reforge")
        return False

    """
    Description - Called at the conclusion of the game to the player who won the game
    """
    def won_game(self):
        self.reward += self.won_game_reward
        self.print("+0 reward for winning game")
        
    def lost_game(self, num_players):
        if num_players >= 4:
            self.reward -= self.won_game_reward
        self.print("-0 reward for losing game")

    """
    Description - Same as loss_round but if the opponent was a ghost
    Inputs      - damage: Int
                    amount of damage inflicted in the combat round
    """
    # TODO - split the negative reward here among the rest of the players to maintain a net equal reward
    # TODO - move the 0.5 to the list of other reward controllers for each of the won / loss round methods
    def won_ghost(self):
        if not self.combat:
            self.win_streak += 1
            self.loss_streak = 0
            self.gold += 1
            self.print("won round against a ghost")
            self.match_history.append(1)

            if self.team_tiers['fortune'] > 0:
                if self.fortune_loss_streak >= len(fortune_returns):
                    self.gold += math.ceil(fortune_returns[len(fortune_returns) - 1] +
                                           15 * (self.fortune_loss_streak - len(fortune_returns)))
                    self.fortune_loss_streak = 0
                    return
                self.gold += math.ceil(fortune_returns[self.fortune_loss_streak])
                self.fortune_loss_streak = 0

    """
    Description - Keeps track of win_streaks, rewards, gold and other values related to winning a combat round.
    Inputs      - damage: Int
                    Amount of damage inflicted in the combat round
    """
    def won_round(self, damage):
        if not self.combat:
            self.win_streak += 1
            self.loss_streak = 0
            self.gold += 1
            self.reward += self.damage_reward * damage
            self.print(str(self.damage_reward * damage) + " reward for winning round against player " + str(self.opponent.player_num))
            self.match_history.append(1)

            if self.team_tiers['fortune'] > 0:
                if self.fortune_loss_streak >= len(fortune_returns):
                    self.gold += math.ceil(fortune_returns[len(fortune_returns) - 1] +
                                           15 * (self.fortune_loss_streak - len(fortune_returns)))
                    self.fortune_loss_streak = 0
                    return
                self.gold += math.ceil(fortune_returns[self.fortune_loss_streak])
                self.fortune_loss_streak = 0

class Comp:
    def __init__(self, champions, positions, item_components, chosen, 
                 mirror=False):
        self.champions = champions
        if mirror:
            self.positions = {champ: (6-pos[0], pos[1]) for champ, pos in zip(champions, positions)}
        else:
            self.positions = {champ: pos for champ, pos in zip(champions, positions)}
        self.champ_items = {champ: item for champ, item in zip(champions, item_components)}
        self.items = [a for i in item_components for a in i ]
        self.crafted_items = []
        self.chosen = chosen
    def in_comp(self, champ):
        return champ in self.champions
    def champion_position(self, champ):
        if self.in_comp(champ):
            return self.positions[champ]
        return None
    def place_item(self, item):
        return None
    def craft_items(self, item_components):
        permut = permutations(item_components, 2)
        for item in self.items:
            if item not in self.crafted_items:
                components = item_builds[item]
                for item_pair in permut:
                    if item_pair[0] in components and item_pair[1] in components:
                        pass
        return None
            
five_cost = [c for c, co in COST.items() if co == 5]
class PlayerAI:
    # Explanation - We may switch to a class config for the AI side later so separating the two now is highly useful.
    def __init__(self, player):
        self.player = player
        self.phase = 0
        self.save = False
        self.comp = self.make_meta_comp()
        
    def make_random_comp(self):
        k = {c: [] for c in range(1, 6)}
        for key, item in COST.items():
            if item:
                k[item].append(key)
        comp = []
        for i in range(1, 5):
            comp += np.random.choice(k[i], 2).tolist()
        comp.append(np.random.choice(k[5]))
        return comp
    
    def make_meta_comp(self):
        roll = np.random.randint(0, 12)
        #if self.player.player_num == 0:
        #    roll = 0
        if roll == 0:
            champs = ['zed',
                    'akali',
                    'evelynn',
                    'kennen',
                    'pyke',
                    'elise',
                    'shen',
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(0, 0),
                         (1, 0),
                         (2, 0),
                         (3, 0),
                         (4, 0),
                         (3, 1),
                         (4, 1),
                         (5, 0),
                         (6, 0),
                         (0, 1)]
            return Comp(champs + added, positions, [], 'shade',
                             mirror=np.random.rand() < 0.4)
        elif roll == 1:
            champs = ['irelia',
                    'jax',
                    'warwick',
                    'leesin',
                    'shen',
                    'ezreal',
                    'lux',
                    'yone'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(1, 3),
                         (2, 3),
                         (3, 3),
                         (4, 3),
                         (5, 3),
                         (0, 0),
                         (6, 0),
                         (6, 2),
                         (1, 0),
                         (2, 0)]
            return Comp(champs + added, positions, [], 'divine',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 2:
            champs = ['vayne',
                    'thresh',
                    'lillia',
                    'jhin',
                    'zilean',
                    'aatrox',
                    'riven',
                    'cassiopeia'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(0, 0),
                         (4, 0),
                         (5, 0),
                         (6, 0),
                         (6, 1),
                         (2, 3),
                         (3, 3),
                         (4, 3),
                         (1, 0),
                         (2, 0)]
            return Comp(champs + added, positions, [], 'dusk',
                             mirror=np.random.rand() < 0.4)
        elif roll == 3:
            champs = ['katarina',
                    'akali',
                    'talon',
                    'lissandra',
                    'morgana',
                    'diana',
                    'pyke',
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(0, 0),
                         (1, 0),
                         (2, 0),
                         (3, 0),
                         (4, 0),
                         (5, 0),
                         (6, 0),
                         (0, 1),
                         (1, 1),
                         (2, 1)]
            return Comp(champs + added, positions, [], 'moonlight',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 13:
            champs = ['xinzhao',
                    'jarviniv',
                    'garen',
                    'katarina',
                    'azir',
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(4, 3),
                         (5, 3),
                         (6, 3),
                         (4, 2),
                         (6, 0),
                         (0, 0),
                         (1, 0),
                         (2, 0),
                         (3, 0),
                         (4, 0)]
            return Comp(champs + added, positions, [], 'warlord',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 5:
            champs = ['jax',
                    'fiora',
                    'yone',
                    'yasuo',
                    'leesin',
                    'lux',
                    'irelia',
                    'ezreal'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(3, 3),
                         (5, 3),
                         (0, 2),
                         (6, 2),
                         (2, 1),
                         (0, 0),
                         (4, 0),
                         (6, 0),
                         (5, 0),
                         (3, 0)]
            return Comp(champs + added, positions, [], 'duelist',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 6:
            champs = ['sejuani',
                    'yuumi',
                    'shen',
                    'jhin',
                    'teemo',
                    'jinx',
                    'nidalee',
                    'vayne'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(3, 3),
                         (3, 1),
                         (4, 1),
                         (0, 0),
                         (1, 0),
                         (2, 0),
                         (3, 0),
                         (6, 0),
                         (5, 0),
                         (4, 0)]
            return Comp(champs + added, positions, [], 'sharpshooter',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 7:
            champs = ['irelia',
                    'yone',
                    'shen',
                    'yuumi',
                    'warwick',
                    'ashe',
                    'kindred',
                    'aphelios'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(1, 3),
                         (3, 3),
                         (5, 3),
                         (6, 3),
                         (2, 1),
                         (0, 0),
                         (1, 0),
                         (2, 0),
                         (5, 0),
                         (6, 0)]
            return Comp(champs + added, positions, [], 'hunter',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 8:
            champs = ['shen',
                    'yone',
                    'irelia',
                    'pyke',
                    'lux',
                    'talon',
                    'morgana',
                    'janna'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(1, 3),
                         (3, 3),
                         (5, 3),
                         (6, 2),
                         (0, 0),
                         (4, 0),
                         (5, 0),
                         (6, 0),
                         (3, 0),
                         (2, 0)]
            return Comp(champs + added, positions, [], 'enlightened',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 9:
            champs = ['cassiopeia',
                    'aatrox',
                    'sejuani',
                    'thresh',
                    'zilean',
                    'yuumi',
                    'janna',
                    'ahri'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(0, 3),
                         (2, 3),
                         (4, 3),
                         (6, 3),
                         (6, 1),
                         (0, 0),
                         (5, 0),
                         (6, 0),
                         (3, 0),
                         (2, 0)]
            return Comp(champs + added, positions, [], 'vanguard',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 10:
            champs = ['elise',
                    'aatrox',
                    'pyke',
                    'kalista',
                    'evelynn',
                    'zilean',
                    'twistedfate',
                    'jhin'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(3, 2),
                         (5, 2),
                         (0, 1),
                         (6, 1),
                         (3, 0),
                         (4, 0),
                         (5, 0),
                         (6, 0),
                         (2, 0),
                         (1, 0)]
            return Comp(champs + added, positions, [], 'cultist',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 11:
            champs = ['sett',
                    'maokai',
                    'warwick',
                    'nunu',
                    'lulu',
                    'kindred',
                    'ezreal',
                    'ashe'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(3, 3),
                         (5, 3),
                         (4, 1),
                         (6, 1),
                         (0, 0),
                         (4, 0),
                         (5, 0),
                         (6, 0),
                         (2, 0),
                         (1, 0)]
            return Comp(champs + added, positions, [], 'elderwood',
                             mirror=np.random.rand() < 0.4)
        
        elif roll == 4:
            champs = ['irelia',
                    'shen',
                    'annie',
                    'fiora',
                    'talon',
                    'morgana',
                    'janna',
                    'nami'
                    ]
            added = list(np.random.choice([i for i in five_cost if i not in champs], 10 - len(champs)))
            positions = [(1, 3),
                         (3, 3),
                         (5, 3),
                         (6, 1),
                         (1, 0),
                         (4, 0),
                         (5, 0),
                         (6, 0),
                         (2, 0),
                         (3, 0)]
            return Comp(champs + added, positions, [], 'mage',
                             mirror=np.random.rand() < 0.4)
            
        
        
    def next_phase(self):
        if self.phase < 7:
            self.phase += 1
        
    def reset_phase(self):
        self.phase = 0
        
    def get_action(self):
        if self.player.actions_taken_this_round == 4:
            return self.idle()
        # buy wanted champions
        if self.phase == 0:
            return self.phase_0()
        # level up
        elif self.phase == 1:
            return self.phase_1()
        # buy random champion if total num champions < max team size
        elif self.phase == 2:
            return self.phase_2()
        # fill board with non-duplicate champions
        elif self.phase == 3:
            return self.phase_3()
        # check if bench needs to be sold
        elif self.phase == 4:
            return self.phase_4()
        # roll excess gold to search for wanted champions
        elif self.phase == 5:
            return self.phase_5()
        # place items
        elif self.phase == 6:
            return self.phase_6()
        # pass
        else:
            return self.phase_7()
    
    def phase_0(self):
        for i, c in enumerate(self.player.shop):
            if c:
                champ_name = c.split('_')[0]
                if champ_name in self.comp.champions:   
                    go = True
                    for trip in self.player.triple_catalog:
                        if trip['name'] == champ_name and trip['level'] == 3:
                            go = False
                            break
                    if go:
                        level = 1
                        cost = COST[champ_name]
                        if c.endswith('_c'):
                            if c.split('_')[1] != self.comp.chosen:
                                continue
                            cost = cost_star_values[cost - 1][1]
                            level = 2
                        if cost <= self.player.gold:
                            if self.player.bench_full():
                                for trip in self.player.triple_catalog:
                                    if trip['name'] == champ_name and trip['num'] == 2 and trip['level'] == level:
                                        return self.buy(i)
                            else:
                                return self.buy(i)
        
        self.next_phase()
        return self.get_action()
        
    def phase_1(self):
        if self.player.round < 4:
            self.next_phase()
            return self.get_action()
        elif self.player.level < 4:
            if self.player.gold < self.player.exp_cost:
                self.save = True
                self.next_phase()
                return self.get_action()
            else:
                self.save = False
                return self.xp()
        elif self.player.level < 5 and self.player.round > 6:
            if self.player.gold < self.player.exp_cost:
                self.save = True
                self.next_phase()
                return self.get_action()
            else:
                self.save = False
                return self.xp()
        elif self.player.level < 6 and self.player.round > 12:
            if self.player.gold < self.player.exp_cost:
                self.save = True
                self.next_phase()
                return self.get_action()
            else:
                self.save = False
                return self.xp()
        elif self.player.level < 7 and self.player.round > 16:
            if self.player.gold < self.player.exp_cost:
                self.save = True
                self.next_phase()
                return self.get_action()
            else:
                self.save = False
                return self.xp()
        elif self.player.level < 8 and self.player.round > 22:
            if self.player.gold < self.player.exp_cost:
                self.save = True
                self.next_phase()
                return self.get_action()
            else:
                self.save = False
                return self.xp()
        elif self.player.level < 9 and self.player.round > 28:
            if self.player.gold < self.player.exp_cost:
                self.save = True
                self.next_phase()
                return self.get_action()
            else:
                self.save = False
                return self.xp()
        else:
            self.next_phase()
            return self.get_action()
    
    def phase_2(self):
        self.next_phase()
        if self.player.bench_full():
            return self.get_action()
        champs = []
        for row in self.player.board:
            for c in row:
                if c and c.name not in champs:
                    champs.append(c.name)
        for c in self.player.bench:
            if c and c.name not in champs:
                champs.append(c.name)
        if len(champs) < self.player.max_units:
            for i, c in enumerate(self.player.shop):
                if c:
                    champ_name = c.split('_')[0]
                    if c not in champs:
                        cost = COST[champ_name]
                        if c.endswith('_c') and champ_name not in self.comp.champions:
                            continue
                            #cost = cost_star_values[cost - 1][1]
                        if cost <= self.player.gold:
                            if not self.player.bench_full():
                                return self.buy(i)
        return self.get_action()
        
    def phase_3(self):
        board_comp = []
        board_other = []
        for row in self.player.board:
            for c in row:
                if c:
                    if c.name in self.comp.champions:
                        if c.name in [i.name for i in board_comp]:
                            dup = [i for i in board_comp if i.name == c.name][0]
                            if dup.stars < c.stars:
                                board_comp.remove(dup)
                                board_other.append(dup)
                                board_comp.append(c)
                            else:
                                board_other.append(c)
                        else:
                            board_comp.append(c)
                    else:
                        board_other.append(c)
        if self.player.num_units_in_play < self.player.max_units:
            bench_comp = []
            bench_other = []
            for i, c in enumerate(self.player.bench):
                if c and (c.name not in [ii.name for ii in board_comp + board_other] or c.chosen):
                    if c.name in self.comp.champions:
                        bench_comp.append([c, i])
                    else:
                        bench_other.append([c, i])
            c = None
            if bench_comp:
                c = bench_comp[np.random.randint(len(bench_comp))]
            elif bench_other:
                c = bench_other[np.random.choice(len(bench_other))]
            if c:
                place = self.comp.champion_position(c[0].name)
                if place:
                    return self.place(c[1], place[0], place[1])
                else:
                    for place in self.comp.positions.values():
                        if self.player.board[place[0]][place[1]]:
                            continue
                        else:
                            self.player.print('placing non comp')
                            return self.place(c[1], place[0], place[1])
        elif board_other:
            for i, c in enumerate(self.player.bench):
                if c and c.name not in [ii.name for ii in board_comp + board_other] and c.name in self.comp.champions:
                    self.player.print('removing')
                    return self.remove(board_other[0].x, board_other[0].y)
        self.next_phase()
        return self.get_action()
    
    def phase_4(self):
        for i, c in enumerate(self.player.bench):
            if c and c.name not in self.comp.champions:
                return self.sell(i)
        self.next_phase()
        return self.get_action()
    
    def phase_5(self):
        if not self.save:
            for i, c in enumerate(self.player.shop):
                if c:
                    champ_name = c.split('_')[0]
                    if champ_name in self.comp.champions:    
                        go = True
                        for trip in self.player.triple_catalog:
                            if trip['name'] == champ_name and trip['level'] == 3:
                                go = False
                                break
                        if go:
                            level = 1
                            cost = COST[champ_name]
                            if c.endswith('_c'):
                                if c.split('_')[1] != self.comp.chosen:
                                    continue
                                cost = cost_star_values[cost - 1][1]
                                level = 2
                            if cost <= self.player.gold:
                                if self.player.bench_full():
                                    for trip in self.player.triple_catalog:
                                        if trip['name'] == champ_name and trip['num'] == 2 and trip['level'] == level:
                                            return self.buy(i)
                                else:
                                    return self.buy(i)            
            if self.player.gold > max(12, min(self.player.round * 2, 42)):
                return self.roll()
            
        self.next_phase()
        return self.get_action()
    
    def phase_6(self):
        for i, item in enumerate(self.player.item_bench):
            if item:
                return self.place_item(i)
        self.next_phase()
        return self.get_action()
    
    def phase_7(self):
        return self.idle()
    
    def idle(self):
        return 2090
    
    def place_item(self, item_slot):
        self.next_phase()
        x, y = (item_slot + 2) // 4 + 10, (item_slot + 2) % 4
        slots = []
        for board_y in range(4):
            for board_x in range(7):
                if self.player.board[board_x][board_y] and len(self.player.board[board_x][board_y].items) < 3:
                    z = board_x*4 + board_y
                    slots.append(z + 38*y + 38*4*x)
        if slots:
            return np.random.choice(slots)
        else:
            return self.idle()
        
    def roll(self):
        return 38*4*14 - 38*4
    
    def buy(self, shop_slot):
        x, y = (shop_slot + 1) // 4 + 9, (shop_slot + 1) % 4
        z = 28
        return z + 38*y + 38*4*x
    
    def sell(self, bench_slot):
        x, y = bench_slot // 4 + 7, bench_slot % 4
        z = 37
        return z + 38*y + 38*4*x
    
    def remove(self, x, y):            
        for i, c in enumerate(self.player.bench):
            if not c:
                return 28+i + 38*y + 38*4*x
        return 37 + 38*y + 38*4*x
    
    def place(self, bench_slot, x, y):
        if x == None or y == None:
            free = []
            for x_0, row in enumerate(self.player.board):
                for y_0, champ in enumerate(row):
                    if not champ:
                        free.append([x_0, y_0])
            x, y = free[np.random.randint(len(free))]                
        z = x*4 + y
        x, y = bench_slot // 4 + 7, bench_slot % 4
        
        return z + 38*y + 38*4*x
    
    def xp(self):
        return 38*4*14 - 38*3
            
            
    
            
                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        