import torch
import numpy as np

from core.config import BaseConfig
from core.dataset import Transforms
from .model import EfficientZeroNet
from Simulator.tft_simulator import TFT_Simulator


class TFTConfig(BaseConfig):
    def __init__(self):
        super(TFTConfig, self).__init__(
            training_steps=1000000,
            last_steps=20000,
            test_interval=5000,
            log_interval=1000,
            vis_interval=1000,
            test_episodes=4,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=5000,
            max_moves=1000,
            test_max_moves=2000,
            history_length=800,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=64,
            max_moves_considered=32,
            batch_size=128,
            td_steps=5,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=True,
            init_zero=True,
            clip_reward=False,
            # storage efficient
            cvt_string=False,
            image_based=True,
            # lr scheduler
            lr_warm_up=25000,
            lr_init=0.00003,
            lr_decay_rate=0.1,
            lr_decay_steps=1000000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=5e4,
            total_transitions=100 * 1000,
            transition_num=1.5e5,
            # frame skip & stack observation
            gray_scale=False,
            frame_skip=1,
            stacked_observations=1,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=1,
            policy_loss_coeff=1,
            consistency_coeff=1,
            commitment_loss_coeff=2,
            chance_loss_coeff=1,
            # reward sum
            lstm_hidden_size=128,
            lstm_horizon_len=10,
            # siamese
            proj_hid=512,
            proj_out=512,
            pred_hid=256,
            pred_out=512,)
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip
        self.num_models = 1
        self.model_switch_interval = 5000000000
        self.num_random_actors = 0
        
        self.num_chance_tokens = 4
        self.learned_agent_actions_start = 1

        self.start_transitions = self.start_transitions // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)
        self.easy_mode = False
        self.bn_mt = 0.1
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.board_embed_size = 1024  # x36 Number of channels in reward head
        self.state_embed_size = 2048  # x36 Number of channels in value head
        self.vec_embed_size = 128  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [256]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [256]
        self.resnet_fc_chance_layers = [256] # Define the hidden layers in the value head of the prediction network
        self.resnet_policy_layers = 31  # Define the depth in the 3D policy head of the prediction network
        self.downsample = False  # Downsample observations before representation network (See paper appendix Network Architecture)

    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 90000: #0.25 * (self.training_steps):
                return 0.5
            elif trained_steps < 180000: #0.5 * (self.training_steps):
                return 0.25
            elif trained_steps < 360000 * (self.training_steps):
               return 0.13
            else:
                return 0.08
        else:
            return 1.0
    
    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        game = self.new_game()
        self.num_players = game.num_players
        self.image_channel = game.image_channel
        obs_shape = (self.num_players, self.image_channel, game.obs_shape[1],game.obs_shape[2])
        self.obs_shape = (self.num_players * self.stacked_observations, self.image_channel, obs_shape[-2], obs_shape[-1])        
        self.action_space_size = game.action_space_size()

    def record_best_actions(self, actions, dists, env):
        if env.log:
            for player, dist in dists.items():
                action = actions[player]
                best_action = np.argmax(dist)
                s = np.sum(dist)
                val = dist[best_action] / s
                a_val = dist[action] / s
                env.PLAYERS[player].print("chosen action: {} / {}, best action: {} / {}".format(action, round(a_val, 2), best_action, round(val, 2)))

    def record_tokens(self, tokens, env):
        if env.log:
            for player, token in tokens.items():
                best_action = np.argmax(token)
                env.PLAYERS[player].print("chance token assigned: {}".format(best_action))


    def get_uniform_network(self):
        return EfficientZeroNet(
            observation_shape=self.obs_shape,
            num_players=self.num_players,
            action_space_size=self.action_space_size,
            num_chance_tokens=self.num_chance_tokens,
            num_blocks=self.blocks,
            num_channels=self.channels,
            board_embed_size=self.board_embed_size,
            state_embed_size=self.state_embed_size,
            vec_embed_size=self.vec_embed_size,
            fc_reward_layers=self.resnet_fc_reward_layers,
            fc_value_layers=self.resnet_fc_value_layers,
            fc_chance_layers=self.resnet_fc_chance_layers,
            policy_layers=self.resnet_policy_layers,
            reward_support_size=self.reward_support.size,
            value_support_size=self.value_support.size,
            downsample=self.downsample,
            inverse_value_transform=self.inverse_value_transform,
            inverse_reward_transform=self.inverse_reward_transform,
            lstm_hidden_size=self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm
            )

    def new_game(self, log=False, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        if test:
            if final_test:
                max_moves = 108000 // self.frame_skip
            else:
                max_moves = self.test_max_moves
            env = TFT_Simulator(env_config=self, log=log)
        else:
            env = TFT_Simulator(env_config=self, log=log)

        return env

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def tft_augmentation(self, obs_batch):
        '''
        this augmentation swaps positions of opponent boards
        in the observation space.
        
        obs_batch shape is (batch_size, channel * (unroll_steps + 1), obs height, obs width)
        '''   
        #obs_batch = obs_batch.copy()
        if self.num_players > 2:
            for b in range(obs_batch.shape[0]):
                for to_swap in np.random.choice(np.arange(1, self.num_players), size=(np.random.randint(4),2), replace=False):
                    temp = obs_batch[b, :, to_swap[0], :, :, :]
                    obs_batch[b, :, to_swap[0], :, :, :] = \
                        obs_batch[b, :, to_swap[1], :, :, :]                        
                    obs_batch[b, :, to_swap[1], :, :, :] = temp            
        return obs_batch

    def transform(self, images):
        return self.transforms.transform(images)


game_config = TFTConfig()