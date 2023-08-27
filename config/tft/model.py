import math
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet, renormalize


class LinearResidualBlock(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 momentum=0.1, 
                 output_activation=nn.LeakyReLU):
        super().__init__()
        if in_channels != out_channels:
            self.skip=False
            self.pool = nn.AdaptiveAvgPool1d(out_channels)
        else:
            self.skip=True
        self.lin = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=momentum)
        self.act1 = nn.LeakyReLU()
        self.linout = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=momentum)
        self.act2 = output_activation()

    def forward(self, x):
        
        identity = x
        if not self.skip:
            identity = self.pool(identity)

        x = self.lin(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.linout(x)
        x = self.bn2(x)
        
        x += identity
        
        x = self.act2(x)
        
        return x

class ResidualMLP(nn.Module):
    def __init__(self, input_size, 
                 num_blocks, 
                 output_size, 
                 output_activation=nn.Identity,
                 activation=nn.LeakyReLU,
                 residual_output=True,
                 momentum=0.1,
                 init_zero=False):
        super().__init__()
        
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                s1 = input_size
            else:
                s1 = output_size
            if i < num_blocks - 1:
                act = activation
                self.blocks += [LinearResidualBlock(s1,
                                               output_size,
                                               output_activation=act)]
            else:
                act = output_activation
                if residual_output:
                    self.blocks += [LinearResidualBlock(s1,
                                               output_size,
                                               output_activation=act)]
                else:
                    self.blocks += nn.Sequential(nn.Linear(s1,
                                                           output_size),                                                                                       
                                                 act())
        
        self.blocks = nn.ModuleList(self.blocks)
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x)
        
        return x

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.LeakyReLU,
    momentum=0.1,
    init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)

def convkxk(in_channels, out_channels, k=3, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=k, stride=stride, padding=k//2, bias=False
    )
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False
    )

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, stride=1, momentum=0.1):
        super().__init__()
        self.correct_channels = False
        if in_channels != out_channels:
            self.conv = conv1x1(in_channels, out_channels)
            self.bn0 = nn.BatchNorm2d(out_channels, momentum=momentum)
            self.correct_channels = True
        self.conv1 = convkxk(in_channels, out_channels, k=k, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = convkxk(out_channels, out_channels, k=k, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

    def forward(self, x):
        identity = x
        if self.correct_channels:
            identity = self.conv(x)
            identity = self.bn0(identity)
            identity = nn.functional.leaky_relu(identity)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = nn.functional.leaky_relu(out)
        return out


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_players,
        vis_blocks_board_prevector,
        vis_blocks_board_premerge,
        vis_blocks_board_postmerge,
        vis_blocks_opponents,
        vis_blocks_state,
        champ_embed_channels,
        vis_hidden_channels,
        vis_embed_channels,
        vec_blocks,
        vec_channels,
        state_channels,
        momentum=0.1,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()

        self.vec_channels = vec_channels
        self.observation_shape = observation_shape
        self.vis_embed_channels = vis_embed_channels
        
        
        self.champ_embed = conv1x1(
            observation_shape[1],
            champ_embed_channels,
        )
        
        board_prevector_block_list = []
        for i in range(vis_blocks_board_prevector):
            if i:
                board_prevector_block_list.append(ResidualBlock(vis_hidden_channels, vis_hidden_channels, k=3, momentum=momentum))
            else:
                board_prevector_block_list.append(ResidualBlock(champ_embed_channels, vis_hidden_channels, k=3, momentum=momentum))

        self.board_prevector_blocks = nn.ModuleList(board_prevector_block_list)
        
        self.board_vector_embed_conv = nn.Conv2d(vis_hidden_channels, vis_embed_channels, kernel_size=1, bias=False)
        
        self.vec_embed_size = observation_shape[-1] * observation_shape[-2] * vec_channels
        
        self.board_premerge_padding = nn.ZeroPad2d((0,0,0,1))
        
        self.vec_concat_size = observation_shape[1] * observation_shape[-1] + \
            champ_embed_channels * 4 + observation_shape[-1] * (observation_shape[-2] - 1) * vis_embed_channels
        
        vec_block_list = []
        for i in range(vec_blocks+1):
            if i:
                vec_block_list.append(LinearResidualBlock(self.vec_embed_size, self.vec_embed_size))
            else:
                vec_block_list.append(nn.Sequential(nn.Linear(self.vec_concat_size, self.vec_embed_size),
                                                    nn.BatchNorm1d(self.vec_embed_size, momentum=momentum),
                                                    nn.LeakyReLU()))
        self.vec_blocks = nn.ModuleList(vec_block_list)
        
        board_premerge_block_list = []
        for i in range(vis_blocks_board_premerge):
            board_premerge_block_list.append(ResidualBlock(vis_hidden_channels, vis_hidden_channels, k=3, momentum=momentum))
        
        self.board_premerge_blocks = nn.ModuleList(board_premerge_block_list)
        
        board_postmerge_block_list = []
        for i in range(vis_blocks_board_postmerge):
            if i:
                board_postmerge_block_list.append(ResidualBlock(vis_hidden_channels, vis_hidden_channels, k=3, momentum=momentum))
            else:
                board_postmerge_block_list.append(ResidualBlock(vis_hidden_channels + vec_channels, vis_hidden_channels, k=3, momentum=momentum))
                
        self.board_postmerge_blocks = nn.ModuleList(board_postmerge_block_list)
        
        opponent_board_block_list = []
        for i in range(vis_blocks_opponents):
            if i:
                opponent_board_block_list.append(ResidualBlock(vis_hidden_channels, vis_hidden_channels, k=3, momentum=momentum))
            else:
                opponent_board_block_list.append(ResidualBlock(vis_hidden_channels * (num_players - 1), vis_hidden_channels, k=3, momentum=momentum))
                
        self.opponent_blocks = nn.ModuleList(opponent_board_block_list)
        
        state_block_list = []
        for i in range(vis_blocks_state):
            if i:
                state_block_list.append(ResidualBlock(vis_hidden_channels, vis_hidden_channels, k=3, momentum=momentum))
            else:
                state_block_list.append(ResidualBlock(vis_hidden_channels * 2, vis_hidden_channels, k=3, momentum=momentum))
                
        self.state_blocks = nn.ModuleList(state_block_list)
        self.num_boards = observation_shape[0]
        
        self.num_players = num_players

    def forward(self, x):
               
        opponents = []
        for i in range(self.num_boards):
            obs = x[:, i, :, :, :]
            board = obs[:, :, :-1, :]
            board = self.champ_embed(board)
            board_vec = torch.flatten(board, start_dim=-2)
            board_sum = torch.sum(board_vec[:, :, :28], dim=-1)
            bench_sum = torch.sum(board_vec[:, :, 28:28+9], dim=-1)
            shop_sum = torch.sum(board_vec[:, :, 28+9:28+9+5], dim=-1)
            ibench_sum = torch.sum(board_vec[:, :, 28+9+5:28+9+5+10], dim=-1)
            for block in self.board_prevector_blocks:
                board = block(board)
                
            vis_vec = self.board_vector_embed_conv(board)  
            vis_vec = torch.reshape(vis_vec, (-1, self.vis_embed_channels * (self.observation_shape[-2] - 1) * self.observation_shape[-1]))
            vec = obs[:, :, -1, :]
            vec = torch.reshape(vec, (-1, self.observation_shape[1] * self.observation_shape[-1]))
            vec = torch.concat([vec, board_sum, bench_sum, shop_sum, ibench_sum, vis_vec], dim=-1)
            for block in self.vec_blocks:
                vec = block(vec)
            vec = vec.view(-1, self.vec_channels, self.observation_shape[-2], self.observation_shape[-1])
                
            board = self.board_premerge_padding(board)
            for block in self.board_premerge_blocks:
                board = block(board)
            
            board = torch.concat([board, vec], dim=1)
            for block in self.board_postmerge_blocks:
                board = block(board)
                
            if i % self.num_players:
                opponents.append(board)
            else:
                player = board
                            
        opponents = torch.concat(opponents, dim=1)
        for block in self.opponent_blocks:
            opponents = block(opponents)
            
        state = torch.concat([player, opponents], dim=1)
        for block in self.state_blocks:
            state = block(state)
        
        return state

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_dynamics_blocks,
        dynamics_channels,
        num_reward_blocks,
        reward_channels,
        action_space_layers,
        fc_reward_layers,
        full_support_size,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        block_output_size_reward: int
            dim of flatten hidden states
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.dynamics_channels = dynamics_channels
        self.action_space_layers = action_space_layers
        self.lstm_hidden_size = lstm_hidden_size
    
        self.conv = conv3x3(dynamics_channels + action_space_layers, dynamics_channels)
        self.bn = nn.BatchNorm2d(dynamics_channels, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(dynamics_channels, dynamics_channels, momentum=momentum) for _ in range(num_dynamics_blocks)]
        )
      
        self.conv1x1_reward = nn.Conv2d(dynamics_channels, reward_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reward_channels, momentum=momentum)
        self.block_output_size_reward = observation_shape[-1] * observation_shape[-2] * reward_channels
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)
    
    def forward(self, x, reward_hidden):
        state = x[:,:-self.action_space_layers,:,:]
        x = self.conv(x)
        x = self.bn(x)
    
        x += state
        x = nn.functional.leaky_relu(x)
    
        for block in self.resblocks:
            x = block(x)
        state = x
    
        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.leaky_relu(x)
    
        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = nn.functional.leaky_relu(value_prefix)
        value_prefix = self.fc(value_prefix)
    
        return state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        block_output_size_value: int
            dim of flatten hidden states
        block_output_size_policy: int
            dim of flatten hidden states
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        """
        super().__init__()
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.bn_policy = nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_value = observation_shape[-1]*observation_shape[-2]*reduced_channels_value
        self.block_output_size_policy = observation_shape[-1]*observation_shape[-2]*reduced_channels_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = mlp(self.block_output_size_policy, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = nn.functional.leaky_relu(value)
        policy = self.conv1x1_policy(x)
        #policy = self.bn_policy(policy)
        #policy = nn.functional.leaky_relu(policy)
        value = value.view(-1, self.block_output_size_value)
        #value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        #policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        #policy = self.fc_policy(policy)
        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_shape,
        num_players,
        action_space_size,
        num_blocks,
        num_channels,
        board_embed_size,
        state_embed_size,
        vec_embed_size,
        fc_reward_layers,
        fc_value_layers,
        policy_layers,
        reward_support_size,
        value_support_size,
        downsample,
        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        bn_mt=0.1,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False
    ):
        """EfficientZero network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_reward: int
            channels of reward head
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        value_support_size: int
            dim of value output
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        bn_mt: float
            Momentum of BN
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        state_norm: bool
            True -> normalization for hidden states
        """
        super(EfficientZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size


        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_players,
            vis_blocks_board_prevector=4,
            vis_blocks_board_premerge=4,
            vis_blocks_board_postmerge=4,
            vis_blocks_opponents=4,
            vis_blocks_state=16,
            champ_embed_channels=256,
            vis_hidden_channels=num_channels,
            vis_embed_channels=32,
            vec_blocks=12,
            vec_channels=32,
            state_channels=256,
            )

        self.dynamics_network = DynamicsNetwork(
            observation_shape,
            num_dynamics_blocks=16,
            dynamics_channels=num_channels,
            num_reward_blocks=4,
            reward_channels=32,
            action_space_layers=2,
            fc_reward_layers=fc_reward_layers,
            full_support_size=reward_support_size,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            observation_shape,
            action_space_size,
            4,
            num_channels,
            32,
            38,
            fc_value_layers,
            fc_reward_layers,
            value_support_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        # projection
        if downsample:
            in_dim = num_channels * math.ceil(observation_shape[-2] / 16) * math.ceil(observation_shape[-1] / 16)
        else:
            in_dim = num_channels * math.ceil(observation_shape[-2]) * math.ceil(observation_shape[-1])
        self.porjection_in_dim = in_dim
        self.projection = nn.Sequential(
            nn.Linear(self.porjection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.LeakyReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.LeakyReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.LeakyReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def prediction(self, encoded_state):     
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        
        
        action_one_hot = (
            torch.zeros(
                (
                    encoded_state.shape[0],
                    2,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        
        
        for i in range(encoded_state.shape[0]):
            a = action[i].data[0]
            x, y, z = a // 38 // 4, a // 38 % 4, a % 38
            action_one_hot[i, 0, x, y] = 1
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
            action_one_hot[i, 1, dest_x, dest_y] = 1
        
            
        '''
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        '''
        #onehot = torch.nn.functional.one_hot(torch.squeeze(action), self.action_space_size)
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.reshape(hidden_state.shape[0], self.porjection_in_dim)
        #hidden_state = hidden_state.view(-1, self.porjection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

