import math
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet, renormalize


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
    def __init__(self, in_channels, out_channels, k=3, downsample=None, stride=1, momentum=0.1):
        super().__init__()
        self.conv1 = convkxk(in_channels, out_channels, k=k, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = convkxk(out_channels, out_channels, k=k, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.leaky_relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2, momentum=momentum)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2, downsample=self.conv2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_players,
        num_blocks,
        num_channels,
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

        num_channels_per_board = num_channels // 2

        self.conv_self_board = conv1x1(
            observation_shape[1],
            num_channels_per_board,
        )
        
        self.conv_board = conv1x1(
            observation_shape[1],
            num_channels_per_board,
        )
        
        self.num_boards = observation_shape[0]
        
        self.num_players = num_players
        self.bn_self_board = nn.BatchNorm2d(num_channels_per_board, momentum=momentum)
        self.bn_board = nn.BatchNorm2d(num_channels_per_board, momentum=momentum)
        
        self.conv_self = conv1x1(
            int(num_channels_per_board * self.num_boards / self.num_players),
            num_channels,
        )
        self.conv_opp = conv1x1(
            int(num_channels_per_board * (self.num_boards - self.num_boards / self.num_players)),
            num_channels,
        )
        
        self.conv = conv1x1(
            num_channels*2,
            num_channels,
        )
        
        self.bn_self = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.bn_opp = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
        
        self.resblocks_self_board = nn.ModuleList(
            [ResidualBlock(num_channels_per_board, num_channels_per_board, k=7, momentum=momentum) for _ in range(num_blocks)]
        )
        
        self.resblocks_board = nn.ModuleList(
            [ResidualBlock(num_channels_per_board, num_channels_per_board, k=7, momentum=momentum) for _ in range(num_blocks)]
        )
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, k=7, momentum=momentum) for _ in range(num_blocks)]
        )


    def forward(self, x):
        
        s = []
        o = []
        for i in range(self.num_boards):
            if i % self.num_players:
                y = self.conv_board(x[:, i, :, :, :])
                y = self.bn_board(y)
                y = nn.functional.leaky_relu(y)
    
                for block in self.resblocks_board:
                    y = block(y)
                o.append(y)
            else:
                z = self.conv_self_board(x[:, i, :, :, :])
                z = self.bn_self_board(z)
                z = nn.functional.leaky_relu(z)
    
                for block in self.resblocks_self_board:
                    z = block(z)
                s.append(z)
                
        
        s = torch.concat(s, dim=1)
        s = self.conv_self(s)
        s = self.bn_self(s)
        s = nn.functional.leaky_relu(s)
        
        o = torch.concat(o, dim=1)
        o = self.conv_opp(o)
        o = self.bn_opp(o)
        o = nn.functional.leaky_relu(o)
        
        x = torch.concat([s, o], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = nn.functional.leaky_relu(x)
        for block in self.resblocks:
            x = block(x)
        return x

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
        num_blocks,
        num_channels,
        action_space_size,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
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
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.lstm_hidden_size = lstm_hidden_size

        self.conv = conv3x3(num_channels + 2, num_channels)
        self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, k=7, momentum=momentum) for _ in range(num_blocks)]
        )

        self.reward_resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, k=7, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = nn.Conv2d(num_channels, reduced_channels_reward, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        state = x[:,:-2,:,:]
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
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        policy_layers,
        full_support_size,
        block_output_size_value,
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
            [ResidualBlock(num_channels, num_channels, k=7, momentum=momentum) for _ in range(num_blocks)]
        )
        self.action_space_size = action_space_size
        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_layers, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = nn.functional.leaky_relu(value)
        policy = self.conv1x1_policy(x)
        policy = nn.functional.leaky_relu(policy)
        value = value.reshape(value.shape[0], self.block_output_size_value)
        #value = value.view(-1, self.block_output_size_value)
        policy = policy.moveaxis(1, -1)
        policy = policy.reshape(policy.shape[0], self.action_space_size)
        #policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        return policy, value


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_shape,
        num_players,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
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
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[-2] / 16)
                * math.ceil(observation_shape[-1] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[-2] * observation_shape[-1])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[-2] / 16)
                * math.ceil(observation_shape[-1] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[-2] * observation_shape[-1])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[-2] / 16)
                * math.ceil(observation_shape[-1] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[-2] * observation_shape[-1])
        )

        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_players,
            num_blocks,
            num_channels,
            momentum=bn_mt,
        )

        self.dynamics_network = DynamicsNetwork(
            num_blocks,
            num_channels,
            self.action_space_size,
            reduced_channels_reward,
            fc_reward_layers,
            reward_support_size,
            block_output_size_reward,
            lstm_hidden_size=lstm_hidden_size,
            momentum=bn_mt,
            init_zero=self.init_zero,
        )

        self.prediction_network = PredictionNetwork(
            action_space_size,
            max(num_blocks // 2, 1),
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            policy_layers,
            value_support_size,
            block_output_size_value,
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
            x, y, z = a // 38 // 9, a // 38 % 9, a % 38
            action_one_hot[i, 0, x, y] = 1
            if x >= 4 and y >= 4:
                dest_x, dest_y = x, y
            else:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                elif z < 37:
                    x_bench = z - 28
                    dest_x, dest_y = x_bench // 5, 4 + x_bench % 5                   
                else:
                    dest_x, dest_y = 6, 8
            action_one_hot[i, 1, dest_x, dest_y] = 1
            
            
        '''
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        '''
        
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

