import math
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet, renormalize


class LinearResidualBlock(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 ln_shape,
                 output_activation=nn.Identity,
                 init_zero=False):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut=True
            self.pool = nn.AdaptiveAvgPool1d(out_channels)
        else:
            self.shortcut=False
        self.ln = nn.LayerNorm(ln_shape)
        self.act = nn.GELU()
        self.lin = nn.Linear(in_channels, 
                             out_channels,
                             bias=False)
        self.output_activation = output_activation()
        if init_zero:
            self.lin.weight.data.fill_(0)
        
    def forward(self, x):
        
        x1 = x
        if self.shortcut:
            x = self.pool(x)
            
        x1 = self.ln(x1)
        x1 = self.act(x1)
        x1 = self.lin(x1)
        
        x = x + x1
        
        return self.output_activation(x)
    
def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.GELU,
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
            layers.append(LinearResidualBlock(sizes[i], sizes[i + 1], sizes[i]))
        else:
            layers += [nn.LayerNorm(sizes[i]),
                       nn.GELU(),
                       nn.Linear(sizes[i], sizes[i + 1], bias=False)]
    if init_zero:
        layers[-1].weight.data.fill_(0)
                
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
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 shape,
                 k=3, 
                 stride=1, 
                 activation=nn.GELU
                 ):
        super().__init__()
        self.correct_channels = False
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.correct_channels = True
            
        self.ln = nn.LayerNorm(shape)
        self.act = activation()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding=k//2, bias=False)

    def forward(self, x):
        identity = x
        if self.correct_channels:
            identity = self.shortcut_conv(x)
            
        x = self.ln(x)
        x = self.act(x)            
        x = self.conv(x)
        x = x + identity

        return x

class UnitEncoding(nn.Module):
    def __init__(self, 
                 unit_embedding_size,
                 item_embedding_size, 
                 origin_embedding_size,
                 output_size=64,
                 ):
        super().__init__()        
        self.unit_embedding = nn.Embedding(72, unit_embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(60, item_embedding_size, padding_idx=0)
        self.origin_embedding = nn.Embedding(27, origin_embedding_size, padding_idx=0)
        
        self.mlp = nn.Sequential(
                    nn.Linear(unit_embedding_size + item_embedding_size + origin_embedding_size + 41, output_size * 4),
                    nn.GELU(),
                    nn.Linear(output_size * 4, output_size),)
               
    def forward(self, x):
        layers = []
        item_layers = []
        for i in range(3):
            item_layers.append(self.item_embedding(x[:,:,i].long()).unsqueeze(-2))
        item_layers = torch.concat(item_layers, dim=-2)
        item_layers = torch.sum(item_layers, dim=-2)
        layers.append(item_layers)
        origin_layers = []
        for i in range(3, 10):
            origin_layers.append(self.origin_embedding(x[:,:,i].long()).unsqueeze(-2))
        origin_layers = torch.concat(origin_layers, dim=-2)
        origin_layers = torch.sum(origin_layers, dim=-2)
        layers.append(origin_layers)
        layers.append(self.unit_embedding(x[:,:,10].long()))
        layers.append(torch.div(x[:, :, 11:], 255.))
        layers = torch.concat(layers, dim=-1)
        layers = self.mlp(layers)
        return layers          

class VectorEncoding(nn.Module):
    def __init__(self, 
                 origin_embedding_size,
                 output_size):
        super().__init__()
        self.origin_embedding = nn.Embedding(27, origin_embedding_size, padding_idx=0)
        self.mlp = nn.Sequential(
                    nn.Linear(52*4 - 20 + origin_embedding_size, output_size * 4),
                    nn.GELU(),
                    nn.Linear(output_size * 4, output_size),)
               
    def forward(self, x):
        layers = []
        origin_layers = []
        for i in range(10):
            origin_layers.append(torch.mul(self.origin_embedding(x[:,i*2].long()), x[:, i*2+1].unsqueeze(-1)).unsqueeze(-2))
        origin_layers = torch.concat(origin_layers, dim=-2)
        origin_layers = torch.sum(origin_layers, dim=-2)
        layers.append(origin_layers)
        layers.append(torch.div(x[:,20:], 255.))
        layers = torch.concat(layers, dim=-1)
        layers = self.mlp(layers)
        return layers
    
class AttentionResBlock(nn.Module):
    def __init__(self, 
                 embedding_size, 
                 num_heads,
                 length,
                 ):
        super().__init__()
        
        self.pre_ln = nn.LayerNorm((length, embedding_size))
        self.post_ln = nn.LayerNorm((length, embedding_size))
        self.attn = nn.MultiheadAttention(embedding_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.GELU(),
            nn.Linear(embedding_size * 4, embedding_size),)
               
    def forward(self, x, kv=None):

        if kv == None:
            kv = x
        x = self.pre_ln(x)
        kv = self.pre_ln(kv)
        a, _ = self.attn(x, kv, kv, need_weights=False)
        x = x + a
        m = self.mlp(self.post_ln(x))
        x = x + m
        return x
    
class ConvertToRegister(nn.Module):
    def __init__(self, 
                 embedding_size, 
                 num_registers,
                 num_resblocks,
                 length
                 ):
        super().__init__()
        
        self.num_registers = num_registers
        self.embedding_size = embedding_size
        self.flat_size = embedding_size * length
        register_size = embedding_size * num_registers
        
        resblocks = []
        for i in range(num_resblocks):
            if i:
                resblocks.append(LinearResidualBlock(self.flat_size, 
                                                        self.flat_size,
                                                        self.flat_size))

        resblocks.append(LinearResidualBlock(self.flat_size, 
                                                register_size,
                                                self.flat_size))
        self.resblocks = nn.ModuleList(resblocks)
               
    def forward(self, x):

        x = x.view(-1, self.flat_size)        

        for block in self.resblocks:
            x = block(x)
        
        x = x.view(-1, self.num_registers, self.embedding_size)

        return x        



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 52):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        fill = torch.zeros((x.size(0), x.size(1), 2)).to(x.device)
        fill = fill + self.pe[:, :x.size(1), :]
        x = torch.concat([x, fill], dim=-1)
        return x


#straight-through estimator is used during the backward to allow the gradients to flow only to the encoder during the backpropagation.
class Onehot_argmax(torch.autograd.Function):
    #more information at : https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    @staticmethod
    def forward(ctx, input):
        #since the codebook is constant ,we can just use a transformation. no need to create a codebook and matmul c_e_t and codebook for argmax
        return torch.zeros_like(input).scatter_(-1, torch.argmax(input, dim=-1,keepdim=True), 1.)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output        

# Encode the observations into hidden states and chance outcome tokens
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_players,
        num_board_cnn_resblocks,
        unit_embed_channels,
        vec_blocks,
        vec_channels,
        state_channels,
        reduced_channels_chance,
        fc_chance_layers, 
        num_chance_tokens,
        num_opponent_registers,
        num_history_registers,
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
        
        self.unit_embed_channels = unit_embed_channels
        self.vec_channels = vec_channels
        
        self.observation_shape = observation_shape
        self.state_length = observation_shape[-1] * observation_shape[-2]
        
        self.num_opponent_registers = num_opponent_registers
        self.num_history_registers = num_history_registers
        
        # encodes all units and items
        self.unit_encoding = UnitEncoding(unit_embedding_size=32,
                                        item_embedding_size=16, 
                                        origin_embedding_size=16,
                                        output_size=unit_embed_channels)
        
        
        '''
        # self attention resnet applied to the board
        self.board_sa = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=28)
        
        # CNN resnet applied to the board
        board_cnn_blocks = []
        for i in range(num_board_cnn_resblocks):
            board_cnn_blocks.append(ResidualBlock(unit_embed_channels, 
                                                    unit_embed_channels, 
                                                    (unit_embed_channels,
                                                     7,
                                                     self.observation_shape[-1]),
                                                    k=3))
        self.board_cnn_blocks = nn.ModuleList(board_cnn_blocks)
        '''
        
        # self attention resblocks applied to the board, benches, and shop
        units_sa_blocks = []
        for i in range(3):
            units_sa_blocks.append(AttentionResBlock(embedding_size=unit_embed_channels,
                                                     num_heads=2,
                                                     length=self.state_length - self.observation_shape[-1]))
        self.units_sa_blocks = nn.ModuleList(units_sa_blocks)
        
        # encodes the non-units state vector
        self.vec_encoding = VectorEncoding(origin_embedding_size=16,
                                           output_size=256)
        
        # embeds the units as a linear vector
        self.lin_units_embedding = nn.Sequential(
                        nn.LayerNorm((self.state_length - self.observation_shape[-1]) * self.unit_embed_channels),
                        nn.GELU(),
                        nn.Linear((self.state_length - self.observation_shape[-1]) * self.unit_embed_channels, 
                                  256,
                                  bias=False))
        
        # calculate the size of the embedded state vector
        self.vec_embed_size = self.state_length * vec_channels       
        
        # calculate the state vector after concatenation with the board/bench/shop embeddings
        self.vec_concat_size = 256 + unit_embed_channels * 4 + 256
        
        # linear resnet for the concatenated state vector
        vec_block_list = []
        for i in range(vec_blocks+1):
            if i:
                vec_block_list.append(LinearResidualBlock(self.vec_embed_size, 
                                                          self.vec_embed_size,
                                                          self.vec_embed_size))
            else:
                vec_block_list.append(nn.Sequential(
                    nn.LayerNorm(self.vec_concat_size),
                    nn.GELU(),
                    nn.Linear(self.vec_concat_size, 
                              self.vec_embed_size,
                              bias=False),
                    ))
        self.vec_resblocks = nn.ModuleList(vec_block_list)
        
        # embed the vector to fill out the units tensor
        self.vec_fill = nn.Sequential(
                        nn.LayerNorm(self.vec_embed_size),
                        nn.GELU(),
                        nn.Linear(self.vec_embed_size, 
                                  self.observation_shape[-1] * self.unit_embed_channels,
                                  bias=False))        
        
        # self attention resnet applied to the board, benches, and shop after
        # merging with the state vector
        postmerge_sa_blocks = []
        for i in range(3):
            postmerge_sa_blocks.append(AttentionResBlock(embedding_size=unit_embed_channels,
                                                     num_heads=2,
                                                     length=self.state_length))         
        self.postmerge_sa_blocks = nn.ModuleList(postmerge_sa_blocks)

        # apply cross attention to compare player boards
        opponent_cross_blocks = []
        for i in range(2):
            opponent_cross_blocks.append(AttentionResBlock(embedding_size=unit_embed_channels,
                                                     num_heads=2,
                                                     length=self.state_length))         
        self.opponent_cross_blocks = nn.ModuleList(opponent_cross_blocks)
        
        self.opponent_register = ConvertToRegister(embedding_size=unit_embed_channels, 
                                                 num_registers=num_opponent_registers,
                                                 num_resblocks=2,
                                                 length=self.state_length
                                                 )
        
        l = self.state_length + num_opponent_registers * (num_players - 1)
        
        # self attention resnet applied after cross attention
        postcross_sa_blocks = []
        for i in range(3):
            postcross_sa_blocks.append(AttentionResBlock(embedding_size=unit_embed_channels,
                                                     num_heads=2,
                                                     length=l))         
        self.postcross_sa_blocks = nn.ModuleList(postcross_sa_blocks)
        
        frame_cross_blocks = []
        for i in range(3):
            frame_cross_blocks.append(AttentionResBlock(embedding_size=unit_embed_channels,
                                                     num_heads=2,
                                                     length=l))         
        self.frame_cross_blocks = nn.ModuleList(frame_cross_blocks)
        
        self.frame_register = ConvertToRegister(embedding_size=unit_embed_channels, 
                                                 num_registers=num_history_registers,
                                                 num_resblocks=2,
                                                 length=l
                                                 )        
        l += num_history_registers
        final_sa_blocks = []
        for i in range(10):
            final_sa_blocks.append(AttentionResBlock(embedding_size=unit_embed_channels,
                                                     num_heads=2,
                                                     length=l))         
        self.final_sa_blocks = nn.ModuleList(final_sa_blocks)       
                                                             
        self.num_boards = observation_shape[0]
        
        self.num_players = num_players
        
        self.num_stacked_frames = self.num_boards // self.num_players
        
        self.chance_conv = LinearResidualBlock(unit_embed_channels, reduced_channels_chance, unit_embed_channels)     
        
        self.block_output_size_chance = l * reduced_channels_chance

        self.fc_chance = mlp(self.block_output_size_chance, 
                            fc_chance_layers, 
                            num_chance_tokens, 
                            )
        
        
    def forward(self, x):
        
        # input shape is [batch_size, stacked_frames * num_players, 14, 4, 54]
               
        # separate units from vector
        units, vec = x[:, :, :-1, :, :], x[:, :, -1, :, :]
        
        # reshape units along batch dimension for vectorized encoding
        # new shape is [batch_size * stacked_frames * num_players, 52, 54]
        units = torch.reshape(units, (-1, self.state_length - self.observation_shape[-1], self.observation_shape[-3])) 
        
        # encode units
        units = self.unit_encoding(units)
        
        #units = self.pos_encoding(units)
        
        
        '''
        # apply Self-Attention and CNN to the boards
        
        # isolate board units
        board, other_units = units[:, :28, :], units[:, 28:, :]
        
        # apply SA
        board = self.board_sa(board)
        
        # move axes and change shape for CNN
        # new shape is [batch_size * stacked_frames * num_players, channels, 7, 4]
        board = torch.moveaxis(board, -1, -2)
        board = torch.reshape(board, (-1, self.unit_embed_channels, 7, self.observation_shape[-1]))       
        
        # CNN resnet
        for block in self.board_cnn_blocks:
            board = block(board)

        # reshape the board for SA
        # new shape is [batch_size * stacked_frames * num_players, 28, channels]
        board = torch.moveaxis(board, -3, -1)
        board = torch.reshape(board, (-1, 28, self.unit_embed_channels))
        
        # concatenate the board with the benches/shops for SA
        # new shape is [batch_size * stacked_frames * num_players, 52, channels]
        units = torch.concat([board, other_units], dim=-2)
        '''
        
        
        # apply SA to the units
        for block in self.units_sa_blocks:
            units = block(units)
        
        # convert the player state vector into learned registers
        # https://arxiv.org/pdf/2309.16588.pdf
        
        # flatten the player state vector
        vec = torch.reshape(vec, (-1, self.observation_shape[-3] * self.observation_shape[-1]))
        
        # encode the player state vector
        vec = self.vec_encoding(vec)        
        
        # make a linear embedding of the units
        lin_units = torch.reshape(units, (-1, (self.state_length - self.observation_shape[-1]) * self.unit_embed_channels))
        lin_units = self.lin_units_embedding(lin_units)        
                
        # sum unit embeddings for concatenation with the linear state vector
        board_sum = torch.sum(units[:, :28, :], dim=-2)
        bench_sum = torch.sum(units[:, 28:28+9, :], dim=-2)
        shop_sum = torch.sum(units[:, 28+9:28+9+5, :], dim=-2)
        ibench_sum = torch.sum(units[:, 28+9+5:28+9+5+10, :], dim=-2)

        # concatenate the player state vector with the visual embedding and the board region unit embedding sums
        vec = torch.concat([vec, board_sum, bench_sum, shop_sum, ibench_sum, lin_units], dim=-1)
        
        # linear resnet for the full state vector
        for block in self.vec_resblocks:
            vec = block(vec)
            
        # append the units tensor with tokens from a vector embedding
        # new shape is [batch_size * stacked_frames * num_players, 56, channels]
        vec_fill = self.vec_fill(vec)
        vec_fill = torch.reshape(vec_fill, (-1, self.observation_shape[-1], self.unit_embed_channels))        
        player_state = torch.concat([units, vec_fill], dim=-2)
        
        
        '''
        # reshape the vector for channels concatenation
        vec = torch.reshape(vec, (-1, self.state_length, self.vec_channels))
        player_state = torch.concat([player_state, vec], dim=-1)
        
        # reduct the player state channels
        player_state = self.post_merge_lin(player_state)
        '''              
        
        # apply SA to the player state
        for block in self.postmerge_sa_blocks:
            player_state = block(player_state)
                
        # break out board axis
        # new shape [batch_size * stacked_frames, num_players, 56, channels]
        frame_state = player_state.view(-1, self.num_players, self.state_length, self.unit_embed_channels)
                
        # perform cross attention on each opponent state and convert the cross
        # into learned registers
        player_state = frame_state[:, 0, :, :]        
        opponent_registers = []        
        for i in range(1, self.num_players):
            opponent_state = frame_state[:, i, :, :]
            cross = player_state
            for block in self.opponent_cross_blocks:
                cross = block(cross, opponent_state)
            opponent_registers.append(self.opponent_register(cross))
        
        player_state = torch.concat([player_state] + opponent_registers, dim=-2)
        
        # apply SA to the player state
        for block in self.postcross_sa_blocks:
            player_state = block(player_state)
                
        # break out frame axis
        # new shape [batch_size, stacked_frames, 56, channels]
        new_length = self.state_length + (self.num_players-1)*self.num_opponent_registers
        state = player_state.view(-1, 
                                  self.num_stacked_frames, 
                                  new_length, 
                                  self.unit_embed_channels)
        
        # perform cross attention on each previous frame and convert the cross
        # into learned registers
        current_frame_state = state[:, -1, :, :]      
        cross = state[:, 0, :, :]
        
        for i in range(1, self.num_stacked_frames):
            nex = state[:, i, :, :]
            for block in self.frame_cross_blocks:
                cross = block(cross, nex)
            
        frame_registers = self.frame_register(cross)
        
        state = torch.concat([current_frame_state, frame_registers], dim=-2)
        
        for block in self.final_sa_blocks:
            state = block(state)
        
        chance = self.chance_conv(state)
        chance = chance.view(-1, self.block_output_size_chance)
        chance = self.fc_chance(chance)
        
        c_e_t = torch.nn.Softmax(-1)(chance)
        # print(("chance minmax:", chance.min(), chance.max(), "softmax minmax:", c_e_t.min(), c_e_t.max()))
        c_t = Onehot_argmax.apply(c_e_t)
        
        return state, c_t, c_e_t

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean




# Predict States given current afterstates and chance tokens
class StateDynamicsNetwork(nn.Module):
    def __init__(
        self,
        length,
        channels,
        reward_channels,
        outcome_space_layers,
        fc_reward_layers,
        full_support_size,
        lstm_hidden_size=64,
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
        
        self.outcome_space_layers = outcome_space_layers
        '''
        self.action_sa = AttentionResNet(embedding_size=channels+outcome_space_layers, 
                                             num_pre_attn_resblocks=2, 
                                             num_attn_resblocks=3, 
                                             num_attn_heads=2, 
                                             num_post_attn_resblocks=2, 
                                             length=length)
        '''
        self.act_conv = LinearResidualBlock(channels+outcome_space_layers, channels, (length, channels+outcome_space_layers))
                
        dynamics_sa_blocks = []
        for i in range(8):
            dynamics_sa_blocks.append(AttentionResBlock(embedding_size=channels,
                                                     num_heads=2,
                                                     length=length))         
        self.dynamics_sa_blocks = nn.ModuleList(dynamics_sa_blocks)        
      
        self.reward_conv = LinearResidualBlock(channels, reward_channels, (length, channels))
        
        self.block_output_size_reward = length * reward_channels
               
        self.fc = mlp(self.block_output_size_reward, 
                        fc_reward_layers, 
                        full_support_size, 
                        init_zero=True, 
                        )

    def forward(self, x, reward_hidden):
        
        x = self.act_conv(x)
    
        for block in self.dynamics_sa_blocks:
            x = block(x)
            
        state = x
    
        x = self.reward_conv(state)
    
        x = x.view(-1, self.block_output_size_reward) #.unsqueeze(0)
        #value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        #value_prefix = value_prefix.squeeze(0)
        value_prefix = self.fc(x)
    
        return state, None, value_prefix

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


# Predict afterstates given current states and actions
class AfterstateDynamicsNetwork(nn.Module):
    def __init__(
        self,
        length,
        channels,
        reward_channels,
        action_space_layers,
        fc_reward_layers,
        full_support_size,
        lstm_hidden_size=64,
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
        
        self.action_space_layers = action_space_layers

        self.act_conv = LinearResidualBlock(channels+action_space_layers, channels, (length, channels+action_space_layers))
        
        dynamics_sa_blocks = []
        for i in range(8):
            dynamics_sa_blocks.append(AttentionResBlock(embedding_size=channels,
                                                     num_heads=2,
                                                     length=length))         
        self.dynamics_sa_blocks = nn.ModuleList(dynamics_sa_blocks)  

    def forward(self, x, reward_hidden):

        x = self.act_conv(x)
    
        for block in self.dynamics_sa_blocks:
            x = block(x)
        
        '''
        x = self.reward_conv(state)
    
        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.fc(value_prefix)
        '''
    
        return x, None, None

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
    
    
# predict the value and chance probabilities given afterstates
class AfterstatePredictionNetwork(nn.Module):
    def __init__(
        self,
        length,
        num_chance_tokens,
        num_channels,
        reduced_channels_value,
        reduced_channels_chance,
        fc_value_layers,
        fc_chance_layers,
        full_support_size,
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

        self.value_conv = LinearResidualBlock(num_channels, reduced_channels_value, (length, num_channels))
        
        self.policy_conv = LinearResidualBlock(num_channels, reduced_channels_chance, (length, num_channels))
        
        self.block_output_size_value = length * reduced_channels_value
        self.block_output_size_policy = length * reduced_channels_chance
        self.fc_value = mlp(self.block_output_size_value, 
                            fc_value_layers, 
                            full_support_size, 
                            init_zero=True, 
                            )
        self.fc_policy = mlp(self.block_output_size_policy, 
                             fc_chance_layers, 
                             num_chance_tokens, 
                             init_zero=True, 
                            )

    def forward(self, x):
        value = self.value_conv(x)
        policy = self.policy_conv(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


# predict the values and action probabilities given states
class StatePredictionNetwork(nn.Module):
    def __init__(
        self,
        length,
        position_length,
        action_space_size,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,       
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
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
        
        self.position_length = position_length

        sa_blocks = []
        for i in range(3):
            sa_blocks.append(AttentionResBlock(embedding_size=num_channels,
                                                     num_heads=2,
                                                     length=length))         
        self.sa_blocks = nn.ModuleList(sa_blocks)

        self.value_conv = LinearResidualBlock(num_channels, reduced_channels_value, (length, num_channels))

        policy_conv = [nn.LayerNorm((position_length,
                                            num_channels,
                                            )),
                            nn.GELU(),
                            nn.Linear(num_channels, 
                                      reduced_channels_policy, 
                                      bias=False
                                      )]
        
        policy_conv[-1].weight.data.fill_(0)
        self.policy_conv = nn.Sequential(*policy_conv)
        
        self.block_output_size_value = length * reduced_channels_value
        self.block_output_size_policy = position_length * reduced_channels_policy
        self.fc_value = mlp(self.block_output_size_value, 
                            fc_value_layers, 
                            full_support_size, 
                            init_zero=True, 
                            )
        #self.fc_policy = mlp(self.block_output_size_policy, 
        #                     fc_policy_layers, 
        #                     action_space_size, 
        #                     init_zero=init_zero, 
        #                     momentum=momentum
        #                    )


    def forward(self, x):
        for block in self.sa_blocks:
            x = block(x)
        value = self.value_conv(x)
        policy = self.policy_conv(x[:, :self.position_length, :])
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        #policy = self.fc_policy(policy)
        
        return policy, value

class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_shape,
        num_players,
        action_space_size,
        num_chance_tokens,
        num_blocks,
        num_channels,
        board_embed_size,
        state_embed_size,
        vec_embed_size,
        fc_reward_layers,
        fc_value_layers,
        fc_chance_layers,
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
        self.num_chance_tokens = num_chance_tokens

        self.hidden_length = observation_shape[-1] * observation_shape[-2] + 2*(num_players-1) + 2

        self.representation_network = RepresentationNetwork(            
            observation_shape=observation_shape,
            num_players=num_players,
            num_board_cnn_resblocks=8,
            unit_embed_channels=num_channels,
            vec_blocks=2,
            vec_channels=16,
            state_channels=num_channels,
            reduced_channels_chance=16,
            fc_chance_layers=fc_chance_layers, 
            num_chance_tokens=num_chance_tokens,
            num_opponent_registers=2,
            num_history_registers=2,
            )

        '''
        self.encoder_network = EncoderNetwork(            
            observation_shape=observation_shape,
            num_players=num_players,
            num_board_cnn_resblocks=8,
            unit_embed_channels=num_channels,
            vec_blocks=8,
            vec_channels=16,
            state_channels=num_channels,
            reduced_channels_chance=16,
            fc_chance_layers=fc_chance_layers, 
            num_chance_tokens=num_chance_tokens,
            )
        '''

        self.afterstate_dynamics_network = AfterstateDynamicsNetwork(
            length=self.hidden_length,
            channels=num_channels,
            reward_channels=16,
            action_space_layers=2,
            fc_reward_layers=fc_reward_layers,
            full_support_size=reward_support_size,
            lstm_hidden_size=lstm_hidden_size,
            )
        
        self.state_dynamics_network = StateDynamicsNetwork(
            length=self.hidden_length,
            channels=num_channels,
            reward_channels=16,
            outcome_space_layers=num_chance_tokens,
            fc_reward_layers=fc_reward_layers,
            full_support_size=reward_support_size,
            lstm_hidden_size=lstm_hidden_size,
            )

        self.afterstate_prediction_network = AfterstatePredictionNetwork(
            length=self.hidden_length,
            num_chance_tokens=num_chance_tokens,
            num_channels=num_channels,
            reduced_channels_value=16,
            reduced_channels_chance=16,
            fc_value_layers=fc_value_layers,
            fc_chance_layers=fc_chance_layers,
            full_support_size=value_support_size,
            )
        
        self.state_prediction_network = StatePredictionNetwork(
            length=self.hidden_length,
            position_length=observation_shape[-1] * observation_shape[-2],
            action_space_size=action_space_size,
            num_channels=num_channels,
            reduced_channels_value=16,
            reduced_channels_policy=31,
            fc_value_layers=fc_value_layers,
            fc_policy_layers=None,
            full_support_size=value_support_size,
            )

        # projection
        if downsample:
            in_dim = num_channels * self.hidden_length
        else:
            in_dim = num_channels * self.hidden_length
        self.porjection_in_dim = in_dim
        self.projection = nn.Sequential(
            nn.Linear(self.porjection_in_dim, self.proj_hid, bias=False),
            nn.LayerNorm(self.proj_hid),
            nn.GELU(),
            nn.Linear(self.proj_hid, self.proj_hid, bias=False),
            nn.LayerNorm(self.proj_hid),
            nn.GELU(),
            nn.Linear(self.proj_hid, self.proj_out, bias=False),
            nn.LayerNorm(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid, bias=False),
            nn.LayerNorm(self.pred_hid),
            nn.GELU(),
            nn.Linear(self.pred_hid, self.pred_out, bias=False),
        )

    def afterstate_prediction(self, encoded_state):     
        chance_token_logit, value = self.afterstate_prediction_network(encoded_state)
        return chance_token_logit, value
    
    def state_prediction(self, encoded_state):     
        policy, value = self.state_prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state, token_onehot, token_softmax  = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state, token_onehot, token_softmax
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized, token_onehot, token_softmax

    # state + action -> afterstate
    def afterstate_dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        
        
        action_one_hot = (
            torch.zeros(
                (
                    encoded_state.shape[0],
                    encoded_state.shape[1],
                    2,
                )
            )
            .to(action.device)
            .float()
        )
        
        
        for i in range(encoded_state.shape[0]):
            a = action[i].data[0]
            x, y, z = a // 31 // 4, a // 31 % 4, a % 31
            flat_x = x*4 + y 
            action_one_hot[i, x*4 + y, 0] = 1
            if x < 7:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                    action_one_hot[i, dest_x*4 + dest_y, 1] = 1
                elif z == 28:
                    action_one_hot[i, 28:37, 1] = 1
                elif z == 29:
                    action_one_hot[i, -4:, 1] = 1
            elif flat_x <= 36:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                    action_one_hot[i, dest_x*4 + dest_y, 1] = 1
                elif z == 29:
                    action_one_hot[i, -4:, 1] = 1
            elif flat_x <= 41:
                action_one_hot[i, 28:37, 1] = 1
            elif x < 13:
                if z < 28:
                    dest_x, dest_y = z // 4, z % 4
                    action_one_hot[i, dest_x*4 + dest_y, 1] = 1   
            else:
                action_one_hot[i, x*4 + y, 1] = 1
                         
                    
        '''
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        '''
        #onehot = torch.nn.functional.one_hot(torch.squeeze(action), self.action_space_size)
        x = torch.cat((encoded_state, action_one_hot), dim=-1)
        next_encoded_state, reward_hidden, value_prefix = self.afterstate_dynamics_network(x, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    # afterstate + chance token -> state
    def state_dynamics(self, encoded_state, reward_hidden, chance_token_onehot):
        # Stack encoded_state with a game specific one hot encoded token
        
        
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    encoded_state.shape[1],
                    self.num_chance_tokens,
                )
            )
            .to(chance_token_onehot.device)
            .float()
        )
        
        chance_token_onehot = torch.unsqueeze(chance_token_onehot, -2)
        action_one_hot = action_one_hot * chance_token_onehot

        '''
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        '''
        #onehot = torch.nn.functional.one_hot(torch.squeeze(action), self.action_space_size)
        x = torch.cat((encoded_state, action_one_hot), dim=-1)
        next_encoded_state, reward_hidden, value_prefix = self.state_dynamics_network(x, reward_hidden)

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

