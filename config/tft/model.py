import math
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet, renormalize


class LinearResidualBlock(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 ln_shape,
                 activation=nn.LeakyReLU):
        super().__init__()
        if in_channels != out_channels:
            self.shortcut=True
            self.pool = nn.AdaptiveAvgPool1d(out_channels)
        else:
            self.shortcut=False
        self.ln = nn.LayerNorm(ln_shape)
        self.act = activation()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        
        x1 = x
        if self.shortcut:
            x = self.pool(x)
            
        x1 = self.ln(x1)
        x1 = self.act(x1)
        x1 = self.lin(x1)
  
        x = x + x1
        
        return x
    
class AttentionResBlock(nn.Module):
    def __init__(self, input_channels, 
                 num_heads, 
                 ln_shape,
                 activation=nn.LeakyReLU):
        super().__init__()
        
        self.ln = nn.LayerNorm(ln_shape)
        self.act = activation()
        self.attn = nn.MultiheadAttention(input_channels, 
                                          num_heads, 
                                          batch_first=True)

    def forward(self, x):
        
        x1 = x

        x1 = self.ln(x1)
        x1 = self.act(x1)
        x1, _ = self.attn(x1, x1, x1)
        
        x = x + x1
        
        return x
    
class CrossAttentionResBlock(nn.Module):
    def __init__(self, input_channels, 
                 num_heads, 
                 ln_shape,
                 activation=nn.LeakyReLU):
        super().__init__()
        
        self.ln = nn.LayerNorm(ln_shape)
        self.act = activation()
        self.attn = nn.MultiheadAttention(input_channels, 
                                          num_heads, 
                                          batch_first=True)

    def forward(self, q, k, v):
        
        q1 = q

        q1 = self.ln(q1)
        q1 = self.act(q1)
        q1, _ = self.attn(q1, k, v)
        
        q = q + q1
        
        return q

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
            layers += [nn.LayerNorm(sizes[i]),
                       act(),
                       nn.Linear(sizes[i], sizes[i + 1])]
        else:
            layers += [nn.LayerNorm(sizes[i]),
                       act(),
                       nn.Linear(sizes[i], sizes[i + 1]),
                       output_activation()]

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
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 shape,
                 k=3, 
                 stride=1, 
                 activation=nn.LeakyReLU
                 ):
        super().__init__()
        self.correct_channels = False
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.correct_channels = True
            
        self.ln = nn.LayerNorm(shape)
        self.act = activation()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=stride, padding=k//2)

    def forward(self, x):
        identity = x
        if self.correct_channels:
            identity = self.shortcut_conv(x)
            
        x = self.ln(x)
        x = self.act(x)            
        x = self.conv(x)
        x = x + identity

        return x

class ItemEncoding(nn.Module):
    def __init__(self, 
                 item_embedding_layer, 
                 embedding_size,
                 output_size=32,
                 ):
        assert output_size > embedding_size
        super().__init__()
        self.output_size = output_size
        self.item_embedding = item_embedding_layer
        self.linear = nn.Linear(9, output_size - embedding_size, bias=False)
               
    def forward(self, x):
        embed = x[:, :, 0].long()
        embed = [self.item_embedding(embed)]
        embed.append(self.linear(torch.div(x[:, :, 1:], 255.)))
        embed = torch.concat(embed, dim=-1)
        return embed

class UnitEncoding(nn.Module):
    def __init__(self, 
                 item_embedding_size, 
                 item_encoded_size,
                 unit_embedding_size, 
                 origin_embedding_layer, 
                 origin_embedding_size,
                 output_size=64,
                 ):
        super().__init__()        
        self.item_encoder = ItemEncoding(nn.Embedding(60, item_embedding_size, padding_idx=0), 
                                         item_embedding_size,
                                         output_size=item_encoded_size)
        assert output_size > unit_embedding_size + item_encoded_size + origin_embedding_size
        self.unit_embedding_layer = nn.Embedding(72, unit_embedding_size, padding_idx=0)
        self.origin_embedding_layer = origin_embedding_layer
        self.linear = nn.Linear(25, output_size - unit_embedding_size - item_encoded_size - origin_embedding_size, bias=False)
               
    def forward(self, x):
        layers = []
        item_layers = []
        for i in range(3):
            item_layers.append(self.item_encoder(x[:,:,i*10:(i+1)*10].long()).unsqueeze(-2))
        item_layers = torch.concat(item_layers, dim=-2)
        item_layers = torch.sum(item_layers, dim=-2)
        layers.append(item_layers)
        layers.append(self.unit_embedding_layer(x[:,:,30].long()))
        origin_layers = []
        for i in range(31, 38):
            origin_layers.append(self.origin_embedding_layer(x[:,:,i].long()).unsqueeze(-2))
        origin_layers = torch.concat(origin_layers, dim=-2)
        origin_layers = torch.sum(origin_layers, dim=-2)
        layers.append(origin_layers)
        lin_layers = []
        for i in range(38, 41):
            n = 4
            if i % 2:
                n += 2
            lin_layers.append(nn.functional.one_hot(x[:, :, i].long(), num_classes=n))        
        lin_layers.append(torch.div(x[:, :, 41:], 255.))
        lin_layers = torch.concat(lin_layers, dim=-1)
        layers.append(self.linear(lin_layers))
        layers = torch.concat(layers, dim=-1)
        return layers
    
class UnitNetwork(nn.Module):
    def __init__(self, 
                 shape,
                 item_embedding_size,
                 item_encoded_size,
                 unit_embedding_size, 
                 unit_encoded_size,
                 origin_embedding_layer, 
                 origin_embedding_size,
                 output_size,
                 num_pre_resblocks,
                 num_attn_blocks,
                 num_attn_heads,
                 num_post_resblocks,):
        super().__init__()
        
        self.unit_encoding = UnitEncoding(item_embedding_size,
                                          item_encoded_size,
                                          unit_embedding_size,
                                          origin_embedding_layer,
                                          origin_embedding_size,
                                          output_size=unit_encoded_size)
        
        pre_attn_resblocks = []
        for i in range(num_pre_resblocks):
            if i:
                pre_attn_resblocks.append(LinearResidualBlock(output_size, output_size, (shape, output_size)))
            else:
                pre_attn_resblocks.append(LinearResidualBlock(unit_encoded_size, output_size, (shape, unit_encoded_size)))

        self.pre_attn_resblocks = nn.ModuleList(pre_attn_resblocks)
        
        attn_resblocks = []
        for i in range(num_attn_blocks):
            attn_resblocks.append(AttentionResBlock(output_size, num_attn_heads, (shape, output_size)))

        self.attn_resblocks = nn.ModuleList(attn_resblocks)
        
        post_attn_resblocks = []
        for i in range(num_post_resblocks):
            post_attn_resblocks.append(LinearResidualBlock(output_size, output_size, (shape, output_size)))

        self.post_attn_resblocks = nn.ModuleList(post_attn_resblocks)
        
        
    def forward(self, x):
              
        x = self.unit_encoding(x)
        
        for block in self.pre_attn_resblocks:
            x = block(x)
            
        for block in self.attn_resblocks:
            x = block(x)
        
        for block in self.post_attn_resblocks:
            x = block(x)
        
        return x        
        

class VectorEncoding(nn.Module):
    def __init__(self, 
                 origin_embedding_layer, 
                 origin_embedding_size,
                 output_size):
        super().__init__()
        self.origin_embedding_layer = origin_embedding_layer
        self.linear = nn.Linear(11, output_size - origin_embedding_size, bias=False)
               
    def forward(self, x):
        layers = []
        origin_layers = []
        for i in range(10):
            origin_layers.append(torch.mul(self.origin_embedding_layer(x[:,i*2].long()), x[:, i*2+1].unsqueeze(-1)).unsqueeze(-2))
        origin_layers = torch.concat(origin_layers, dim=-2)
        origin_layers = torch.sum(origin_layers, dim=-2)
        layers.append(origin_layers)
        layers.append(self.linear(torch.div(x[:,20:31], 255.)))
        layers = torch.concat(layers, dim=-1)
        return layers

class CrossAttentionNetwork(nn.Module):
    def __init__(self, 
                 embedding_size, 
                 num_pre_attn_resblocks,
                 num_attn_resblocks,
                 num_attn_heads,
                 num_post_attn_resblocks,
                 shape
                 ):
        super().__init__()
        pre_attn_resblocks = []
        for i in range(num_pre_attn_resblocks):
            pre_attn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (shape, embedding_size)))
        self.pre_attn_resblocks = nn.ModuleList(pre_attn_resblocks)
        attn_resblocks = []
        for i in range(num_attn_resblocks):
            attn_resblocks.append(CrossAttentionResBlock(embedding_size, num_attn_heads, (shape, embedding_size)))
        self.attn_resblocks = nn.ModuleList(attn_resblocks)
        post_attn_resblocks = []
        for i in range(num_post_attn_resblocks):
            post_attn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (shape, embedding_size)))
        self.post_attn_resblocks = nn.ModuleList(post_attn_resblocks)
               
    def forward(self, x, cross_list):
        crosses = []
        for c in cross_list:
            for block in self.pre_attn_resblocks:
                c = block(c)
            for block in self.attn_resblocks:
                c = block(x, c, c)
            for block in self.post_attn_resblocks:
                c = block(c)
            crosses.append(c.unsqueeze(-2))
        crosses = torch.concat(crosses, dim=-2)
        crosses = torch.mean(crosses, dim=-2)
        return crosses
            
        

# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
        self,
        observation_shape,
        num_players,
        vis_blocks_board_prevector,
        vis_blocks_board_premerge,
        vis_blocks_board_postmerge,
        vis_blocks_post_cross_attention,
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
        self.champ_embed_channels = champ_embed_channels
        self.origin_embedding = nn.Embedding(27, 8, padding_idx=0)
        
        self.champ_network = UnitNetwork(
                                        shape=(observation_shape[-2] - 1) * observation_shape[-1],
                                        item_embedding_size=8,
                                        item_encoded_size=16,
                                        unit_embedding_size=16, 
                                        unit_encoded_size=champ_embed_channels,
                                        origin_embedding_layer=self.origin_embedding, 
                                        origin_embedding_size=8,
                                        output_size=champ_embed_channels,
                                        num_pre_resblocks=4,
                                        num_attn_blocks=6,
                                        num_attn_heads=2,
                                        num_post_resblocks=4)
        
        self.vec_encoding = VectorEncoding(self.origin_embedding, 
                                           8,
                                           128)
        
        self.vec_channels = vec_channels
        self.observation_shape = observation_shape
        self.vis_embed_channels = vis_embed_channels
        self.vis_hidden_channels = vis_hidden_channels
        
        
        board_prevector_block_list = []
        for i in range(vis_blocks_board_prevector):
            if i:
                board_prevector_block_list.append(ResidualBlock(vis_hidden_channels, 
                                                                vis_hidden_channels, 
                                                                (vis_hidden_channels,
                                                                 self.observation_shape[-2] - 1,
                                                                 self.observation_shape[-1]),
                                                                k=3))
            else:
                board_prevector_block_list.append(ResidualBlock(champ_embed_channels, 
                                                                vis_hidden_channels, 
                                                                (champ_embed_channels,
                                                                 self.observation_shape[-2] - 1,
                                                                 self.observation_shape[-1]),
                                                                k=3))

        self.board_prevector_blocks = nn.ModuleList(board_prevector_block_list)
        
        self.board_vector_embed_conv = nn.Sequential(
            nn.LayerNorm((vis_hidden_channels, self.observation_shape[-2] - 1, self.observation_shape[-1])),
            nn.LeakyReLU(),
            nn.Conv2d(vis_hidden_channels, vis_embed_channels, kernel_size=1, bias=False)
            )
        
        self.vec_embed_size = observation_shape[-1] * observation_shape[-2] * vec_channels
        
        self.board_premerge_padding = nn.ZeroPad2d((0,0,0,1))
        
        # vector encoding size + 4*unit encoding size + vis linear embedding size
        self.vec_concat_size = 128 + \
            champ_embed_channels * 4 + observation_shape[-1] * (observation_shape[-2] - 1) * vis_embed_channels
        
        vec_block_list = []
        for i in range(vec_blocks+1):
            if i:
                vec_block_list.append(LinearResidualBlock(self.vec_embed_size, 
                                                          self.vec_embed_size,
                                                          self.vec_embed_size))
            else:
                vec_block_list.append(nn.Sequential(
                    nn.LayerNorm(self.vec_concat_size),
                    nn.LeakyReLU(),
                    nn.Linear(self.vec_concat_size, self.vec_embed_size),
                    ))
        self.vec_blocks = nn.ModuleList(vec_block_list)
                
        board_premerge_block_list = []
        for i in range(vis_blocks_board_premerge):
            board_premerge_block_list.append(ResidualBlock(vis_hidden_channels, 
                                                           vis_hidden_channels,
                                                           (vis_hidden_channels,
                                                            self.observation_shape[-2],
                                                            self.observation_shape[-1]),
                                                           k=3))
        self.board_premerge_blocks = nn.ModuleList(board_premerge_block_list)
        
        board_postmerge_block_list = []
        for i in range(vis_blocks_board_postmerge):
            if i:
                board_postmerge_block_list.append(ResidualBlock(vis_hidden_channels, 
                                                                vis_hidden_channels, 
                                                                (vis_hidden_channels,
                                                                 self.observation_shape[-2],
                                                                 self.observation_shape[-1]),
                                                                k=3))
            else:
                board_postmerge_block_list.append(ResidualBlock(vis_hidden_channels + vec_channels, 
                                                                vis_hidden_channels, 
                                                                (vis_hidden_channels + vec_channels,
                                                                 self.observation_shape[-2],
                                                                 self.observation_shape[-1]),
                                                                k=3))
        self.board_postmerge_blocks = nn.ModuleList(board_postmerge_block_list)
        
        self.cross_attention_network = CrossAttentionNetwork(embedding_size=vis_hidden_channels, 
                                                            num_pre_attn_resblocks=2,
                                                            num_attn_resblocks=6,
                                                            num_attn_heads=2,
                                                            num_post_attn_resblocks=4,
                                                            shape=observation_shape[-1] * observation_shape[-2])
        
        post_cross_attn_blocks = []
        for i in range(vis_blocks_post_cross_attention):
            post_cross_attn_blocks.append(ResidualBlock(vis_hidden_channels, 
                                                            vis_hidden_channels, 
                                                            (vis_hidden_channels,
                                                             self.observation_shape[-2],
                                                             self.observation_shape[-1]),
                                                            k=3))
        self.post_cross_attn_blocks = nn.ModuleList(post_cross_attn_blocks)
                                                     
        self.num_boards = observation_shape[0]
        
        self.num_players = num_players

    def forward(self, x):
               
        opponents = []
        for i in range(self.num_boards):
            
            # get the board from the observation stack
            obs = x[:, i, :, :, :]
            
            # visual component of the board
            vis = obs[:, :-1, :, :]
            
            # reshape vis for the unit embedding step
            unit_embed = vis.view(-1, self.observation_shape[-1] * (self.observation_shape[-2]-1), self.observation_shape[-3])            
            
            # embed the each unit
            unit_embed = self.champ_network(unit_embed)
            
            # sum unit embeddings for concatenation with the vec
            board_sum = torch.sum(unit_embed[:, :28, :], dim=-2)
            bench_sum = torch.sum(unit_embed[:, 28:28+9, :], dim=-2)
            shop_sum = torch.sum(unit_embed[:, 28+9:28+9+5, :], dim=-2)
            ibench_sum = torch.sum(unit_embed[:, 28+9+5:28+9+5+10, :], dim=-2)
            
            # reshape unit embeddings for visual network
            vis = unit_embed.view(-1, self.observation_shape[-2] - 1, self.observation_shape[-1], self.champ_embed_channels)
            
            # move the channels axis to the front
            vis = torch.moveaxis(vis, -1, -3)
                        
            # vis resnet prior to vec embedding
            for block in self.board_prevector_blocks:
                vis = block(vis)
            
            # create linear embedding of vis                
            vis_vec = self.board_vector_embed_conv(vis)  
            vis_vec = torch.reshape(vis_vec, (-1, self.vis_embed_channels * (self.observation_shape[-2] - 1) * self.observation_shape[-1]))
            
            # vector component of the board
            vec = obs[:, -1, :, :]
            
            # flatten the board vector
            vec = torch.reshape(vec, (-1, self.observation_shape[-3] * self.observation_shape[-1]))
            
            # encode the board vector
            vec = self.vec_encoding(vec)
            
            # concatenate the board vector with the visual embedding and the board region unit embedding sums
            vec = torch.concat([vec, board_sum, bench_sum, shop_sum, ibench_sum, vis_vec], dim=-1)
            
            # linear resnet for the full vector
            for block in self.vec_blocks:
                vec = block(vec)
                
            # reshape the linear vector for the visual network
            vec = vec.view(-1, self.vec_channels, self.observation_shape[-2], self.observation_shape[-1])
                
            # pad the vis to return to input size
            vis = self.board_premerge_padding(vis)
            
            # vis resnet prior to merging with reshaped vector
            for block in self.board_premerge_blocks:
                vis = block(vis)
            
            # concatenate vis with the reshaped vector along channels axis
            vis = torch.concat([vis, vec], dim=1)
            
            # vis resnet after merging with reshaped vector
            for block in self.board_postmerge_blocks:
                vis = block(vis)
                
            # now that positional information has been added from the CNN, we can
            # prepare the board for cross-attention for board strength comparisons
            
            # move the channels axis
            vis = torch.moveaxis(vis, -3, -1)
            
            # reshape the board as a sequence of embeddings
            vis = torch.reshape(vis, (-1, self.observation_shape[-2] * self.observation_shape[-1], self.vis_hidden_channels))            
            
            # append board to player or opponent list
            if i % self.num_players:
                opponents.append(vis)
            else:
                player = vis
        
        # apply cross-attention to boards
        state = self.cross_attention_network(player, opponents)
        
        # reshape for cnn
        state = torch.reshape(state, (-1, self.observation_shape[-2], self.observation_shape[-1], self.vis_hidden_channels))
        state = torch.moveaxis(state, -1, -3)
        
        for block in self.post_cross_attn_blocks:
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
    
        self.act_conv = nn.Sequential(nn.LayerNorm((dynamics_channels + action_space_layers,
                                observation_shape[-2],
                                observation_shape[-1])),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(dynamics_channels + action_space_layers, 
                                                dynamics_channels, 
                                                kernel_size=3, 
                                                stride=1, 
                                                padding=1, 
                                                bias=False
                                                ))
        
        self.resblocks = nn.ModuleList(
            [ResidualBlock(dynamics_channels, 
                           dynamics_channels,
                           (dynamics_channels,
                            observation_shape[-2],
                            observation_shape[-1])
                           ) for _ in range(num_dynamics_blocks)]
        )
      
        self.reward_conv = nn.Sequential(nn.LayerNorm((dynamics_channels,
                                observation_shape[-2],
                                observation_shape[-1])),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(dynamics_channels, 
                                                reward_channels, 
                                                kernel_size=1, 
                                                stride=1, 
                                                bias=False
                                                ))
        
        self.block_output_size_reward = observation_shape[-1] * observation_shape[-2] * reward_channels
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        
        self.fc = mlp(self.lstm_hidden_size, 
                        fc_reward_layers, 
                        full_support_size, 
                        init_zero=init_zero, 
                        )

    def forward(self, x, reward_hidden):
        state = x[:,:-self.action_space_layers,:,:]
        
        x = self.act_conv(x)
    
        x += state
    
        for block in self.resblocks:
            x = block(x)
        state = x
    
        x = self.reward_conv(x)
    
        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
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
                                [ResidualBlock(num_channels, 
                                               num_channels,
                                               (num_channels, 
                                                observation_shape[-2],
                                                observation_shape[-1],
                                                )) for _ in range(num_blocks)]
                                )

        self.value_conv = nn.Sequential(nn.LayerNorm((num_channels,
                                observation_shape[-2],
                                observation_shape[-1])),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(num_channels, 
                                                reduced_channels_value, 
                                                kernel_size=1, 
                                                stride=1, 
                                                bias=False
                                                ))
        self.policy_conv = nn.Sequential(nn.LayerNorm((num_channels,
                                observation_shape[-2],
                                observation_shape[-1])),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(num_channels, 
                                                reduced_channels_policy, 
                                                kernel_size=1, 
                                                stride=1, 
                                                bias=False
                                                ))
        
        self.block_output_size_value = observation_shape[-1]*observation_shape[-2]*reduced_channels_value
        self.block_output_size_policy = observation_shape[-1]*observation_shape[-2]*reduced_channels_policy
        self.fc_value = mlp(self.block_output_size_value, 
                            fc_value_layers, 
                            full_support_size, 
                            init_zero=init_zero, 
                            )
        #self.fc_policy = mlp(self.block_output_size_policy, 
        #                     fc_policy_layers, 
        #                     action_space_size, 
        #                     init_zero=init_zero, 
        #                     momentum=momentum
        #                    )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.value_conv(x)
        policy = self.policy_conv(x)
        value = value.view(-1, self.block_output_size_value)
        policy = torch.moveaxis(policy, -3, -1)
        policy = torch.reshape(policy, (-1, self.block_output_size_policy))
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
            vis_blocks_board_prevector=8,
            vis_blocks_board_premerge=4,
            vis_blocks_board_postmerge=4,
            vis_blocks_post_cross_attention=8,
            champ_embed_channels=num_channels,
            vis_hidden_channels=num_channels,
            vis_embed_channels=16,
            vec_blocks=8,
            vec_channels=32,
            state_channels=num_channels,
            )

        self.dynamics_network = DynamicsNetwork(
            observation_shape,
            num_dynamics_blocks=12,
            dynamics_channels=num_channels,
            num_reward_blocks=2,
            reward_channels=16,
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
            2,
            num_channels,
            16,
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

