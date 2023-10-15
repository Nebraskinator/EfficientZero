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
        self.lin = nn.Linear(in_channels, 
                             out_channels,
                             bias=False)

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
                                          batch_first=True,
                                          bias=False)

    def forward(self, x):
        
        x1 = x

        x1 = self.ln(x1)
        x1 = self.act(x1)
        x1, _ = self.attn(x1, x1, x1, need_weights=False)
        
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
                                          batch_first=True,
                                          bias=False)

    def forward(self, q, k, v):
        
        q1 = q

        q1 = self.ln(q1)
        q1 = self.act(q1)
        k = self.ln(k)
        k = self.act(k)
        v = self.ln(v)
        v = self.act(v)
        q1, _ = self.attn(q1, k, v, need_weights=False)
        
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
                       nn.Linear(sizes[i], sizes[i + 1], bias=False)]
        else:
            layers += [nn.LayerNorm(sizes[i]),
                       act(),
                       nn.Linear(sizes[i], sizes[i + 1], bias=False),
                       output_activation()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        #layers[-2].bias.data.fill_(0)

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
        self.linear = nn.Linear(27, 
                                output_size - unit_embedding_size - item_encoded_size - origin_embedding_size, 
                                bias=False)
               
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

class VectorEncoding(nn.Module):
    def __init__(self, 
                 origin_embedding_layer, 
                 origin_embedding_size,
                 output_size):
        super().__init__()
        self.origin_embedding_layer = origin_embedding_layer
        self.linear = nn.Linear(54*4 - 20, output_size - origin_embedding_size, bias=False)
               
    def forward(self, x):
        layers = []
        origin_layers = []
        for i in range(10):
            origin_layers.append(torch.mul(self.origin_embedding_layer(x[:,i*2].long()), x[:, i*2+1].unsqueeze(-1)).unsqueeze(-2))
        origin_layers = torch.concat(origin_layers, dim=-2)
        origin_layers = torch.sum(origin_layers, dim=-2)
        layers.append(origin_layers)
        layers.append(self.linear(torch.div(x[:,20:], 255.)))
        layers = torch.concat(layers, dim=-1)
        return layers
    
class AttentionResNet(nn.Module):
    def __init__(self, 
                 embedding_size, 
                 num_pre_attn_resblocks,
                 num_attn_resblocks,
                 num_attn_heads,
                 num_post_attn_resblocks,
                 length
                 ):
        super().__init__()
        pre_attn_resblocks_q = []
        for i in range(num_pre_attn_resblocks):
            pre_attn_resblocks_q.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.pre_attn_resblocks_q = nn.ModuleList(pre_attn_resblocks_q)
        pre_attn_resblocks_kv = []
        for i in range(num_pre_attn_resblocks):
            pre_attn_resblocks_kv.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.pre_attn_resblocks_kv = nn.ModuleList(pre_attn_resblocks_kv)
        attn_resblocks = []
        for i in range(num_attn_resblocks):
            attn_resblocks.append(CrossAttentionResBlock(embedding_size, num_attn_heads, (length, embedding_size)))
        self.attn_resblocks = nn.ModuleList(attn_resblocks)
        post_attn_resblocks = []
        for i in range(num_post_attn_resblocks):
            post_attn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.post_attn_resblocks = nn.ModuleList(post_attn_resblocks)
               
    def forward(self, x, kv=None):

        for block in self.pre_attn_resblocks_q:
            x = block(x)
        if kv == None:
            kv = x
        else:
            for block in self.pre_attn_resblocks_kv:
                kv = block(kv)
        for block in self.attn_resblocks:
            x = block(x, kv, kv)
        for block in self.post_attn_resblocks:
            x = block(x)
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

        resblocks.append(nn.Sequential(
            nn.LayerNorm(self.flat_size),
            nn.LeakyReLU(),
            nn.Linear(self.flat_size, 
                      register_size,
                      bias=False),
            ))
        self.resblocks = nn.ModuleList(resblocks)
               
    def forward(self, x):

        x = x.view(-1, self.flat_size)        

        for block in self.resblocks:
            x = block(x)
        
        x = x.view(-1, self.num_registers, self.embedding_size)

        return x        
    
class BoardComparisonResNet(nn.Module):
    def __init__(self, 
                 embedding_size, 
                 num_pre_cnn_resblocks,
                 num_cnn_resblocks,
                 num_pre_attn_resblocks,
                 num_attn_resblocks,
                 num_attn_heads,
                 num_post_attn_resblocks,
                 length
                 ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.length = length

        """
        pre_cnn_resblocks = []
        for i in range(num_pre_cnn_resblocks):
            pre_cnn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.pre_cnn_resblocks = nn.ModuleList(pre_cnn_resblocks)
        
        matchup_cnn_blocks = []
        for i in range(num_cnn_resblocks):
            matchup_cnn_blocks.append(ResidualBlock(embedding_size, 
                                                    embedding_size, 
                                                    (embedding_size,
                                                     7,
                                                     8),
                                                    k=3))
        self.matchup_cnn_blocks = nn.ModuleList(matchup_cnn_blocks)
        """
        pre_attn_resblocks = []
        for i in range(num_pre_attn_resblocks):
            pre_attn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.pre_attn_resblocks = nn.ModuleList(pre_attn_resblocks)
        attn_resblocks = []
        for i in range(num_attn_resblocks):
            attn_resblocks.append(CrossAttentionResBlock(embedding_size, num_attn_heads, (length, embedding_size)))
        self.attn_resblocks = nn.ModuleList(attn_resblocks)
        post_attn_resblocks = []
        for i in range(num_post_attn_resblocks):
            post_attn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.post_attn_resblocks = nn.ModuleList(post_attn_resblocks)
               
    def forward(self, x, cross_list):
        crosses = []
        #x_board = x[:, :28, :]
        #x_board = torch.reshape(x_board, (-1, 7, 4, self.embedding_size))
        for c in cross_list:
            #for block in self.pre_cnn_resblocks:
            #    c = block(c)            
            #c_board = c[:, :28, :]
            #c_board = torch.flip(c_board, dims=(1,))
            #c_board = torch.reshape(c_board, (-1, 7, 4, self.embedding_size))
            #matchup = torch.concat([x_board, c_board], dim=-2)
            #matchup = torch.moveaxis(matchup, -1, -3)
            #for block in self.matchup_cnn_blocks:
            #    matchup = block(matchup)
            #matchup = torch.moveaxis(matchup, -3, -1)
            #x_match = torch.reshape(matchup[:, :, :4, :], (-1, 28, self.embedding_size))
            #c_match = torch.reshape(matchup[:, :, 4:, :], (-1, 28, self.embedding_size))
            #c = nn.functional.pad(torch.flip(c_match, dims=(1,)), (0, 0, 0, self.length - 28)) + c
            #x_match = nn.functional.pad(x_match, (0, 0, 0, self.length - 28)) + x
            
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

class HistoryCrossAttention(nn.Module):
    def __init__(self, 
                 embedding_size, 
                 num_pre_attn_resblocks,
                 num_attn_resblocks,
                 num_attn_heads,
                 num_post_attn_resblocks,
                 length
                 ):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.length = length
       
        pre_attn_resblocks = []
        for i in range(num_pre_attn_resblocks):
            pre_attn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.pre_attn_resblocks = nn.ModuleList(pre_attn_resblocks)
        attn_resblocks = []
        for i in range(num_attn_resblocks):
            attn_resblocks.append(CrossAttentionResBlock(embedding_size, num_attn_heads, (length, embedding_size)))
        self.attn_resblocks = nn.ModuleList(attn_resblocks)
        post_attn_resblocks = []
        for i in range(num_post_attn_resblocks):
            post_attn_resblocks.append(LinearResidualBlock(embedding_size, embedding_size, (length, embedding_size)))
        self.post_attn_resblocks = nn.ModuleList(post_attn_resblocks)
               
    def forward(self, x, cross_list):

        for c in cross_list:           
            for block in self.pre_attn_resblocks:
                c = block(c)
            for block in self.attn_resblocks:
                x = block(x, c, c)
            for block in self.post_attn_resblocks:
                x = block(c)
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
        
        # this embedding layer is used by both the unit encoding and vector encoding modules
        self.origin_embedding = nn.Embedding(27, 8, padding_idx=0)
        
        # encodes all units and items
        self.unit_encoding = UnitEncoding(item_embedding_size=8, 
                                          item_encoded_size=16, 
                                          unit_embedding_size=16, 
                                          origin_embedding_layer=self.origin_embedding, 
                                          origin_embedding_size=8,
                                          output_size=unit_embed_channels)
        
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
        
        # self attention resnet applied to the board, benches, and shop
        self.units_sa = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=self.state_length - self.observation_shape[-1])       
        
        # encodes the non-units state vector
        self.vec_encoding = VectorEncoding(origin_embedding_layer=self.origin_embedding, 
                                           origin_embedding_size=8,
                                           output_size=128)
        
        # embeds the units as a linear vector
        self.lin_units_embedding = nn.Sequential(
                        nn.LayerNorm((self.state_length - self.observation_shape[-1]) * self.unit_embed_channels),
                        nn.LeakyReLU(),
                        nn.Linear((self.state_length - self.observation_shape[-1]) * self.unit_embed_channels, 
                                  128,
                                  bias=False))
        
        # calculate the size of the embedded state vector
        self.vec_embed_size = self.state_length * vec_channels       
        
        # calculate the state vector after concatenation with the board/bench/shop embeddings
        self.vec_concat_size = 128 + unit_embed_channels * 4 + 128
        
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
                    nn.LeakyReLU(),
                    nn.Linear(self.vec_concat_size, 
                              self.vec_embed_size,
                              bias=False),
                    ))
        self.vec_resblocks = nn.ModuleList(vec_block_list)
        
        # embed the vector to fill out the units tensor
        self.vec_fill = nn.Sequential(
                        nn.LayerNorm(self.vec_embed_size),
                        nn.LeakyReLU(),
                        nn.Linear(self.vec_embed_size, 
                                  self.observation_shape[-1] * self.unit_embed_channels,
                                  bias=False))        
        
        '''
        # linear layer to reduce the embedding size after merging with
        # the reshaped state vector
        self.post_merge_lin = nn.Sequential(nn.LayerNorm((self.state_length,
                                vec_channels + unit_embed_channels)),
                                      nn.LeakyReLU(),
                                      nn.Linear(vec_channels + unit_embed_channels,
                                                unit_embed_channels,
                                                bias=False))
        '''        
        
        # self attention resnet applied to the board, benches, and shop after
        # merging with the state vector
        self.postmerge_sa = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=self.state_length)         
        
        # apply cross attention to compare player boards
        self.opponent_cross = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=self.state_length)
        
        self.opponent_register = ConvertToRegister(embedding_size=unit_embed_channels, 
                                                 num_registers=num_opponent_registers,
                                                 num_resblocks=2,
                                                 length=self.state_length
                                                 )
        
        l = self.state_length + num_opponent_registers * (num_players - 1)
        
        # self attention resnet applied after cross attention
        self.postcross_sa = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=l)
        
        self.frame_cross = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=l)
        
        self.frame_register = ConvertToRegister(embedding_size=unit_embed_channels, 
                                                 num_registers=num_history_registers,
                                                 num_resblocks=2,
                                                 length=l
                                                 )
        
        l += num_history_registers
        self.final_sa = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=l)        
                                                             
        self.num_boards = observation_shape[0]
        
        self.num_players = num_players
        
        self.num_stacked_frames = self.num_boards // self.num_players
        
        self.chance_conv = nn.Sequential(nn.LayerNorm((l,
                                                    unit_embed_channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(unit_embed_channels, 
                                                reduced_channels_chance, 
                                                bias=False
                                                ))
        
        
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
        
        # apply SA to the units
        units = self.units_sa(units)
        
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
            
        # append the units tensor with a reshaped vector embedding
        # new shape is [batch_size * stacked_frames * num_players, 56, channels]
        vec_fill = self.vec_fill(vec)
        vec_fill = torch.reshape(vec_fill, (-1, self.observation_shape[-1], self.unit_embed_channels))        
        player_state = torch.concat([units, vec_fill], dim=-2)
                              
        # apply SA to the player state
        player_state = self.postmerge_sa(player_state)                
                
        # break out board axis
        # new shape [batch_size * stacked_frames, num_players, 56, channels]
        frame_state = player_state.view(-1, self.num_players, self.state_length, self.unit_embed_channels)
                
        # perform cross attention on each opponent state and convert the cross
        # into learned registers
        player_state = frame_state[:, 0, :, :]        
        opponent_registers = []        
        for i in range(1, self.num_players):
            opponent_state = frame_state[:, i, :, :]
            cross = self.opponent_cross(player_state, opponent_state)
            opponent_registers.append(self.opponent_register(cross))
        
        player_state = torch.concat([player_state] + opponent_registers, dim=-2)
        
        # apply SA to the player state
        player_state = self.postcross_sa(player_state)
                
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
        cur = state[:, 0, :, :]
        
        for i in range(1, self.num_stacked_frames):
            nex = state[:, i, :, :]
            cross = self.frame_cross(cur, nex)
            cur = nex
        frame_registers = self.frame_register(cur)
        
        state = torch.concat([current_frame_state, frame_registers], dim=-2)
        
        state = self.final_sa(state)
        
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

'''
# Encode the observations into hidden states and chance outcome tokens
class EncoderNetwork(nn.Module):
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
        
        # this embedding layer is used by both the unit encoding and vector encoding modules
        self.origin_embedding = nn.Embedding(27, 8, padding_idx=0)
        
        # encodes all units and items
        self.unit_encoding = UnitEncoding(item_embedding_size=8, 
                                          item_encoded_size=16, 
                                          unit_embedding_size=16, 
                                          origin_embedding_layer=self.origin_embedding, 
                                          origin_embedding_size=8,
                                          output_size=unit_embed_channels)               
        
        # encodes the non-units state vector
        self.vec_encoding = VectorEncoding(origin_embedding_layer=self.origin_embedding, 
                                           origin_embedding_size=8,
                                           output_size=unit_embed_channels * observation_shape[-1])
                        
        # self attention resnet applied to the board, benches, and shop after
        # merging with the state vector
        self.postmerge_sa = AttentionResNet(embedding_size=unit_embed_channels, 
                                            num_pre_attn_resblocks=2, 
                                            num_attn_resblocks=3, 
                                            num_attn_heads=2, 
                                            num_post_attn_resblocks=2, 
                                            length=self.state_length)         
        
        # apply cross attention to compare player boards
        self.cross_attention_network = BoardComparisonResNet(embedding_size=unit_embed_channels, 
                                                            num_pre_cnn_resblocks=2,
                                                            num_cnn_resblocks=8,
                                                            num_pre_attn_resblocks=1,
                                                            num_attn_resblocks=2,
                                                            num_attn_heads=2,
                                                            num_post_attn_resblocks=1,
                                                            length=self.state_length)
               
        # apply cross attention to compare player boards
        self.post_cross_attention_network = HistoryCrossAttention(embedding_size=unit_embed_channels, 
                                                            num_pre_attn_resblocks=1,
                                                            num_attn_resblocks=2,
                                                            num_attn_heads=2,
                                                            num_post_attn_resblocks=2,
                                                            length=self.state_length)
                                                     
        self.num_boards = observation_shape[0]
        
        self.num_players = num_players
        
        self.chance_conv = nn.Sequential(nn.LayerNorm((self.state_length,
                                                    unit_embed_channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(unit_embed_channels, 
                                                reduced_channels_chance, 
                                                bias=False
                                                ))
        
        self.block_output_size_chance = self.state_length * reduced_channels_chance

        self.fc_chance = mlp(self.block_output_size_chance, 
                            fc_chance_layers, 
                            num_chance_tokens, 
                            )

    def forward(self, x):
        
        states = []
        for i in range(self.num_boards // self.num_players):
            
            opponents = []
            for i in range(self.num_players):
                
                # get the player state from the observation stack
                obs = x[:, i, :, :, :]
                
                # separate out the units in the board, benches and shops
                units = obs[:, :-1, :, :]
                
                # reshape for the unit encoding step
                units = units.view(-1, self.state_length - self.observation_shape[-1], self.observation_shape[-3])            
                
                # encode each unit
                units = self.unit_encoding(units)
                               
                # vector component of the player state
                vec = obs[:, -1, :, :]
                
                # flatten the player state vector
                vec = torch.reshape(vec, (-1, self.observation_shape[-3] * self.observation_shape[-1]))
                
                # encode the player state vector
                vec = self.vec_encoding(vec)                
                vec = torch.reshape(vec, (-1, self.observation_shape[-1], self.unit_embed_channels))
                
                units_concat = torch.concat([units, vec], dim=-2)

                player_state = self.postmerge_sa(units_concat)
                
                                # append board to player or opponent list
                if i % self.num_players:
                    opponents.append(player_state)
                else:
                    player = player_state
            
            # apply cross-attention to boards
            cross = self.cross_attention_network(player, opponents)
            
            # add the cross attention output to the player state
            state = player + cross
            
            states.append(state)

        state = states[-1]
        if len(states) > 1:
            state = self.post_cross_attention_network(state, states)
                           
        chance = self.chance_conv(state)
        chance = chance.view(-1, self.block_output_size_chance)
        chance = self.fc_chance(chance)
        
        c_e_t = torch.nn.Softmax(-1)(chance)
        # print(("chance minmax:", chance.min(), chance.max(), "softmax minmax:", c_e_t.min(), c_e_t.max()))
        c_t = Onehot_argmax.apply(c_e_t)
        
        return c_t, c_e_t

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean
'''

def check_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} contains nan")
    if torch.isinf(tensor).any():
        print(f"{name} contains inf")

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
        
        self.action_sa = AttentionResNet(embedding_size=channels+outcome_space_layers, 
                                             num_pre_attn_resblocks=2, 
                                             num_attn_resblocks=3, 
                                             num_attn_heads=2, 
                                             num_post_attn_resblocks=2, 
                                             length=length)
    
        self.act_conv = nn.Sequential(nn.LayerNorm((length,
                                                    channels + outcome_space_layers,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(channels+outcome_space_layers, 
                                                channels, 
                                                bias=False
                                                ))
        
        self.dynamics_sa = AttentionResNet(embedding_size=channels, 
                                             num_pre_attn_resblocks=2, 
                                             num_attn_resblocks=3, 
                                             num_attn_heads=2, 
                                             num_post_attn_resblocks=2, 
                                             length=length)
        
      
        self.reward_conv = nn.Sequential(nn.LayerNorm((length,
                                                    channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(channels, 
                                                reward_channels, 
                                                bias=False
                                                ))
        
        self.block_output_size_reward = length * reward_channels
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, 
                            hidden_size=lstm_hidden_size)
        
        self.fc = mlp(lstm_hidden_size, 
                        fc_reward_layers, 
                        full_support_size, 
                        init_zero=init_zero, 
                        )

    def forward(self, x, reward_hidden):
        
        state = x[:, :, :-self.outcome_space_layers]
        
        x = self.action_sa(x)
        x = self.act_conv(x)
    
        x = state + x
    
        state = self.dynamics_sa(x)
    
        x = self.reward_conv(state)
    
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
        
        self.action_sa = AttentionResNet(embedding_size=channels+action_space_layers, 
                                             num_pre_attn_resblocks=2, 
                                             num_attn_resblocks=3, 
                                             num_attn_heads=2, 
                                             num_post_attn_resblocks=2, 
                                             length=length)
    
        self.act_conv = nn.Sequential(nn.LayerNorm((length,
                                                    channels+action_space_layers,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(channels+action_space_layers, 
                                                channels, 
                                                bias=False
                                                ))
        
        self.dynamics_sa = AttentionResNet(embedding_size=channels, 
                                             num_pre_attn_resblocks=2, 
                                             num_attn_resblocks=3, 
                                             num_attn_heads=2, 
                                             num_post_attn_resblocks=2, 
                                             length=length)
        
      
        self.reward_conv = nn.Sequential(nn.LayerNorm((length,
                                                    channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(channels, 
                                                reward_channels, 
                                                bias=False
                                                ))
        
        self.block_output_size_reward = length * reward_channels
        self.lstm = nn.LSTM(input_size=self.block_output_size_reward, hidden_size=lstm_hidden_size)
        
        self.fc = mlp(lstm_hidden_size, 
                        fc_reward_layers, 
                        full_support_size, 
                        init_zero=init_zero, 
                        )

    def forward(self, x, reward_hidden):
        
        state = x[:, :, :-self.action_space_layers]
        
        x = self.action_sa(x)
        x = self.act_conv(x)
    
        x = state + x
    
        state = self.dynamics_sa(x)
    
        x = self.reward_conv(state)
    
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
        
        self.sa = AttentionResNet(embedding_size=num_channels, 
                                             num_pre_attn_resblocks=2, 
                                             num_attn_resblocks=3, 
                                             num_attn_heads=2, 
                                             num_post_attn_resblocks=2, 
                                             length=length)

        self.value_conv = nn.Sequential(nn.LayerNorm((length,
                                                    num_channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(num_channels, 
                                                reduced_channels_value, 
                                                bias=False
                                                ))
        
        self.policy_conv = nn.Sequential(nn.LayerNorm((length,
                                                    num_channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(num_channels, 
                                                reduced_channels_chance, 
                                                bias=False
                                                ))
        
        self.block_output_size_value = length * reduced_channels_value
        self.block_output_size_policy = length * reduced_channels_chance
        self.fc_value = mlp(self.block_output_size_value, 
                            fc_value_layers, 
                            full_support_size, 
                            init_zero=init_zero, 
                            )
        self.fc_policy = mlp(self.block_output_size_policy, 
                             fc_chance_layers, 
                             num_chance_tokens, 
                             init_zero=init_zero, 
                            )

    def forward(self, x):
        x = self.sa(x)
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
        
        self.sa = AttentionResNet(embedding_size=num_channels, 
                                             num_pre_attn_resblocks=2, 
                                             num_attn_resblocks=3, 
                                             num_attn_heads=2, 
                                             num_post_attn_resblocks=2, 
                                             length=length)

        self.value_conv = nn.Sequential(nn.LayerNorm((length,
                                                    num_channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(num_channels, 
                                                reduced_channels_value, 
                                                bias=False
                                                ))
        
        self.policy_conv = nn.Sequential(nn.LayerNorm((position_length,
                                                    num_channels,
                                                    )),
                                      nn.LeakyReLU(),
                                      nn.Linear(num_channels, 
                                                reduced_channels_policy, 
                                                bias=False
                                                ))
        
        
        self.block_output_size_value = length * reduced_channels_value
        self.block_output_size_policy = position_length * reduced_channels_policy
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
        x = self.sa(x)
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
            vec_blocks=8,
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
            reduced_channels_policy=38,
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
            nn.LeakyReLU(),
            nn.Linear(self.proj_hid, self.proj_hid, bias=False),
            nn.LayerNorm(self.proj_hid),
            nn.LeakyReLU(),
            nn.Linear(self.proj_hid, self.proj_out, bias=False),
            nn.LayerNorm(self.proj_out)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid, bias=False),
            nn.LayerNorm(self.pred_hid),
            nn.LeakyReLU(),
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
            x, y, z = a // 38 // 4, a // 38 % 4, a % 38
            action_one_hot[i, x*4 + y, 0] = 1.
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
            action_one_hot[i, dest_x*4 + dest_y, 1] = 1.
        
            
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

