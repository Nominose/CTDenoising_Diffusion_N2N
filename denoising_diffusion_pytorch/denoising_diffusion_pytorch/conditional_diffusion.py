import math
import copy
import os
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from skimage.measure import block_reduce

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from CTDenoising_Diffusion_N2N.denoising_diffusion_pytorch.denoising_diffusion_pytorch.attend import Attend
import CTDenoising_Diffusion_N2N.denoising_diffusion_pytorch.denoising_diffusion_pytorch.kernel as kernel

from CTDenoising_Diffusion_N2N.denoising_diffusion_pytorch.denoising_diffusion_pytorch.version import __version__

import CTDenoising_Diffusion_N2N.functions_collection as ff
import CTDenoising_Diffusion_N2N.Data_processing as Data_processing

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 2D functions
def Upsample2D(dim, dim_out = None): 
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample2D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm2D(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
    
class Block(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)  # groups: The number of groups to split the channels into. This determines how the normalization is applied across the channels. For example, if groups=2, the channels will be divided into two groups, and normalization will be applied separately within each group.
        self.act = nn.SiLU()  # sigmoid linear unit

    def forward(self, x, scale_shift = None):
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock2D(nn.Module): # input dimension is dim , output dimension is dim_out. for time_emb, the input dimension is time_emb_dim, output dimension is dim_out * 2
    '''experience two basic convolution+group_normlization+SiLu blocks, and then add the input to the output of the second block.'''
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2) # fully-connected layer
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)  # scale shift is how to incorporate the time embedding into x

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention2D(nn.Module): # input dimension is dim, same dimension for input and output
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm2D(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm2D(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)  # split input into q k v evenly
        # here each q, k ,v has the dim = [b, hidden_dim, h, w]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        # here each q, k ,v has the dim = [b, heads, hidden_dim, h*w]

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication
        # k*v:  [b, heads, hidden_dim, h*w] mul [b, heads, hidden_dim, h*w] -> [b, heads, hidden_dim, hidden_dim]

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # context*q: [b, heads, hidden_dim, hidden_dim] mul [b, heads, hidden_dim, h*w] -> [b, heads, hidden_dim, h*w]

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        # out: [b, heads, hidden_dim, h*w] -> [b, heads*hidden_dim, h, w]

        return self.to_out(out)
    

class Attention2D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm2D(dim) 
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # here each q, k ,v has the dim = [b, hidden_dim, h, w]

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)
        # here each q, k ,v has the dim = [b, heads, h*w, hidden_dim]

        out = self.attend(q, k, v)
        # first q*k: [b, heads, h*w, hidden_dim] mul [b, heads, h*w, hidden_dim] -> [b, heads, h*w, h*w]   (einsum(f"b h i d, b h j d -> b h i j", q, k) * scale)
        # second *v: [b, heads, h*w, h*w] mul [b, heads, h*w, hidden_dim] -> [b, heads, h*w, hidden_dim]  (einsum(f"b h i j, b h j d -> b h i d", attn, v))

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


# 3D functions
def Upsample3D(dim, dim_out = None, upsample_factor = (2,2,1)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) d -> b (c p1 p2) h w d', p1 = 2, p2 = 2),
        nn.Conv3d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm3D(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1 , 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# building block modules

class Block3D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)  
        self.act = nn.SiLU()  

    def forward(self, x, scale_shift = None):
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock3D(nn.Module): # input dimension is dim , output dimension is dim_out. for time_emb, the input dimension is time_emb_dim, output dimension is dim_out * 2
    '''experience two basic convolution+group_normlization+SiLu blocks, and then add the input to the output of the second block.'''
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2) # fully-connected layer
        ) if exists(time_emb_dim) else None

        self.block1 = Block3D(dim, dim_out, groups = groups)
        self.block2 = Block3D(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)  # scale shift is how we incorporate the time embedding into x

        h = self.block2(h)

        return h + self.res_conv(x)
    
class LinearAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            RMSNorm3D(dim)
        )

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=self.heads), qkv)  # h = head, c = dim_head

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z = d)
        return self.to_out(out)


class Attention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x) 

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h (x y z) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = d)
        return self.to_out(out)
    
# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module): # output dimension is dim
    '''https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/'''
    def __init__(self, dim, theta = 10000): # theta is the n on the guidance webpage, dim is d (Dimension of the output embedding space).
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)  # weights are not 1/(n**(2i/d)), instead it's learnable

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# model

class Unet(nn.Module):
    def __init__(
        self,
        problem_dimension, # 3D or 2D 
        init_dim,
        out_dim,
        channels,
        conditional_diffusion,
        condition_channels = 1, 

        dim_mults = (1, 2, 4, 8),
        downsample_list = (True, True, True, False),
        upsample_list = (True, True, True, False),
        self_condition = False,   # use the prediction from the previous iteration as the condition of next iteration
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (None, None, None, True),
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition
        self.conditional_diffusion = conditional_diffusion
        self.channels = channels
        input_channels = channels + (condition_channels if self.conditional_diffusion else 0)  # add one channel for the condition

        self.problem_dimension = problem_dimension

        # define some layers 
        conv_layer = nn.Conv2d if self.problem_dimension == '2D' else nn.Conv3d
        ResnetBlock = ResnetBlock2D if self.problem_dimension == '2D' else ResnetBlock3D
        Attention = Attention2D if self.problem_dimension == '2D' else Attention3D
        LinearAttention = LinearAttention2D if self.problem_dimension == '2D' else LinearAttention3D
        downsample_layer = Downsample2D if self.problem_dimension == '2D' else Downsample3D
        upsample_layer = Upsample2D if self.problem_dimension == '2D' else Upsample3D

        self.init_conv = conv_layer(input_channels, init_dim, 5, padding = 2) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16,32,64,128]
        in_out = list(zip(dims[:-1], dims[1:])) 
        # [(16,16), (16, 32), (32, 64), (64, 128)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)  # Here, block_klass is being defined as a new function that is essentially a ResnetBlock, but with the groups argument set to resnet_block_groups. This means that when you call block_klass, you only need to provide the remaining arguments that ResnetBlock expects (such as dim and dim_out), and groups will be automatically set to the value of resnet_block_groups.

        # time embeddings

        time_dim = init_dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(init_dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = init_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),  # Gaussian error activation function
            nn.Linear(time_dim, time_dim))

        # attention
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            no_downsample = downsample_list[ind] == False

            if layer_full_attn == True:
                attn_klass = FullAttention
            elif layer_full_attn == False:
                attn_klass = LinearAttention

            # in each downsample stage, 
            # we have 4 layers: 2 resnet blocks (doesn't increase the feature number), 1 attention layer, and 1 downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if layer_full_attn is not None else nn.Identity(),
                downsample_layer(dim_in, dim_out) if not no_downsample else conv_layer(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            no_upsample = upsample_list[ind] == False

            if layer_full_attn == True:
                attn_klass = FullAttention
            elif layer_full_attn == False:
                attn_klass = LinearAttention
          
            # in each upsample stage,
            # we have 4 layers: 2 resnet blocks (does change the feature number), 1 attention layer, and 1 upsampling layer (upsample x and y by 2, then decrease the feature number by 2)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if layer_full_attn is not None else nn.Identity(),
                upsample_layer(dim_out, dim_in) if not no_upsample else  conv_layer(dim_out, dim_in, 5, padding = 2)  
            ]))


        self.out_dim = out_dim

        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim = time_dim)
        self.final_conv = conv_layer(init_dim, self.out_dim, 1)  # output channel is initial channel number


    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1) # = 8

    def forward(self, x, time, condition = None):
        # concatenate the condition to the input along the dimension = 1
        if self.conditional_diffusion:
            if exists(condition) == 0:
                raise ValueError('condition is required for conditional diffusion')
            x = torch.cat((x, condition), dim = 1)
            # print('after adding condition, x shape is', x.shape)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)

            x = attn(x) + x
            h.append(x)

            x = downsample(x)
          
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)   # h.pop() is the output of the corresponding downsample stage
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)

            x = block2(x, t)

            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        final_image = self.final_conv(x)
      
        return final_image

# gaussian diffusion trainer class 
 
def extract(a, t, x_shape):
    # extract at from a list of a, then add empty axes to match the image dimension
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) 

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def reverse_warmup_beta_schedule(timesteps, beta_start=5e-5, beta_end=1e-2, warmup_frac=0.7):
    """
    DDM² reverse warm-up schedule
    
    - 前 (1-warmup_frac)*timesteps 步：β 保持在 beta_start
    - 后 warmup_frac*timesteps 步：β 从 beta_start 线性增加到 beta_end
    """
    betas = torch.full((timesteps,), beta_start, dtype=torch.float64)
    warmup_time = int(timesteps * warmup_frac)
    betas[timesteps - warmup_time:] = torch.linspace(
        beta_start, beta_end, warmup_time, dtype=torch.float64
    )
    return betas


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size, 
        timesteps = 2000,
        sampling_timesteps = None,
        objective = 'pred_x0',  # previous definition is "pred_v"
        clip_or_not = None,
        clip_range = None,

        force_ddim = False,
        beta_schedule = 'reverse_warmup',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not model.random_or_learned_sinusoidal_cond
        self.problem_dimension = model.problem_dimension

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.conditional_diffusion = self.model.conditional_diffusion

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v', 'forward_recon'}, \
    'objective must be one of pred_noise / pred_x0 / pred_v / forward_recon'


        # clip to stablize
        self.clip_or_not = clip_or_not
        self.clip_range = clip_range
        assert self.clip_or_not is not None, 'clip_or_not must be specified'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        elif beta_schedule == 'reverse_warmup':  # ← 新增
            beta_schedule_fn = reverse_warmup_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs) # pre-defined schedule_fn_kwargs in main function arguments

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)  # This one is alpha(t-1)-bar. Practically, this pads the tensor with one element at the beginning and no elements at the end, using a padding value of 1.

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        print('is ddim sampling', self.is_ddim_sampling)
        self.ddim_sampling_eta = ddim_sampling_eta
        self.force_ddim = force_ddim

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # this is sqrt(1- alpha(t)-bar)  /  sqrt(alpha(t)-bar) in the note

        sqrt_alphas_cumprod_prev = torch.sqrt(torch.cat([torch.tensor([1.0]), alphas_cumprod]))
        register_buffer('sqrt_alphas_cumprod_prev', sqrt_alphas_cumprod_prev)

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # use option 2 in my note

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))  # The method a.clamp(min) applies element-wise clamping to a tensor min, ensuring that all values are greater than or equal to min
        register_buffer('posterior_mean_coef_x0', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # for x0
        register_buffer('posterior_mean_coef_xt', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))  # for xt

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)  # element-wise division
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))
        elif objective == 'forward_recon':
            register_buffer('loss_weight', maybe_clipped_snr / snr)


        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    # @property is a built-in Python decorator that allows you to define a method as if it were a class attribute. This means you can access it like an attribute rather than calling it as a method.
    # for example, if in class "Circle" we have a function "area" as the property, then we can call this function as Circle.area instead of Circle.area()
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef_x0, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t,  condition = None):
        if self.conditional_diffusion:
            if exists(condition) == 0:
                raise ValueError('conditional diffusion is specified, but no condition is provided')
            model_output = self.model(x, t, condition)
        else:
            model_output = self.model(x, t)

        if self.clip_or_not:
            maybe_clip = partial(torch.clamp, min = self.clip_range[0], max = self.clip_range[1]) 
        else:
            maybe_clip = identity
       
        if self.objective == 'pred_noise': 
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

            x_start = maybe_clip(x_start)

            if self.clip_or_not:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)


        return ModelPrediction(pred_noise, x_start)  # if prediction = ModelPrediction(pred_noise = a, x_start = b), then prediction.pred_noise = a, prediction.x_start = b

    def p_mean_variance(self, x, t, condition = None, output_noise = True):
        if self.conditional_diffusion:
            if exists(condition) == 0:
                raise ValueError('conditional diffusion is specified, but no condition is provided')
            preds = self.model_predictions(x, t, condition)
        else:
            preds = self.model_predictions(x, t)
        
        x_start = preds.pred_x_start

        if self.clip_or_not:
            x_start.clamp_(self.clip_range[0], self.clip_range[1])
            # print('in p_mean_variance, x_start max and min: ',torch.max(x_start), torch.min(x_start))

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        if output_noise == False:
            return model_mean, posterior_variance, posterior_log_variance, x_start
        else:
            return model_mean, posterior_variance, posterior_log_variance, x_start, pred_noise

    @torch.inference_mode()
    # In PyTorch, torch.inference_mode() is a context manager that temporarily sets the mode of the autograd engine to inference mode. This means that operations inside the context are treated as if they are being used for inference, rather than for training.
    def p_sample(self, x, t, condition = None, output_noise = True):
        b, *_, device = *x.shape, self.device   # * in front of a list means unpacking the list, b = batch
        batched_times = torch.full((b,), t, device = device, dtype = torch.long) # torch.full() is a function in PyTorch that creates a tensor of a specified size and fills it with a specified value.
        
        if output_noise == False:
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, condition = condition,  output_noise = output_noise)
        else:
            model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(x = x, t = batched_times, condition = condition,  output_noise = output_noise)
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        if output_noise == False:
            return pred_img, x_start
        else:
            return pred_img, x_start, pred_noise
        

    @torch.inference_mode()
    def p_sample_loop(self, shape, condition = None, output_noise = True):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)  # this is random noise
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            model_output = self.p_sample(img, t, condition = condition, output_noise = output_noise)
            if output_noise == True:
                img,x_start, pred_noise = model_output
            else:
                img, x_start = model_output
            imgs.append(img)
       
        final_answer_x0 = img #if not return_all_timesteps else torch.stack(imgs, dim = 1)

        final_answer_x0 = self.unnormalize(final_answer_x0)
        return final_answer_x0


    @torch.inference_mode()
    def ddim_sample(self, shape, condition = None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        print('using DDIM, eta: ',eta)
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        coefficients = []
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            
            pred = self.model_predictions(img, time_cond, condition = condition)

            if time_next < 0:
                img = pred.pred_x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = pred.pred_x_start * alpha_next.sqrt() + \
                  c * pred.pred_noise + \
                  sigma * noise
            imgs.append(img)

        ret = img #if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)

        return ret
    
    @torch.inference_mode()
    def sample(self, condition = None, batch_size = 16):
       
        if self.force_ddim == False:
            sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        else:
            sample_fn = self.ddim_sample
        if self.problem_dimension == '2D':
            return sample_fn((batch_size, self.channels, self.image_size[0], self.image_size[1]), condition = condition)
        elif self.problem_dimension == '3D':
            return sample_fn((batch_size, self.channels, self.image_size[0], self.image_size[1], self.image_size[2]), condition = condition)
        else:
            raise ValueError(f'unknown problem dimension {self.problem_dimension}')

    @torch.inference_mode()
    def compute_matched_state(self, original_X, y_bar):
        """
        计算 per-sample 的 matched_state（和论文索引一致）
        """
        B = original_X.shape[0]
        device = original_X.device
        
        # 计算残差
        residual = original_X - y_bar
        residual = residual - residual.mean(dim=tuple(range(1, residual.ndim)), keepdim=True)
        
        # 计算每个样本的 sigma
        sigmas = residual.flatten(1).std(dim=1, unbiased=False)  # (B,)
        
        # 【关键】使用论文的索引体系：sqrt(1 - alpha_prev^2)
        # sqrt_alphas_cumprod_prev 长度 T+1，索引 0..T
        schedule = (1 - self.sqrt_alphas_cumprod_prev**2).sqrt().to(device)  # (T+1,)
        
        # 匹配
        t_star = (schedule[None, :] - sigmas[:, None]).abs().argmin(dim=1)  # (B,)
        
        # 防止 t_star=0（会导致除以0）
        t_star = t_star.clamp(min=1, max=self.num_timesteps - 1)
        
        return t_star.long()

    @torch.inference_mode()
    def state_matching(self, original_X, y_bar):
        """
        推理时的 State Matching（返回单个 t*，取 batch 最大值）
        """
        matched_state = self.compute_matched_state(original_X, y_bar)
        t_star = int(matched_state[0].item())
        
        print(f"State Matching: t* = {t_star}")
        return t_star

    @torch.inference_mode()
    def ddm2_denoise(self, noisy_image, y_bar=None, start_t=None, use_flip=True):
        """
        DDM² 去噪
        
        Args:
            noisy_image: 原始噪声图像
            y_bar: N2N 输出（用于 State Matching）
            start_t: 起始时间步（如果为 None，自动计算）
            use_flip: 是否使用 flip augmentation
        """
        device = noisy_image.device
        
        # State Matching
        if start_t is None:
            if y_bar is None:
                raise ValueError("需要提供 y_bar 进行 State Matching，或手动指定 start_t")
            start_t = self.state_matching(noisy_image, y_bar)
        
        start_t = min(start_t, self.num_timesteps - 1)
        print(f"DDM² denoising from t={start_t}")
        
        if use_flip:
            return self._denoise_with_flip(noisy_image, start_t)
        else:
            return self._denoise_single(noisy_image, start_t)

    def _denoise_single(self, noisy_image, start_t):
        """单次反向采样（使用现有的 p_sample）"""
        device = noisy_image.device
        batch_size = noisy_image.shape[0]
        img = noisy_image.clone()
        
        for t in tqdm(reversed(range(0, start_t)), desc='DDM² denoising', total=start_t + 1, leave=False):
            # 使用现有的 p_sample
            img, x_start = self.p_sample(img, t, condition=None, output_noise=False)
        
        # [-1, 1] -> [0, 1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        
        return img


    def _denoise_with_flip(self, noisy_image, start_t):
        """带 flip augmentation 的反向采样"""
        device = noisy_image.device
        batch_size = noisy_image.shape[0]
        img = noisy_image.clone()
        
        for t in tqdm(reversed(range(0, start_t)), desc='DDM² denoising', total=start_t + 1, leave=False):
            batched_times = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # flip denoise: 4次翻转平均（在预测 x_start 这一步做）
            x_start = self._flip_predict(img, batched_times)
            
            # clip
            if self.clip_or_not:
                x_start.clamp_(self.clip_range[0], self.clip_range[1])
            
            # 使用现有的 q_posterior
            model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=img, t=batched_times)
            
            # 采样
            noise = torch.randn_like(img) if t > 0 else 0.
            img = model_mean + (0.5 * model_log_variance).exp() * noise
        
        # [-1, 1] -> [0, 1]
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        
        return img

    def _flip_predict(self, x, t):
        """4次翻转平均预测 x_start"""
        results = []
        
        # 原图
        out = self.model(x, t)
        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t, out)
        elif self.objective == 'pred_x0':
            x_start = out
        elif self.objective == 'pred_v':
            x_start = self.predict_start_from_v(x, t, out)
        results.append(x_start)
        
        # 水平翻转
        x_h = torch.flip(x, dims=[-1])
        out_h = self.model(x_h, t)
        if self.objective == 'pred_noise':
            x_start_h = self.predict_start_from_noise(x_h, t, out_h)
        elif self.objective == 'pred_x0':
            x_start_h = out_h
        elif self.objective == 'pred_v':
            x_start_h = self.predict_start_from_v(x_h, t, out_h)
        results.append(torch.flip(x_start_h, dims=[-1]))
        
        # 垂直翻转
        x_v = torch.flip(x, dims=[-2])
        out_v = self.model(x_v, t)
        if self.objective == 'pred_noise':
            x_start_v = self.predict_start_from_noise(x_v, t, out_v)
        elif self.objective == 'pred_x0':
            x_start_v = out_v
        elif self.objective == 'pred_v':
            x_start_v = self.predict_start_from_v(x_v, t, out_v)
        results.append(torch.flip(x_start_v, dims=[-2]))
        
        # 双翻转
        x_hv = torch.flip(x, dims=[-2, -1])
        out_hv = self.model(x_hv, t)
        if self.objective == 'pred_noise':
            x_start_hv = self.predict_start_from_noise(x_hv, t, out_hv)
        elif self.objective == 'pred_x0':
            x_start_hv = out_hv
        elif self.objective == 'pred_v':
            x_start_hv = self.predict_start_from_v(x_hv, t, out_hv)
        results.append(torch.flip(x_start_hv, dims=[-2, -1]))
        
        return torch.stack(results, dim=0).mean(dim=0)        

    def _denoise_with_flip_aug(self, noisy_image, start_t):
        """带 Flip Augmentation 的去噪（提升鲁棒性）"""
        results = []
        
        # 原图
        results.append(self._denoise_single(noisy_image, start_t))
        
        # 水平翻转
        flipped_h = torch.flip(noisy_image, dims=[-1])
        result_h = self._denoise_single(flipped_h, start_t)
        results.append(torch.flip(result_h, dims=[-1]))
        
        # 垂直翻转
        flipped_v = torch.flip(noisy_image, dims=[-2])
        result_v = self._denoise_single(flipped_v, start_t)
        results.append(torch.flip(result_v, dims=[-2]))
        
        # 双翻转
        flipped_hv = torch.flip(noisy_image, dims=[-2, -1])
        result_hv = self._denoise_single(flipped_hv, start_t)
        results.append(torch.flip(result_hv, dims=[-2, -1]))
        
        # 平均
        return torch.stack(results, dim=0).mean(dim=0)

    def p_losses(self, y_bar, original_X, t, matched_state):
        """
        DDM² Stage III: 使用论文的索引体系
        """
        B = y_bar.shape[0]
        device = y_bar.device
        
        if self.problem_dimension == '2D':
            _, C, H, W = y_bar.shape
        else:
            _, C, H, W, D = y_bar.shape
        
        # 【关键】使用 sqrt_alphas_cumprod_prev（和论文一致）
        fixed_alphas = self.sqrt_alphas_cumprod_prev[matched_state]  # (B,)
        
        # reshape
        if self.problem_dimension == '2D':
            fixed_alphas = fixed_alphas.view(B, 1, 1, 1)
        else:
            fixed_alphas = fixed_alphas.view(B, 1, 1, 1, 1)
        
        # 计算噪声（和论文一样的公式）
        noise = (original_X - fixed_alphas * y_bar.detach()) / (1 - fixed_alphas**2).sqrt()
        
        # Eq 4: 均值校正
        noise_mean = noise.mean(dim=tuple(range(1, noise.ndim)), keepdim=True)
        noise = noise - noise_mean.detach()
        
        # 校正 y_bar
        y_bar_corrected = y_bar + noise_mean.detach() * (1 - fixed_alphas**2).sqrt() / fixed_alphas
        
        # Noise shuffle
        if self.problem_dimension == '2D':
            spatial_size = H * W
            noise_flat = noise.view(B, C, -1)
            rand_idx = torch.randperm(spatial_size, device=device)
            noise_shuffled = noise_flat[:, :, rand_idx].view(B, C, H, W).detach()
        else:
            spatial_size = H * W * D
            noise_flat = noise.view(B, C, -1)
            rand_idx = torch.randperm(spatial_size, device=device)
            noise_shuffled = noise_flat[:, :, rand_idx].view(B, C, H, W, D).detach()
        
        # q_sample 构造 S_t
        sqrt_alpha_t = extract(self.sqrt_alphas_cumprod, t, y_bar.shape)
        sqrt_one_minus_alpha_t = extract(self.sqrt_one_minus_alphas_cumprod, t, y_bar.shape)
        
        S_t = sqrt_alpha_t * y_bar_corrected + sqrt_one_minus_alpha_t * noise_shuffled
        
        # 模型预测
        model_out = self.model(S_t, t)
        
        # Loss
        target = original_X
        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        return loss.mean(), model_out, target
   
    
    def forward(self, y_bar, original_X, *args, **kwargs):
        """
        DDM² 前向传播（per-sample matched_state）
        """
        if self.problem_dimension == '2D':
            b, c, h, w = original_X.shape
        else:
            b, c, h, w, d = original_X.shape
        
        device = original_X.device
        
        # 计算 per-sample matched_state（向量化）
        matched_state = self.compute_matched_state(original_X, y_bar)
        
        # 随机采样 t
        t = torch.randint(1, self.num_timesteps, (b,), device=device).long()
        
        # 调用 p_losses
        loss, model_out, target = self.p_losses(y_bar, original_X, t, matched_state)
        
        return loss, model_out, target

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        generator_train,
        generator_val,
        train_batch_size,
        *,
        accum_iter = 1, # gradient accumulation steps
        train_num_steps = 100000, # total training epochs
        results_folder = None,
        train_lr = 1e-4,
        train_lr_decay_every = 200, 
        save_models_every = 1,
        validation_every = 1,
        
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),

        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        
        max_grad_norm = 1.,
         
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model   # it's not just the model architecture, but the actual model with loss calculation
        self.conditional_diffusion = self.model.conditional_diffusion
        print('conditional diffusion: ', self.conditional_diffusion)
        self.channels = diffusion_model.channels
 
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.accum_iter = accum_iter

        # dataset and dataloader
        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl = self.accelerator.prepare(dl)
        self.cycle_dl = cycle(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(self.ds_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl_val = self.accelerator.prepare(dl_val)


        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.max_grad_norm = max_grad_norm
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
        ff.make_folder([self.results_folder])

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.validation_every = validation_every

    @property
    def device(self):
        return self.accelerator.device

    def save(self, stepNum):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'decay_steps': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }
        
        torch.save(data, os.path.join(self.results_folder, 'model-' + str(stepNum) + '.pt'))

    def load_model(self, trained_model_filename):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(trained_model_filename, map_location=device)
        
        self.model.load_state_dict(model_state)
        
        self.step = data['step']

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        self.scheduler.load_state_dict(data['decay_steps'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self, pre_trained_model = None ,start_step = None, beta = 0):
            accelerator = self.accelerator
            device = accelerator.device

            # load pre-trained
            if pre_trained_model is not None:
                self.load_model(pre_trained_model)
                print('model loaded from ', pre_trained_model)

            if start_step is not None:
                self.step = start_step
            
            self.scheduler.step_size = 1
            val_loss = np.inf
            val_diffusion_loss = np.inf
            val_bias_loss = np.inf
            training_log = []

            self.scheduler.step_size = 1
            val_loss = np.inf; val_diffusion_loss = np.inf; val_bias_loss = np.inf
            training_log = []

            with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
                
                while self.step < self.train_num_steps:
                    print('training epoch: ', self.step + 1)
                    print('learning rate: ', self.scheduler.get_last_lr()[0])

                    average_loss = []; average_diffusion_loss = []; average_bias_loss = 0.0
                    count = 0
                    for batch in self.dl:
                        if count == 0 or count % self.accum_iter == 0 or count == len(self.dl) - 1 or count == len(self.dl):
                            self.opt.zero_grad()

                        # dataloader 返回: (y_bar, X_original)
                        batch_ybar, batch_x_orig = batch

                        y_bar = batch_ybar.to(device)
                        x_orig = batch_x_orig.to(device)
                        
                        # 范围转换：[0,1] -> [-1,1]
                        if x_orig.max() <= 1.0 and x_orig.min() >= 0.0:
                            x_orig = x_orig * 2.0 - 1.0
                            y_bar = y_bar * 2.0 - 1.0

                        with self.accelerator.autocast():
                            # 调用新定义的 forward(y_bar, original_X)
                            diffusion_loss, model_output, target = self.model(
                                y_bar, 
                                x_orig
                            )

                            # gauss_kernel = kernel.get_gaussian_kernel(kernel_size=37, sigma=6)
                            # lowpass_out = kernel.apply_lowpass_gaussian(model_output, gauss_kernel)
                            # lowpass_target = kernel.apply_lowpass_gaussian(torch.clone(x_orig), gauss_kernel)

                            # bias_loss = F.mse_loss(lowpass_out, lowpass_target, reduction='mean')

                            loss = diffusion_loss# + beta * bias_loss

                        if count % self.accum_iter == 0 or count == len(self.dl) - 1 or count == len(self.dl):
                            self.accelerator.backward(loss)
                            accelerator.wait_for_everyone()
                            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            self.opt.step()

                        average_loss.append(loss.item())
                        average_diffusion_loss.append(diffusion_loss.item())
                        # average_bias_loss.append(bias_loss.item())  # no bias
                        count += 1

                    average_loss = sum(average_loss) / len(average_loss)
                    average_diffusion_loss = sum(average_diffusion_loss) / len(average_diffusion_loss)
                    # average_bias_loss = sum(average_bias_loss) / len(average_bias_loss)
                
                    pbar.set_description(f'average loss: {average_loss:.4f}, diffusion loss: {average_diffusion_loss:.4f}')
                    accelerator.wait_for_everyone()

                    self.step += 1

                    # save the model
                    if self.step !=0 and divisible_by(self.step, self.save_model_every):
                        print('i am saving model at step: ', self.step)
                        self.save(self.step)
                        print('model saved')
                        # update the parameter
                    if self.step !=0 and divisible_by(self.step, self.train_lr_decay_every):
                        print('i am updating learning rate at step: ', self.step)
                        self.scheduler.step()

                    self.ema.update()

                    # do the validation if necessary
                    if self.step != 0 and divisible_by(self.step, self.validation_every):
                        print('validation at step: ', self.step)
                        self.model.eval()
                        with torch.no_grad():
                            val_loss = []; val_diffusion_loss = []; val_bias_loss = []

                            for batch in self.dl_val:
                                #新的数据格式：dataloader 返回 (y_bar, X_original)
                                batch_ybar, batch_x_orig = batch

                                y_bar  = batch_ybar.to(device)   # N2N 输出 ȳ
                                x_orig = batch_x_orig.to(device) # 原始 noisy X

                                with self.accelerator.autocast():
                                    # 调用新的 forward(y_bar, original_X)
                                    diffusion_loss, model_output, target = self.model(
                                        y_bar,
                                        x_orig
                                    )

                                    # bias loss 部分保持不变，只是目标从 data_x0 换成 x_orig
                                    gauss_kernel   = kernel.get_gaussian_kernel(kernel_size=37, sigma=6)
                                    lowpass_out    = kernel.apply_lowpass_gaussian(model_output, gauss_kernel)
                                    lowpass_target = kernel.apply_lowpass_gaussian(torch.clone(x_orig), gauss_kernel)

                                    bias_loss = F.mse_loss(lowpass_out, lowpass_target, reduction='mean')

                                    loss = diffusion_loss + beta * bias_loss
                                
                                val_loss.append(loss.item())
                                val_diffusion_loss.append(diffusion_loss.item())
                                val_bias_loss.append(bias_loss.item())

                            val_loss           = sum(val_loss) / len(val_loss)
                            val_diffusion_loss = sum(val_diffusion_loss) / len(val_diffusion_loss)
                            val_bias_loss      = sum(val_bias_loss) / len(val_bias_loss)

                            print('validation loss: ', val_loss, 
                                'validation diffusion loss: ', val_diffusion_loss,
                                'validation bias loss: ', val_bias_loss)

                        self.model.train(True)

                    # save the training log
                    training_log.append([self.step,self.scheduler.get_last_lr()[0], average_loss, average_diffusion_loss, 
                                        average_bias_loss, val_loss, val_diffusion_loss, val_bias_loss])
                    df = pd.DataFrame(training_log,columns = ['iteration','learning_rate','training_loss','training_diffusion_loss','training_bias_loss',
                                                                'validation_loss','validation_diffusion_loss','validation_bias_loss'])
                    log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                    df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)

                    # at the end of each epoch, call on_epoch_end
                    self.ds.on_epoch_end(); self.ds_val.on_epoch_end()
                    pbar.update(1)

            accelerator.print('training complete')


# Sampling class
class Sampler(object):
    def __init__(
        self,
        diffusion_model,
        generator,
        batch_size,
        device = 'cuda',
    ):
        super().__init__()

        # model
        self.model = diffusion_model  
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            self.device = torch.device("cpu")

        self.conditional_diffusion = self.model.conditional_diffusion
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling
        self.image_size = diffusion_model.image_size
        self.batch_size = batch_size

        # dataset and dataloader

        self.generator = generator
        dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        
        self.histogram_equalization = self.generator.histogram_equalization
        print('histogram equalization: ', self.histogram_equalization)
        self.bins = self.generator.bins
        self.bins_mapped = self.generator.bins_mapped     
        self.background_cutoff = self.generator.background_cutoff
        self.maximum_cutoff = self.generator.maximum_cutoff
        self.normalize_factor = self.generator.normalize_factor

        self.dl = dl
        self.cycle_dl = cycle(dl)
 
        # EMA:
        self.ema = EMA(diffusion_model)
        self.ema.to(self.device)

    def load_model(self, trained_model_filename):

        data = torch.load(trained_model_filename, map_location=self.device)

        self.model.load_state_dict(data['model'])

        self.step = data['step']

        self.ema.load_state_dict(data["ema"])


    def sample_2D(self, trained_model_filename, condition_img, start_t=None, batch_size=1):
        """
        DDM² 2D 采样
        """
        background_cutoff = self.background_cutoff
        maximum_cutoff = self.maximum_cutoff
        self.load_model(trained_model_filename) 
        
        device = self.device
        self.ema.ema_model.eval()
        
        num_slices = condition_img.shape[-1]
        pred_img = np.zeros((self.image_size[0], self.image_size[1], num_slices), dtype=np.float32)

        dataset = self.dl.dataset
        
        # 安全检查
        if len(dataset) != num_slices:
            raise ValueError(
                f"Dataset length ({len(dataset)}) != Volume slices ({num_slices}). "
                f"请确保 DataLoader 只包含当前病人的数据！"
            )
        
        with torch.inference_mode():

# ==================== 【新增】全局 State Matching ====================
            if start_t is None:
                print("Performing global State Matching...")
                num_samples = min(10, num_slices)
                y_bar_samples = []
                x_orig_samples = []
                
                for idx in range(num_samples):
                    data = dataset[idx]
                    y_bar_samples.append(data[0])
                    x_orig_samples.append(data[1])
                
                y_bar_sample = torch.stack(y_bar_samples, dim=0).to(device)
                x_orig_sample = torch.stack(x_orig_samples, dim=0).to(device)
                
                # 范围转换
                if x_orig_sample.max() <= 1.0 and x_orig_sample.min() >= 0.0:
                    x_orig_sample = x_orig_sample * 2.0 - 1.0
                    y_bar_sample = y_bar_sample * 2.0 - 1.0
                
                start_t = self.ema.ema_model.state_matching(x_orig_sample, y_bar_sample)
                print(f"Global State Matching: t* = {start_t}")
            # ==================== State Matching 结束 ====================

            for batch_start in tqdm(range(0, num_slices, batch_size), desc="DDM² Sampling"):
                batch_end = min(batch_start + batch_size, num_slices)
                
                # 收集 batch 数据
                y_bar_list = []
                x_orig_list = []
                
                for idx in range(batch_start, batch_end):
                    data = dataset[idx]
                    y_bar_list.append(data[0])
                    x_orig_list.append(data[1])
                
                y_bar = torch.stack(y_bar_list, dim=0).to(device)
                x_orig = torch.stack(x_orig_list, dim=0).to(device)
                
                # 调试：检查输入范围
                if batch_start == 0:
                    print(f"DEBUG - x_orig range: [{x_orig.min():.4f}, {x_orig.max():.4f}]")
                    print(f"DEBUG - y_bar range: [{y_bar.min():.4f}, {y_bar.max():.4f}]")
                
                # 【关键】范围检查：如果是 [0,1]，转换为 [-1,1]
                if x_orig.min() >= -0.01 and x_orig.max() <= 1.01:
                    if x_orig.max() <= 1.0 and x_orig.min() >= 0.0:
                        if batch_start == 0:
                            print("DEBUG: Detecting [0, 1] input, converting to [-1, 1]...")
                        x_orig = x_orig * 2.0 - 1.0
                        y_bar = y_bar * 2.0 - 1.0
                
                # DDM² 去噪（开启 flip）
                pred_batch = self.ema.ema_model.ddm2_denoise(
                    noisy_image=x_orig,
                    y_bar=y_bar,
                    start_t=start_t,
                    use_flip=True  # 使用 flip augmentation
                )
                
                if batch_start == 0:
                    print(f"DEBUG - output range: [{pred_batch.min():.4f}, {pred_batch.max():.4f}]")
                
                # 存储
                pred_batch_np = pred_batch.detach().cpu().numpy()
                for i, slice_idx in enumerate(range(batch_start, batch_end)):
                    pred_img[:, :, slice_idx] = pred_batch_np[i, 0, :, :]

        # 后处理
        pred_img = Data_processing.crop_or_pad(
            pred_img, 
            [condition_img.shape[0], condition_img.shape[1], condition_img.shape[-1]], 
            value=np.min(pred_img)
        )
        
        print(f"DEBUG - Before clip: [{pred_img.min():.4f}, {pred_img.max():.4f}]")
        pred_img = np.clip(pred_img, 0, 1)
        pred_img = pred_img * (maximum_cutoff - background_cutoff) + background_cutoff
        
        if self.histogram_equalization:
            pred_img = Data_processing.apply_transfer_to_img(
                pred_img, self.bins, self.bins_mapped, reverse=True
            )
        
        print(f'Final image range: [{pred_img.min():.1f}, {pred_img.max():.1f}]')
        return pred_img
