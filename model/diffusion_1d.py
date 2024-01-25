from accelerate import Accelerator
from collections import namedtuple
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from functools import partial
from imageio import imwrite
import math
from pathlib import Path
import pdb
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch_geometric.data.dataloader import DataLoader
from tqdm.auto import tqdm
from torch.autograd import grad
import sys
import os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from cindm.data.nbody_dataset import NBodyDataset
from cindm.utils import p, get_item_1d , COLOR_LIST,CustomLoss
import numpy as np
import matplotlib.pyplot as plt
from cindm.utils import Printer,CustomSampler
from cindm.filepath import EXP_PATH
import matplotlib.pylab as plt
import matplotlib.backends.backend_pdf
# import UHMC
from cindm.utils import visulization, p
grad_mean_list=[]
ss_list=[]
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2 
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # [32]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class LinearAttentionTemporal(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# model

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] #[64, 64, 128, 256, 512]
        in_out = list(zip(dims[:-1], dims[1:])) #[(64, 64), (64, 128), (128, 256), (256, 512)]
        print(f'[ models/temporal ] Channel dimensions: {in_out}')
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):# x.shape=[batch_size,FLAGS.conditioned_steps+FLAGS.rollout_steps,num_bodies*nun_feactures] time=[batch_size]
        # pdb.set_trace()
        x = einops.rearrange(x, 'b h t -> b t h')
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        x = self.init_conv(x) #[32, 64, num_bodies*nun_feactures]
        r = x.clone()#[32, 64, num_bodies*nun_feactures]

        t = self.time_mlp(time) #time.shape [32]
        # t.shape [25, 256]

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x) #[25, 64, 2] [25, 64, 4]  [25, 64, 8] [25, 64, num_bodies*nun_feactures]

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        x=self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    """
        a: [B]
        t: [B], all with the same number
        x_shape: [B, time_steps, F]
    """
    b, *_ = t.shape
    out = a.gather(-1, t)  # [B], obtaining a[t], for all examples in the batch
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # [B, 1, 1]

def linear_beta_schedule(timesteps):
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
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

import einops
from einops.layers.torch import Rearrange


class TemporalUnet1D(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()
        self.horizon=horizon
        self.transition_dim=transition_dim
        self.channels=transition_dim
        # self.init_conv=nn.Conv1d(transition_dim,dim,7,padding=3)
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}') # [(16, 64), (64, 128), (128, 256), (256, 512)]
        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if self.horizon%8==0:
                is_last = ind >= (num_resolutions - 1) # conditioned 4 to predict 40 steps to meet the shape need for unet
            elif self.horizon%4==0:
                is_last = ind >= (num_resolutions - 2)
            elif self.horizon%2==0:
                is_last = ind >= (num_resolutions - 3)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttentionTemporal(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttentionTemporal(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1) # conditioned 4 to predict 40 steps
            # is_last = ind >= (num_resolutions - 1)
            if self.horizon%8==0:
                self.ups.append(nn.ModuleList([
                    # ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                    # ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                    ResidualTemporalBlock(dim_out * 2, dim_out, embed_dim=time_dim, horizon=horizon),
                    ResidualTemporalBlock(dim_out, dim_in, embed_dim=time_dim, horizon=horizon),
                    Residual(PreNorm(dim_in, LinearAttentionTemporal(dim_in))) if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                    ]))
            elif self.horizon%4==0:
                self.ups.append(nn.ModuleList([
                    # ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                    # ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                    ResidualTemporalBlock(dim_out * 2, dim_out, embed_dim=time_dim, horizon=horizon),
                    ResidualTemporalBlock(dim_out, dim_in, embed_dim=time_dim, horizon=horizon),
                    Residual(PreNorm(dim_in, LinearAttentionTemporal(dim_in))) if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last and ind!=0 else nn.Identity() # conditioned 4 to predict 40 steps
                    ]))
            elif self.horizon%2==0:
                self.ups.append(nn.ModuleList([
                    # ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                    # ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                    ResidualTemporalBlock(dim_out * 2, dim_out, embed_dim=time_dim, horizon=horizon),
                    ResidualTemporalBlock(dim_out, dim_in, embed_dim=time_dim, horizon=horizon),
                    Residual(PreNorm(dim_in, LinearAttentionTemporal(dim_in))) if attention else nn.Identity(),
                    Upsample1d(dim_in) if (not is_last and ind!=0 and ind!=1) else nn.Identity() # conditioned 4 to predict 40 steps
                    ]))
            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time,cond):
        '''
            x : [ batch ,horizon, transition ] 
            time: [batch_size]
        '''
        # pdb.set_trace()
        x = einops.rearrange(x, 'b h t -> b t h')
        # x_cos=torch.cos(x)
        # x_sin=torch.sin(x)
        # x=torch.cat([x,x_cos,x_sin],dim=1)

        # x = self.init_conv(x) #[32, FLAGS.conditioned_steps+FLAGS.rollout_steps,num_bodies*nun_feactures] --> [32,64, num_bodies*nun_feactures]
        t = self.time_mlp(time) #t [batch_szie 64]
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x) #[32, 128, 8] [32, 256, 4] [32, 512, 2]

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x) #[32, 256, 2]

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

class ResidualBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])


        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x inp_channels x horizon ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class Unet1D_forward_model(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()
        self.horizon=horizon
        self.transition_dim=transition_dim
        self.channels=transition_dim
        # self.init_conv=nn.Conv1d(transition_dim,dim,7,padding=3)
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}') # [(16, 64), (64, 128), (128, 256), (256, 512)]

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            # pdb.set_trace()
            if self.horizon%8==0:
                is_last = ind >= (num_resolutions - 1) # conditioned 4 to predict 40 steps to meet the shape need for unet
            elif self.horizon%4==0:
                is_last = ind >= (num_resolutions - 2)
            elif self.horizon%2==0:
                is_last = ind >= (num_resolutions - 3)

            self.downs.append(nn.ModuleList([
                ResidualBlock(dim_in, dim_out, horizon=horizon),
                ResidualBlock(dim_out, dim_out, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttentionTemporal(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttentionTemporal(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1) # conditioned 4 to predict 40 steps
            # is_last = ind >= (num_resolutions - 1)
            if self.horizon%8==0:
                self.ups.append(nn.ModuleList([
                    # ResidualBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                    # ResidualBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                    ResidualBlock(dim_out * 2, dim_out, horizon=horizon),
                    ResidualBlock(dim_out, dim_in, horizon=horizon),
                    Residual(PreNorm(dim_in, LinearAttentionTemporal(dim_in))) if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                    ]))
            elif self.horizon%4==0:
                self.ups.append(nn.ModuleList([
                    # ResidualBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                    # ResidualBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                    ResidualBlock(dim_out * 2, dim_out,horizon=horizon),
                    ResidualBlock(dim_out, dim_in, horizon=horizon),
                    Residual(PreNorm(dim_in, LinearAttentionTemporal(dim_in))) if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last and ind!=0 else nn.Identity() # conditioned 4 to predict 40 steps
                    ]))
            elif self.horizon%2==0:
                self.ups.append(nn.ModuleList([
                    # ResidualBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                    # ResidualBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                    ResidualBlock(dim_out * 2, dim_out,horizon=horizon),
                    ResidualBlock(dim_out, dim_in, horizon=horizon),
                    Residual(PreNorm(dim_in, LinearAttentionTemporal(dim_in))) if attention else nn.Identity(),
                    Upsample1d(dim_in) if (not is_last and ind!=0 and ind!=1) else nn.Identity() # conditioned 4 to predict 40 steps
                    ]))
            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self,cond):
        '''
            x : [ batch ,horizon, transition ] 
            time: [batch_size]
        '''
        # pdb.set_trace()
        x=torch.randn((cond.shape[0],self.horizon,self.transition_dim)).to(cond.device)
        x[:,:cond.shape[1],:]=cond
        x = einops.rearrange(x, 'b h t -> b t h')
        # x_cos=torch.cos(x)
        # x_sin=torch.sin(x)
        # x=torch.cat([x,x_cos,x_sin],dim=1)

        # x = self.init_conv(x) #[32, FLAGS.conditioned_steps+FLAGS.rollout_steps,num_bodies*nun_feactures] --> [32,64, num_bodies*nun_feactures]
        h = []
        # pdb.set_trace()
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x)
            x = resnet2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x) #[32, 128, 8] [32, 256, 4] [32, 512, 2]

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = attn(x)
            x = upsample(x) #[32, 256, 2]

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x



class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        model_unconditioned=None,
        betas_inference=None,
        *,
        image_size, #FLAGS.rollout_steps
        conditioned_steps,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        loss_weight_discount=0.95,
        num_time_steps_UHMC=100,
        is_diffusion_condition=None,
        backward_steps=5,
        backward_lr=1,
    ):
        super().__init__()
        self.model = model
        self.model_unconditioned=model_unconditioned ##
        self.betas_inference=betas_inference
        self.channels =self.model.channels #n_bodies*num_feactures
        # self.self_condition = self.model.self_condition #False
        
        self.is_diffusion_condition=is_diffusion_condition
        self.self_condition =False

        self.num_timesteps_UHMC=num_time_steps_UHMC

        self.image_size = image_size

        self.conditioned_steps=conditioned_steps
        self.rollout_steps=image_size

        self.objective = objective
        self.backward_steps = backward_steps
        self.backward_lr = backward_lr

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas #[1000]
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.loss_weight_discount=loss_weight_discount
        # sampling related parameters
        # scalar_for_gradient=np.sqrt(1 / (1 - alphas_cumprod))
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)  # from 0 to 1
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        # register_buffer('scalar_for_gradient', scalar_for_gradient)
        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

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
        """
        posterior_mean_coef1: betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2: (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +  
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, cond, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, **kwargs):
        """
            x: [B, pred_steps, F] F: feature_size
            cond: [B, cond_steps, F]
        """
        if self.conditioned_steps!=0:
            x = torch.cat([cond, x], dim=1)  

        if "compose_mode" in kwargs and "inside" in kwargs["compose_mode"]:
            compose_mode = kwargs["compose_mode"]
            n_composed = kwargs["n_composed"]
            compose_start_step = kwargs["compose_start_step"]
            single_model_step = kwargs["single_model_step"]
            compose_n_bodies = kwargs["compose_n_bodies"]
            assert single_model_step > 0

            """
            x: [B, T+n_composed*t1, feature_size]
            pred_noise_aggr: has shape [n_composed+1, B, T+n_composed*t1, compose_n_bodies (sender), compose_n_bodies (receiver), 4]
            mask_aggr: [n_composed+1, B, T+n_composed*t1, compose_n_bodies*4]
            """

            # Initialize pred_noise_aggr:
            pred_noise_aggr = torch.zeros((n_composed + 1, x.shape[0], x.shape[1], compose_n_bodies, compose_n_bodies, 4), device=x.device, dtype=x.dtype)
            mask_aggr = torch.zeros((n_composed + 1,) + x.shape, device=x.device, dtype=x.dtype)
            # Compose:
            for kk in range(n_composed + 1):
                mask_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step] = torch.ones(x.shape[0], single_model_step, compose_n_bodies*4)
                for ii in range(compose_n_bodies):
                    for jj in range(compose_n_bodies):
                        if ii < jj:
                            index = torch.cat([torch.arange(ii*4, (ii+1)*4), torch.arange(jj*4, (jj+1)*4)])

                            pred_noise_ele = self.model(
                                x[:,kk*compose_start_step: kk*compose_start_step + single_model_step, index],
                                t,
                                x_self_cond,
                            )
                            pred_noise_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, jj, ii] = pred_noise_ele[...,:4]  # (gradient from jj to ii)
                            pred_noise_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, ii, jj] = pred_noise_ele[...,4:]  # (gradient from ii to jj)

            # pred_noise_aggr: 
            #   before [n_composed+1, B, T+n_composed*t1, compose_n_bodies (sender), compose_n_bodies (receiver), 4]
            if compose_mode == "mean-inside":
                pred_noise_aggr = (pred_noise_aggr.sum(-3)/(compose_n_bodies-1)).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]
                model_output = pred_noise_aggr.sum(0) / mask_aggr.sum(0)
            elif compose_mode == "sum-inside":
                pred_noise_aggr = (pred_noise_aggr.sum(-3)).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]
                model_output = pred_noise_aggr.sum(0) / mask_aggr.mean(0)
            else:
                raise
        else:
            if self.model_unconditioned != None:
                model_output = self.gradient(x, t[0], 4)
            else:
                model_output = self.model(x, t, x_self_cond) # [B, pred_steps + cond_steps, F], cost about 0.02s
        grad_mean_list.append(model_output.mean().to("cpu"))
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
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
        if self.conditioned_steps!=0:
            pred_noise = pred_noise[:, cond.size(1):]
            x_start = x_start[:, cond.size(1):]
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, cond, t, x_self_cond = None, clip_denoised = True, **kwargs):
        preds = self.model_predictions(x, cond, t, x_self_cond, **kwargs)
        x_start = preds.pred_x_start
        pred_noise = preds.pred_noise

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t) # posterior_variance, posterior_log_variance: [B, 1, 1]
        # model_mean_from_noise = (x - extract(self.betas / self.sqrt_one_minus_alphas_cumprod, t, x.shape) * preds.pred_noise) / extract(torch.sqrt(1 - self.betas), t, x.shape)

        return model_mean, posterior_variance, posterior_log_variance, x_start, pred_noise

    @torch.no_grad()
    def p_sample(
        self,
        x,
        cond,
        t: int,
        x_self_cond = None,
        clip_denoised = True,
        design_fn = None,
        design_guidance = "standard",
        initial_state_overwrite=None,
    ):
        """
        Different design_guidance follows the paper "Universal Guidance for Diffusion Models"
        """
        if "recurrence" not in design_guidance:
            b, *_, device = *x.shape, x.device
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            model_mean, _, model_log_variance, x_start, _ = self.p_mean_variance(
                x = x,
                cond=cond,
                t = batched_times,
                x_self_cond = x_self_cond,
                clip_denoised = clip_denoised,
            )
            
            if design_fn is not None:
                eta = extract(self.betas, batched_times, x.shape) / extract(torch.sqrt(self.alphas_cumprod_prev), batched_times, x.shape)
                if design_guidance.startswith("standard"):
                    with torch.enable_grad():
                        x_clone = x.clone().detach().requires_grad_()
                        design_obj = design_fn(x_clone)
                        grad_design = grad(design_obj, x_clone)[0]
                    if design_guidance == "standard":
                        grad_design_final = grad_design
                    elif design_guidance == "standard-alpha":
                        grad_design_final = eta * grad_design
                    else:
                        raise
                elif design_guidance.startswith("universal"):
                    if design_guidance == "universal-forward":
                        with torch.enable_grad():
                            x_clone = x_start.clone().detach().requires_grad_()
                            design_obj = design_fn(x_clone)
                            grad_design = grad(design_obj, x_clone)[0]
                        grad_design_final = eta * grad_design
                    elif design_guidance == "universal-backward":
                        with torch.enable_grad():
                            x_clone = x_start.clone().detach().requires_grad_()
                            for kk in range(self.backward_steps):
                                design_obj = design_fn(x_clone)
                                grad_design = grad(design_obj, x_clone)[0]
                                if kk == 1:
                                    grad_design_final = eta * grad_design
                                x_clone = x_clone - grad_design * self.backward_lr
                                x_clone = x_clone.clone().detach().requires_grad_()
                        delta_x0 = x_clone.clone().detach() - x_start.clone().detach()
                        grad_design_final = grad_design_final - extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                    else:
                        raise
                pred_img = model_mean - grad_design_final
            else:
                pred_img = model_mean

            # overwrite the first few steps:
            if initial_state_overwrite is not None:
                initial_steps = initial_state_overwrite.shape[1]
                pred_img = torch.cat([initial_state_overwrite, pred_img[:,initial_steps:]], 1)
                # # Delta_x0:
                # delta_x0 = torch.cat([initial_state_overwrite - pred_img[:,:initial_steps], torch.zeros_like(pred_img[:,initial_steps:])], 1)
                # # Translate back to x_t space:
                # pred_img = pred_img + extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
            pred_img = pred_img + (0.5 * model_log_variance).exp() * noise
            return pred_img, x_start
        else:
            b, *_, device = *x.shape, x.device
            recurrence_times = eval(design_guidance.split("-")[-1])
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            for recurrence_n in range(recurrence_times):
                # Calculate x0 (x_start) and \mu_t(x_t, x0) from x_t:
                model_mean, _, model_log_variance, x_start, _ = self.p_mean_variance(x = x, cond=cond, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
                # Update \mu_t (estimation of x_{t-1} without noise) using design function:
                if design_fn is not None:
                    eta = extract(self.betas, batched_times, x.shape) / extract(torch.sqrt(self.alphas_cumprod_prev), batched_times, x.shape)
                    if design_guidance.startswith("standard"):
                        with torch.enable_grad():
                            x_clone = x.clone().detach().requires_grad_()
                            design_obj = design_fn(x_clone)
                            grad_design = grad(design_obj, x_clone)[0]
                        if design_guidance.startswith("standard-recurrence"):
                            grad_design_final = grad_design
                        elif design_guidance.startswith("standard-alpha-recurrence"):
                            grad_design_final = eta * grad_design
                        else:
                            raise
                    elif design_guidance.startswith("universal"):
                        if design_guidance.startswith("universal-forward-recurrence"):
                            with torch.enable_grad():
                                x_clone = x_start.clone().detach().requires_grad_()
                                design_obj = design_fn(x_clone)
                                grad_design = grad(design_obj, x_clone)[0]
                            grad_design_final = eta * grad_design
                        elif design_guidance.startswith("universal-backward-recurrence"):
                            with torch.enable_grad():
                                x_clone = x_start.clone().detach().requires_grad_()
                                for kk in range(self.backward_steps):
                                    design_obj = design_fn(x_clone)
                                    grad_design = grad(design_obj, x_clone)[0]
                                    if kk == 1:
                                        grad_design_final = eta * grad_design
                                    x_clone = x_clone - grad_design * self.backward_lr
                                    x_clone = x_clone.clone().detach().requires_grad_()
                            delta_x0 = x_clone.clone().detach() - x_start.clone().detach()
                            grad_design_final = grad_design_final - extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                        else:
                            raise
                    # Obtain updated \mu_t (estimation of x_{t-1} without noise)
                    pred_img = model_mean - grad_design_final
                else:
                    # Obtain updated \mu_t (estimation of x_{t-1} without noise)
                    pred_img = model_mean
                
                # overwrite the first few steps:
                if initial_state_overwrite is not None:
                    # initial_steps = initial_state_overwrite.shape[1]
                    # # Delta_x0:
                    # delta_x0 = torch.cat([initial_state_overwrite - pred_img[:,:initial_steps], torch.zeros_like(pred_img[:,initial_steps:])], 1)
                    # # Translate back to x_t space:
                    # pred_img = pred_img + extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                    initial_steps = initial_state_overwrite.shape[1]
                    pred_img = torch.cat([initial_state_overwrite, pred_img[:,initial_steps:]], 1)

                # Relaxation:
                noise_prime = torch.randn_like(pred_img)
                x = extract(torch.sqrt(self.alphas_cumprod / self.alphas_cumprod_prev), batched_times, x.shape) * pred_img + \
                    extract(torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev), batched_times, x.shape) * noise_prime

            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
            pred_img = pred_img + (0.5 * model_log_variance).exp() * noise
            return pred_img, x_start


    @torch.no_grad()
    def p_sample_compose_inside(
        self,
        x,
        cond,
        t: int,
        x_self_cond = None,
        clip_denoised = True,
        design_fn = None,
        design_guidance = "standard",
        initial_state_overwrite = None,
        compose_mode = "mean-inside",
        n_composed = 0,
        compose_start_step = 4,
        single_model_step = -1,
        compose_n_bodies = 2,
    ):
        """
        Different design_guidance follows the paper "Universal Guidance for Diffusion Models"
        """
        if "recurrence" not in design_guidance:
            b, *_, device = *x.shape, x.device
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            if "inside" not in compose_mode:
                model_mean, _, model_log_variance, x_start, pred_noise__ = self.p_mean_variance(
                    x = x,
                    cond=cond,
                    t=batched_times,
                    x_self_cond=x_self_cond,
                    clip_denoised=clip_denoised,
                    compose_mode=compose_mode,
                )
            else:
                model_mean, _, model_log_variance, x_start, _ = self.p_mean_variance(
                    x = x,
                    cond=cond,
                    t=batched_times,
                    x_self_cond=x_self_cond,
                    clip_denoised=clip_denoised,
                    compose_mode=compose_mode,
                    n_composed=n_composed,
                    compose_start_step=compose_start_step,
                    single_model_step=single_model_step,
                    compose_n_bodies=compose_n_bodies,
                )
            
            if design_fn is not None:
                eta = extract(self.betas, batched_times, x.shape) / extract(torch.sqrt(self.alphas_cumprod_prev), batched_times, x.shape)
                if design_guidance.startswith("standard"):
                    with torch.enable_grad():
                        x_clone = x.clone().detach().requires_grad_()
                        design_obj = design_fn(x_clone)
                        grad_design = grad(design_obj, x_clone)[0]
                    if design_guidance == "standard":
                        grad_design_final = grad_design
                    elif design_guidance == "standard-alpha":
                        grad_design_final = eta * grad_design
                    else:
                        raise
                elif design_guidance.startswith("universal"):
                    if design_guidance == "universal-forward":
                        with torch.enable_grad():
                            x_clone = x_start.clone().detach().requires_grad_()
                            design_obj = design_fn(x_clone)
                            grad_design = grad(design_obj, x_clone)[0]
                        grad_design_final = eta * grad_design
                    elif design_guidance == "universal-backward":
                        with torch.enable_grad():
                            x_clone = x_start.clone().detach().requires_grad_()
                            for kk in range(self.backward_steps):
                                design_obj = design_fn(x_clone)
                                grad_design = grad(design_obj, x_clone)[0]
                                if kk == 1:
                                    grad_design_final = eta * grad_design
                                x_clone = x_clone - grad_design * self.backward_lr
                                x_clone = x_clone.clone().detach().requires_grad_()
                        delta_x0 = x_clone.clone().detach() - x_start.clone().detach()
                        grad_design_final = grad_design_final - extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                    else:
                        raise
                pred_img = model_mean - grad_design_final
            else:
                pred_img = model_mean

            # overwrite the first few steps:
            if initial_state_overwrite is not None:
                initial_steps = initial_state_overwrite.shape[1]
                pred_img = torch.cat([initial_state_overwrite, pred_img[:,initial_steps:]], 1)
                # # Delta_x0:
                # delta_x0 = torch.cat([initial_state_overwrite - pred_img[:,:initial_steps], torch.zeros_like(pred_img[:,initial_steps:])], 1)
                # # Translate back to x_t space:
                # pred_img = pred_img + extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
            pred_img = pred_img + (0.5 * model_log_variance).exp() * noise
            return pred_img, x_start
        else:
            b, *_, device = *x.shape, x.device
            recurrence_times = eval(design_guidance.split("-")[-1])
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            for recurrence_n in range(recurrence_times):
                # Calculate x0 (x_start) and \mu_t(x_t, x0) from x_t:
                if "inside" not in compose_mode:
                    model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(
                        x = x,
                        cond=cond,
                        t=batched_times,
                        x_self_cond=x_self_cond,
                        clip_denoised=clip_denoised,
                        compose_mode=compose_mode,
                    )
                else:
                    model_mean, _, model_log_variance, x_start,pred_noise = self.p_mean_variance(
                        x = x,
                        cond=cond,
                        t=batched_times,
                        x_self_cond=x_self_cond,
                        clip_denoised=clip_denoised,
                        compose_mode=compose_mode,
                        n_composed=n_composed,
                        compose_start_step=compose_start_step,
                        single_model_step=single_model_step,
                        compose_n_bodies=compose_n_bodies,
                    )
                # pdb.set_trace()
                # Update \mu_t (estimation of x_{t-1} without noise) using design function:
                if design_fn is not None:
                    eta = extract(self.betas, batched_times, x.shape) / extract(torch.sqrt(self.alphas_cumprod_prev), batched_times, x.shape)
                    if design_guidance.startswith("standard"):
                        with torch.enable_grad():
                            x_clone = x.clone().detach().requires_grad_()
                            design_obj = design_fn(x_clone)
                            grad_design = grad(design_obj, x_clone)[0]
                        if design_guidance.startswith("standard-recurrence"):
                            grad_design_final = grad_design
                        elif design_guidance.startswith("standard-alpha-recurrence"):
                            grad_design_final = eta * grad_design
                        else:
                            raise
                    elif design_guidance.startswith("universal"):
                        if design_guidance.startswith("universal-forward-recurrence"):
                            with torch.enable_grad():
                                x_clone = x_start.clone().detach().requires_grad_()
                                design_obj = design_fn(x_clone)
                                grad_design = grad(design_obj, x_clone)[0]
                            grad_design_final = eta * grad_design
                        elif design_guidance.startswith("universal-backward-recurrence"):
                            with torch.enable_grad():
                                x_clone = x_start.clone().detach().requires_grad_()
                                for kk in range(self.backward_steps):
                                    design_obj = design_fn(x_clone)
                                    grad_design = grad(design_obj, x_clone)[0]
                                    if kk == 1:
                                        grad_design_final = eta * grad_design
                                    x_clone = x_clone - grad_design * self.backward_lr
                                    x_clone = x_clone.clone().detach().requires_grad_()
                            delta_x0 = x_clone.clone().detach() - x_start.clone().detach()
                            grad_design_final = grad_design_final - extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                        else:
                            raise
                    # Obtain updated \mu_t (estimation of x_{t-1} without noise)
                    pred_img = model_mean - grad_design_final
                else:
                    # Obtain updated \mu_t (estimation of x_{t-1} without noise)
                    pred_img = model_mean
                
                # overwrite the first few steps:
                if initial_state_overwrite is not None:
                    # initial_steps = initial_state_overwrite.shape[1]
                    # # Delta_x0:
                    # delta_x0 = torch.cat([initial_state_overwrite - pred_img[:,:initial_steps], torch.zeros_like(pred_img[:,initial_steps:])], 1)
                    # # Translate back to x_t space:
                    # pred_img = pred_img + extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                    initial_steps = initial_state_overwrite.shape[1]
                    pred_img = torch.cat([initial_state_overwrite, pred_img[:,initial_steps:]], 1)

                # Relaxation:
                noise_prime = torch.randn_like(pred_img)
                x = extract(torch.sqrt(self.alphas_cumprod / self.alphas_cumprod_prev), batched_times, x.shape) * pred_img + \
                    extract(torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev), batched_times, x.shape) * noise_prime

            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
            pred_img = pred_img + (0.5 * model_log_variance).exp() * noise
            # pdb.set_trace()
            if self.sampling_timesteps==1000:
                return pred_img, x_start
            else:
                pred_noise_design=pred_noise+grad_design_final
                return pred_noise_design,x_start


    @torch.no_grad()
    def p_sample_compose_outside(
        self,
        x,
        cond,
        t: int,
        x_self_cond = None,
        clip_denoised = True,
        design_fn = None,
        design_guidance = "standard",
        compose_mode = "mean",
        n_composed = 0,
        compose_start_step = 4,
        single_model_step = -1,
        compose_n_bodies = 2,
        initial_state_overwrite = None,
    ):
        """
        Different design_guidance follows the paper "Universal Guidance for Diffusion Models"

        The model operate on [0, T], [t1, T+t1], [2*t1, T+2*t1],...[n_composed*t1, T+n_composed*t1]

        Args:
            x: [B, T+n_composed*t1, feature_size]
            model_mean_aggr, x_start_aggr: has shape [n_composed+1, B, T+n_composed*t1, compose_n_bodies (sender), compose_n_bodies (receiver), 4]
            mask_aggr: [n_composed+1, B, T+n_composed*t1, compose_n_bodies*4]
        """
        if "recurrence" not in design_guidance:
            b, *_, device = *x.shape, x.device
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)

            # Compute the \hat{x}_{t-1} (model_mean) and \hat{x}_0 (x_start) for each composed time chunk
            # and average over them:
            assert single_model_step > 0
            if compose_mode == "mean":
                model_mean_aggr = torch.zeros((n_composed + 1, x.shape[0], x.shape[1], compose_n_bodies, compose_n_bodies, 4), device=x.device, dtype=x.dtype)
                x_start_aggr = torch.zeros((n_composed + 1, x.shape[0], x.shape[1], compose_n_bodies, compose_n_bodies, 4), device=x.device, dtype=x.dtype)
            elif compose_mode == "noise_sum":
                pred_noise_aggr = torch.zeros((n_composed + 1, x.shape[0], x.shape[1], compose_n_bodies, compose_n_bodies, 4), device=x.device, dtype=x.dtype)
            else:
                raise
            mask_aggr = torch.zeros((n_composed + 1,) + x.shape, device=x.device, dtype=x.dtype)
            for kk in range(n_composed + 1):
                mask_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step] = torch.ones(x.shape[0], single_model_step, compose_n_bodies*4)
                for ii in range(compose_n_bodies):
                    for jj in range(compose_n_bodies):
                        if ii < jj:
                            index = torch.cat([torch.arange(ii*4, (ii+1)*4), torch.arange(jj*4, (jj+1)*4)])

                            model_mean_ele, _, model_log_variance, x_start_ele, pred_noise_ele = self.p_mean_variance(
                                x = x[:,kk*compose_start_step: kk*compose_start_step + single_model_step, index],
                                cond=cond,
                                t = batched_times,
                                x_self_cond = x_self_cond,
                                clip_denoised = clip_denoised,
                            )
                            
                            if compose_mode == "mean":
                                model_mean_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, jj, ii] = model_mean_ele[...,:4]  # (gradient from jj to ii)
                                model_mean_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, ii, jj] = model_mean_ele[...,4:]  # (gradient from ii to jj)
                                x_start_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, jj, ii] = x_start_ele[...,:4]
                                x_start_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, ii, jj] = x_start_ele[...,4:]
                            elif compose_mode == "noise_sum":
                                pred_noise_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, jj, ii] = pred_noise_ele[...,:4]  # (gradient from jj to ii)
                                pred_noise_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, ii, jj] = pred_noise_ele[...,4:]  # (gradient from ii to jj)
                            else:
                                raise
            
            if compose_mode == "mean":
                model_mean_aggr = (model_mean_aggr.sum(-3)/(compose_n_bodies-1)).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]
                x_start_aggr = (x_start_aggr.sum(-3)/(compose_n_bodies-1)).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]

                x_start = x_start_aggr.sum(0) / mask_aggr.sum(0)  # [B, T+n_composed*t1, compose_n_bodies*4]
                model_mean = model_mean_aggr.sum(0) / mask_aggr.sum(0)  # [B, T+n_composed*t1, compose_n_bodies*4]
            elif compose_mode == "noise_sum":
                # pred_noise_aggr: 
                #   before [n_composed+1, B, T+n_composed*t1, compose_n_bodies (sender), compose_n_bodies (receiver), 4]
                #  
                pred_noise_aggr = pred_noise_aggr.sum(-3).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]
                pred_noise = pred_noise_aggr.sum(0) / mask_aggr.mean(0)
                x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
                if clip_denoised:
                    x_start.clamp_(-1., 1.)

                model_mean, _, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = batched_times)
                
            else:
                raise

            if design_fn is not None:
                eta = extract(self.betas, batched_times, x.shape) / extract(torch.sqrt(self.alphas_cumprod_prev), batched_times, x.shape)
                if design_guidance.startswith("standard"):
                    with torch.enable_grad():
                        x_clone = x.clone().detach().requires_grad_()
                        design_obj = design_fn(x_clone)
                        grad_design = grad(design_obj, x_clone)[0]
                    if design_guidance == "standard":
                        grad_design_final = grad_design
                    elif design_guidance == "standard-alpha":
                        grad_design_final = eta * grad_design
                    else:
                        raise
                elif design_guidance.startswith("universal"):
                    if design_guidance.startswith("universal-forward"):
                        with torch.enable_grad():
                            x_clone = x_start.clone().detach().requires_grad_()
                            design_obj = design_fn(x_clone)
                            grad_design = grad(design_obj, x_clone)[0]
                        if design_guidance == "universal-forward":
                            grad_design_final = eta * grad_design
                        elif design_guidance == "universal-forward-pure":
                            grad_design_final = grad_design
                        else:
                            raise
                    elif design_guidance.startwith("universal-backward"):
                        with torch.enable_grad():
                            x_clone = x_start.clone().detach().requires_grad_()
                            for kk in range(self.backward_steps):
                                design_obj = design_fn(x_clone)
                                grad_design = grad(design_obj, x_clone)[0]
                                if kk == 1:
                                    if design_guidance == "universal-backward":
                                        grad_design_final = eta * grad_design
                                    elif design_guidance == "universal-backward-pure":
                                        grad_design_final = grad_design
                                    else:
                                        raise
                                x_clone = x_clone - grad_design * self.backward_lr
                                x_clone = x_clone.clone().detach().requires_grad_()
                        delta_x0 = x_clone.clone().detach() - x_start.clone().detach()
                        grad_design_final = grad_design_final - extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                    else:
                        raise
                pred_img = model_mean - grad_design_final
            else:
                pred_img = model_mean

            # overwrite the first few steps:
            if initial_state_overwrite is not None:
                initial_steps = initial_state_overwrite.shape[1]
                pred_img = torch.cat([initial_state_overwrite, pred_img[:,initial_steps:]], 1)

            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
            pred_img = pred_img + (0.5 * model_log_variance).exp() * noise
            return pred_img, x_start
        else:
            b, *_, device = *x.shape, x.device
            recurrence_times = eval(design_guidance.split("-")[-1])
            batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
            for recurrence_n in range(recurrence_times):

                # Compute the \hat{x}_{t-1} (model_mean) and \hat{x}_0 (x_start) for each composed time chunk
                # and average over them:
                assert single_model_step > 0
                if compose_mode == "mean":
                    model_mean_aggr = torch.zeros((n_composed + 1, x.shape[0], x.shape[1], compose_n_bodies, compose_n_bodies, 4), device=x.device, dtype=x.dtype)
                    x_start_aggr = torch.zeros((n_composed + 1, x.shape[0], x.shape[1], compose_n_bodies, compose_n_bodies, 4), device=x.device, dtype=x.dtype)
                elif compose_mode == "noise_sum":
                    pred_noise_aggr = torch.zeros((n_composed + 1, x.shape[0], x.shape[1], compose_n_bodies, compose_n_bodies, 4), device=x.device, dtype=x.dtype)
                else:
                    raise
                mask_aggr = torch.zeros((n_composed + 1,) + x.shape, device=x.device, dtype=x.dtype)

                for kk in range(n_composed + 1):
                    mask_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step] = torch.ones(x.shape[0], single_model_step, compose_n_bodies*4)
                    for ii in range(compose_n_bodies):
                        for jj in range(compose_n_bodies):
                            if ii < jj:
                                index = torch.cat([torch.arange(ii*4, (ii+1)*4), torch.arange(jj*4, (jj+1)*4)])
                                model_mean_ele, _, model_log_variance, x_start_ele, pred_noise_ele = self.p_mean_variance(
                                    x = x[:,kk*compose_start_step: kk*compose_start_step + single_model_step, index],
                                    cond=cond,
                                    t = batched_times,
                                    x_self_cond = x_self_cond,
                                    clip_denoised = clip_denoised,
                                )

                                if compose_mode == "mean":
                                    model_mean_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, jj, ii] = model_mean_ele[...,:4]  # (gradient from jj to ii)
                                    model_mean_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, ii, jj] = model_mean_ele[...,4:]  # (gradient from ii to jj)
                                    x_start_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, jj, ii] = x_start_ele[...,:4]
                                    x_start_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, ii, jj] = x_start_ele[...,4:]
                                elif compose_mode == "noise_sum":
                                    pred_noise_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, jj, ii] = pred_noise_ele[...,:4]  # (gradient from jj to ii)
                                    pred_noise_aggr[kk,:,kk*compose_start_step: kk*compose_start_step + single_model_step, ii, jj] = pred_noise_ele[...,4:]  # (gradient from ii to jj)
                                else:
                                    raise

                
                if compose_mode == "mean":
                    model_mean_aggr = (model_mean_aggr.sum(-3)/(compose_n_bodies-1)).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]
                    x_start_aggr = (x_start_aggr.sum(-3)/(compose_n_bodies-1)).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]

                    x_start = x_start_aggr.sum(0) / mask_aggr.sum(0)  # [B, T+n_composed*t1, compose_n_bodies*4]
                    model_mean = model_mean_aggr.sum(0) / mask_aggr.sum(0)  # [B, T+n_composed*t1, compose_n_bodies*4]
                elif compose_mode == "noise_sum":
                    # pred_noise_aggr: 
                    #   before [n_composed+1, B, T+n_composed*t1, compose_n_bodies (sender), compose_n_bodies (receiver), 4]
                    #  
                    pred_noise_aggr = pred_noise_aggr.sum(-3).flatten(start_dim=3)  # [n_composed + 1, B, T+n_composed*t1, compose_n_bodies*4]
                    pred_noise = pred_noise_aggr.sum(0) / mask_aggr.mean(0) # [B, T+n_composed*t1, compose_n_bodies*4]
                    x_start = self.predict_start_from_noise(x, batched_times, pred_noise)
                    if clip_denoised:
                        x_start.clamp_(-1., 1.)

                    model_mean, _, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = batched_times)

                else:
                    raise

                # Update \mu_t (estimation of x_{t-1} without noise) using design function:
                if design_fn is not None:
                    eta = extract(self.betas, batched_times, x.shape) / extract(torch.sqrt(self.alphas_cumprod_prev), batched_times, x.shape)
                    if design_guidance.startswith("standard"):
                        with torch.enable_grad():
                            x_clone = x.clone().detach().requires_grad_()
                            design_obj = design_fn(x_clone)
                            grad_design = grad(design_obj, x_clone)[0]
                        if design_guidance.startswith("standard-recurrence"):
                            grad_design_final = grad_design
                        elif design_guidance.startswith("standard-alpha-recurrence"):
                            grad_design_final = eta * grad_design
                        else:
                            raise
                    elif design_guidance.startswith("universal"):
                        if design_guidance.startswith("universal-forward"):
                            with torch.enable_grad():
                                x_clone = x_start.clone().detach().requires_grad_()
                                design_obj = design_fn(x_clone)
                                grad_design = grad(design_obj, x_clone)[0]
                            if design_guidance.startswith("universal-forward-recurrence"):
                                grad_design_final = eta * grad_design
                            elif design_guidance.startswith("universal-forward-pure-recurrence"):
                                grad_design_final = grad_design
                            else:
                                raise
                        elif design_guidance.startswith("universal-backward"):
                            with torch.enable_grad():
                                x_clone = x_start.clone().detach().requires_grad_()
                                for kk in range(self.backward_steps):
                                    design_obj = design_fn(x_clone)
                                    grad_design = grad(design_obj, x_clone)[0]
                                    if kk == 1:
                                        if design_guidance.startswith("universal-backward-recurrence"):
                                            grad_design_final = eta * grad_design
                                        elif design_guidance.startswith("universal-backward-pure-recurrence"):
                                            grad_design_final = grad_design
                                        else:
                                            raise
                                    x_clone = x_clone - grad_design * self.backward_lr
                                    x_clone = x_clone.clone().detach().requires_grad_()
                            delta_x0 = x_clone.clone().detach() - x_start.clone().detach()
                            grad_design_final = grad_design_final - extract(self.sqrt_alphas_cumprod * self.betas / (torch.sqrt(1 - self.betas) * (1 - self.alphas_cumprod)), batched_times, x.shape) * delta_x0
                        else:
                            raise
                    # Obtain updated \mu_t (estimation of x_{t-1} without noise)
                    pred_img = model_mean - grad_design_final
                else:
                    # Obtain updated \mu_t (estimation of x_{t-1} without noise)
                    pred_img = model_mean
                
                # overwrite the first few steps:
                if initial_state_overwrite is not None:
                    initial_steps = initial_state_overwrite.shape[1]
                    pred_img = torch.cat([initial_state_overwrite, pred_img[:,initial_steps:]], 1)

                # Relaxation:
                noise_prime = torch.randn_like(pred_img)
                x = extract(torch.sqrt(self.alphas_cumprod / self.alphas_cumprod_prev), batched_times, x.shape) * pred_img + \
                    extract(torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev), batched_times, x.shape) * noise_prime

            noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
            pred_img = pred_img + (0.5 * model_log_variance).exp() * noise
            return pred_img, x_start
            

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        n_composed=0,
        compose_start_step=4,
        compose_n_bodies=2,
        compose_mode="mean",
        design_fn=None,
        design_guidance="standard",
        initial_state_overwrite=None,
        initialization_mode=0,
        initialization_img=None,
    ):
        
        batch, device = shape[0], self.betas.device
        if initialization_mode==0:
            img = torch.randn((shape[0], shape[1] + n_composed * compose_start_step, compose_n_bodies*4), device=device)  # [B, T+n_composed*t1, F]
        elif initialization_mode==1:
            img=initialization_img.reshape((shape[0], shape[1] + n_composed * compose_start_step, compose_n_bodies*4))
        else:
            img=initialization_img.reshape((shape[0], shape[1] + n_composed * compose_start_step, compose_n_bodies*4))+\
                torch.randn((shape[0], shape[1] + n_composed * compose_start_step, compose_n_bodies*4), device=device)
        assert compose_start_step < shape[1]

        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            if "inside" in compose_mode:
                img, x_start = self.p_sample_compose_inside(
                    img,
                    cond,
                    t,
                    self_cond,
                    design_fn=design_fn,
                    design_guidance=design_guidance,
                    compose_mode=compose_mode,
                    n_composed=n_composed,
                    compose_start_step=compose_start_step,
                    single_model_step=shape[1],
                    compose_n_bodies=compose_n_bodies,
                    initial_state_overwrite=initial_state_overwrite,
                )
            else:
                """The model operate on [0, T], [t1, T+t1], [2*t1, T+2*t1],...[n_composed*t1, T+n_composed*t1]"""
                img, x_start = self.p_sample_compose_outside(
                    img,
                    cond,
                    t,
                    self_cond,
                    design_fn=design_fn,
                    design_guidance=design_guidance,
                    initial_state_overwrite=initial_state_overwrite,
                    n_composed=n_composed,
                    compose_start_step=compose_start_step,
                    single_model_step=shape[1],
                    compose_n_bodies=compose_n_bodies,
                    compose_mode=compose_mode,
                )
            if self.conditioned_steps == 0 and cond != None:
                time_cond = torch.full((shape[0],), t, device=device, dtype=torch.long)
                noise_cond = torch.randn_like(cond).to(device=device)
                img[:,:cond.shape[1],:] = self.q_sample(x_start=cond, t=time_cond, noise=noise_cond)#set first several time steps with groundtruth or named as condition

        return img


    @torch.no_grad()
    def ddim_sample(self, 
                    shape,
                    cond,
                    n_composed=None,
                    clip_denoised = True,
                    compose_start_step=4,
                    compose_n_bodies=2,
                    compose_mode="mean",
                    design_fn=None,
                    design_guidance="standard",
                    initial_state_overwrite=None,
                    initialization_mode=0,
                    initialization_img=None,
                ):
        # pdb.set_trace()
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # if self.conditioned_steps==0 and cond.shape[1]!=0:
        #     pass
        # else:
        img = torch.randn((shape[0], shape[1], shape[2]), device = device)

        x_start = None
        # pdb.set_trace()
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((shape[0],), time, device=device, dtype=torch.long) #time_cond.shape [25,256]
            self_cond = x_start if self.self_condition else None
            if design_fn==None:
                pred_noise, x_start, *_ = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = clip_denoised)##every step cost 0.02ms
            else:
                pred_noise, x_start=self.p_sample_compose_inside(
                    img,
                    cond,
                    time,
                    self_cond,
                    design_fn=design_fn,
                    design_guidance=design_guidance,
                    compose_mode=compose_mode,
                    n_composed=n_composed,
                    compose_start_step=compose_start_step,
                    single_model_step=shape[1],
                    compose_n_bodies=compose_n_bodies,
                    initial_state_overwrite=initial_state_overwrite,
                )
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            if time_next < 0:
                # pdb.set_trace()
                if cond!=None:
                    img[:,:cond.shape[1],:]=cond
                img = x_start
                continue
            if self.conditioned_steps==0 and cond!=None:
                # pdb.set_trace()
                noise_cond=torch.randn_like(cond).to(device=device)
                img[:,:cond.shape[1],:]=self.q_sample(x_start = cond, t = time_cond, noise = noise_cond)#set first several time steps with groundtruth or named as condition


        plt.figure(figsize=(30,20))
        plt.plot(grad_mean_list,color="blue",label="grad_mean_list")
        # plt.plot(ss_list,color="red",label="ss_list")
        plt.title('grad_list')
        plt.grid(True)
        plt.savefig("grad_mean_list_p_sample.png")

        if self.conditioned_steps==0 and cond!=None:
            return img
        else:
            return img

    @torch.no_grad()
    def composing_time_sample(self, shape, cond, clip_denoised = True,n_composed=2):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        # pdb.set_trace()
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn((shape[0], shape[1], shape[2]), device = device)
        img_infered=torch.randn(((n_composed+1)*shape[0], shape[1], shape[2]), device = device)
        cond_infered=torch.randn(((n_composed+1)*cond.shape[0], cond.shape[1], cond.shape[2]), device = device)
        cond_infered[0:shape[0],:,:]=cond
        x_start_infered=torch.randn(((n_composed+1)*shape[0], shape[1], shape[2]), device = device)
        pred_noise_infered=torch.randn(((n_composed+1)*shape[0], shape[1], shape[2]), device = device)
        img_temp=torch.randn((shape[0], shape[1], shape[2]), device = device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch*(n_composed+1),), time, device=device, dtype=torch.long) #time_cond.shape [25,256]
            self_cond = x_start if self.self_condition else None

            for i_temp in range(n_composed): # keypoints: the last few time steps of the result of the previous inference as the condition of the subsequent time step inference
                cond_infered[(i_temp+1)*shape[0]:(i_temp+2)*shape[0],:,:]=\
                    img_infered[i_temp*shape[0]:(i_temp+1)*shape[0],-self.conditioned_steps:,:]

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            #get the initial denoised results
            noise = torch.randn_like(img_infered)
            pred_noise_infered,x_start_infered,*_ = self.model_predictions(img_infered, cond_infered, time_cond, self_cond, clip_x_start = clip_denoised)##every step cost 0.02ms

            if time_next < 0:
                img_infered = x_start_infered
                continue

            img_infered = x_start_infered * alpha_next.sqrt() + \
                  c * pred_noise_infered + \
                  sigma * noise

        img=img_infered[0:shape[0],:,:]
        img_infered_reshape=img_infered[shape[0]:2*shape[0],-20:,]
        if n_composed!=1:
            for k in range(n_composed-1):
                img_infered_reshape=torch.cat([img_infered_reshape,img_infered[(k+2)*shape[0]:(k+3)*shape[0],-20:,:]],dim=1)

        return img,img_infered_reshape
    
    @torch.no_grad()
    def gradient(self,x_t,t,n_bodies,scalar_for_gradient=None):
        '''
                x_t: [batch_size,conditioned_steps+rollout_steps,n_bodies*features] [1,24,16] n_bodies=(16/4)=4
                t :[n_bodies]
        '''
        ##just test for 4 bodies now
        # pdb.set_trace()
        batch_size=x_t.shape[0]
        if n_bodies==4:
            x_t=x_t.reshape((x_t.shape[0],x_t.shape[1],4,int(x_t.shape[2]/4))) #[1 24 4 4] [batch_szie conditioned_steps+rollout_steps n_bodies n_features] features include x,y,vx,vy
            x_t_body1=x_t[0:,:,0,:] #[batch_size 24 4],to get body1's data
            x_t_body2=x_t[0:,:,1,:]
            x_t_body3=x_t[0:,:,2,:]
            x_t_body4=x_t[0:,:,3,:]

            # x_t=x_t.reshape((x_t.shape[0],x_t.shape[1],2,int(x_t.shape[2]/2))) #[2 24 2 4] [batch_szie conditioned_steps+rollout_steps n_bodies n_features] features include x,y,vx,vy
            # x_t_body1=x_t[0:1,:,0,:] #[1 24 4],to get body1's data
            # x_t_body2=x_t[0:1,:,1,:]
            # x_t_body3=x_t[1:,:,0,:]
            # x_t_body4=x_t[1:,:,1,:]


            x_t_body1_2=torch.cat([x_t_body1,x_t_body2],dim=2) #[Batch_size 24 8] to get body1 and body2's data to get conditioned_noise
            x_t_body1_3=torch.cat([x_t_body1,x_t_body3],dim=2)
            x_t_body1_4=torch.cat([x_t_body1,x_t_body4],dim=2)
            x_t_body2_3=torch.cat([x_t_body2,x_t_body3],dim=2)
            x_t_body2_4=torch.cat([x_t_body2,x_t_body4],dim=2)
            x_t_body3_4=torch.cat([x_t_body3,x_t_body4],dim=2)

            x_t_input=torch.cat([x_t_body1_2,x_t_body1_3,x_t_body1_4,x_t_body2_3,x_t_body2_4,x_t_body3_4],dim=0) #【1,2  1,3  1,4   2,3   2,4   3,4】[6 24 8]
            time_cond = torch.full((x_t_input.shape[0],), t, device=x_t.device, dtype=torch.long)
            # if x_input_unconditioned==None:
            #     x_input_unconditioned=torch.randn_like(x_t_input)
            # pdb.set_trace()
            noise_conditioned=self.model(x_t_input,time_cond,cond=None)
            noise_conditioned=noise_conditioned.reshape((noise_conditioned.shape[0],noise_conditioned.shape[1],2,int(noise_conditioned.shape[2]/2)))
            # noise_unconditioned=self.model_unconditioned(x_t_unconditioned,time_cond,cond=None).reshape((noise_unconditioned.shape[0],noise_unconditioned.shape[1],2,noise_unconditioned.shape[2]/2))
            time_cond_for_uncond=torch.full((x_t_body1.shape[0],), t, device=x_t.device, dtype=torch.long)
            noise_body1_unconditioned=self.model_unconditioned(x_t_body1,time_cond_for_uncond,cond=None) #[1,24,4]
            noise_body2_unconditioned=self.model_unconditioned(x_t_body2,time_cond_for_uncond,cond=None)
            noise_body3_unconditioned=self.model_unconditioned(x_t_body3,time_cond_for_uncond,cond=None)
            noise_body4_unconditioned=self.model_unconditioned(x_t_body4,time_cond_for_uncond,cond=None)
            # to get noise_body1 conditioned 1,2 1,3 1,and plus noise_body1_unconditioned
            coefficient_unconditioned_grad=1.4
            noise_body1 = noise_conditioned[0:batch_size,:,0,:]+noise_conditioned[batch_size:batch_size*2,:,0,:]+noise_conditioned[2*batch_size:3*batch_size,:,0,:] -coefficient_unconditioned_grad*noise_body1_unconditioned  #[1,24,4] <----from Equation 18
            noise_body2 = noise_conditioned[0:batch_size,:,1,:]+noise_conditioned[3*batch_size:4*batch_size,:,0,:]+noise_conditioned[4*batch_size:5*batch_size,:,0,:]-coefficient_unconditioned_grad*noise_body2_unconditioned  
            noise_body3 = noise_conditioned[batch_size:batch_size*2,:,1,:]+noise_conditioned[batch_size*3:4*batch_size,:,1,:]+noise_conditioned[5*batch_size:,:,0,:]-coefficient_unconditioned_grad*noise_body3_unconditioned  
            noise_body4 = noise_conditioned[2*batch_size:3*batch_size,:,1,:]+noise_conditioned[4*batch_size:5*batch_size,:,1,:]+noise_conditioned[5*batch_size:,:,1,:]-coefficient_unconditioned_grad*noise_body4_unconditioned#[1,24,4]  just -2.9 一直减小

            # noise_body1=(noise_conditioned[0:20,:,0,:]+noise_conditioned[20:40,:,0,:]+noise_conditioned[40:60,:,0,:])/3  #[1,24,4] <----from Equation 18
            # noise_body2=(noise_conditioned[0:20,:,1,:]+noise_conditioned[60:80,:,0,:]+noise_conditioned[80:100,:,0,:])/3
            # noise_body3=(noise_conditioned[20:40,:,1,:]+noise_conditioned[60:80,:,1,:]+noise_conditioned[100:,:,0,:])/3
            # noise_body4=(noise_conditioned[40:60,:,1,:]+noise_conditioned[80:100,:,1,:]+noise_conditioned[100:,:,1,:])/3#[1,24,4]  just -2.9 一直减小
            # noise_body1=noise_conditioned[0:1,:,0,:]+noise_conditioned[1:2,:,0,:]+noise_conditioned[2:3,:,0,:]
            # noise_body2=noise_conditioned[0:1,:,1,:]+noise_conditioned[3:4,:,0,:]+noise_conditioned[4:5,:,0,:]
            # noise_body3=noise_conditioned[1:2,:,1,:]+noise_conditioned[3:4,:,1,:]+noise_conditioned[5:,:,0,:]
            # noise_body4=noise_conditioned[2:3,:,1,:]+noise_conditioned[4:5,:,1,:]+noise_conditioned[5:,:,1,:] #

            # to get noise_body1 conditioned 1,2 1,3 1,and plus noise_body1_unconditioned
            # noise_body1=noise_body1_unconditioned  #[1,24,4] <----from Equation 18
            # noise_body2=noise_body2_unconditioned  
            # noise_body3=noise_body3_unconditioned  
            # noise_body4=noise_body4_unconditioned#[1,24,4]  ##-0.8 --》-0.9
    
            # noise_body1_2_3_4=torch.cat([noise_body1,noise_body2,noise_body3,noise_body4],dim=2) #[1,24,16]
            noise_body1_2_3_4=torch.cat([torch.cat([noise_body1,noise_body2],dim=2),torch.cat([noise_body3,noise_body4],dim=2)],dim=2) #[batch_size,24,16]
            if t>400:
                return -1*scalar_for_gradient[t]*noise_body1_2_3_4
            else:
                return noise_body1_2_3_4
        elif n_bodies==3:
            # pdb.set_trace()
            x_t=x_t.reshape((x_t.shape[0],x_t.shape[1],3,int(x_t.shape[2]/3))) #[B 24 3 4] [batch_szie conditioned_steps+rollout_steps n_bodies n_features] features include x,y,vx,vy
            x_t_body1=x_t[0:,:,0,:] #[batch_size 24 4],to get body1's data
            x_t_body2=x_t[0:,:,1,:]
            x_t_body3=x_t[0:,:,2,:]

            # x_t=x_t.reshape((x_t.shape[0],x_t.shape[1],2,int(x_t.shape[2]/2))) #[2 24 2 4] [batch_szie conditioned_steps+rollout_steps n_bodies n_features] features include x,y,vx,vy
            # x_t_body1=x_t[0:1,:,0,:] #[1 24 4],to get body1's data
            # x_t_body2=x_t[0:1,:,1,:]
            # x_t_body3=x_t[1:,:,0,:]
            # x_t_body4=x_t[1:,:,1,:]


            x_t_body1_2=torch.cat([x_t_body1,x_t_body2],dim=2) #[Batch_size 24 8] to get body1 and body2's data to get conditioned_noise
            x_t_body1_3=torch.cat([x_t_body1,x_t_body3],dim=2)
            x_t_body2_3=torch.cat([x_t_body2,x_t_body3],dim=2)

            x_t_input=torch.cat([x_t_body1_2,x_t_body1_3,x_t_body2_3],dim=0) #【1,2  1,3  1,4   2,3   2,4   3,4】[3 24 8]
            time_cond = torch.full((x_t_input.shape[0],), t, device=x_t.device, dtype=torch.long)
            # if x_input_unconditioned==None:
            #     x_input_unconditioned=torch.randn_like(x_t_input)
            # pdb.set_trace()
            noise_conditioned=self.model(x_t_input,time_cond,cond=None)
            noise_conditioned=noise_conditioned.reshape((noise_conditioned.shape[0],noise_conditioned.shape[1],2,int(noise_conditioned.shape[2]/2))) #[B*3,24,2,4]
            # noise_unconditioned=self.model_unconditioned(x_t_unconditioned,time_cond,cond=None).reshape((noise_unconditioned.shape[0],noise_unconditioned.shape[1],2,noise_unconditioned.shape[2]/2))
            time_cond_for_uncond=torch.full((x_t_body1.shape[0],), t, device=x_t.device, dtype=torch.long)
            noise_body1_unconditioned=self.model_unconditioned(x_t_body1,time_cond_for_uncond,cond=None) #[1,24,4]
            noise_body2_unconditioned=self.model_unconditioned(x_t_body2,time_cond_for_uncond,cond=None)
            noise_body3_unconditioned=self.model_unconditioned(x_t_body3,time_cond_for_uncond,cond=None)
            # to get noise_body1 conditioned 1,2 1,3 1,and plus noise_body1_unconditioned
            noise_body1=noise_conditioned[0:20,:,0,:]+noise_conditioned[20:40,:,0,:] -noise_body1_unconditioned  #[1,24,4] <----from Equation 18
            noise_body2=noise_conditioned[0:20,:,1,:]+noise_conditioned[40:60,:,0,:]-noise_body2_unconditioned  
            noise_body3=noise_conditioned[20:40,:,1,:]+noise_conditioned[40:60,:,1,:]-noise_body3_unconditioned  

            # noise_body1=noise_conditioned[0:20,:,0,:]+noise_conditioned[20:40,:,0,:]+noise_conditioned[40:60,:,0,:]  #[1,24,4] <----from Equation 18
            # noise_body2=noise_conditioned[0:20,:,1,:]+noise_conditioned[60:80,:,0,:]+noise_conditioned[80:100,:,0,:]
            # noise_body3=noise_conditioned[20:40,:,1,:]+noise_conditioned[60:80,:,1,:]+noise_conditioned[100:,:,0,:]
            # noise_body4=noise_conditioned[40:60,:,1,:]+noise_conditioned[80:100,:,1,:]+noise_conditioned[100:,:,1,:]#[1,24,4]  just -2.9 
            # # noise_body1=noise_conditioned[0:1,:,0,:]+noise_conditioned[1:2,:,0,:]+noise_conditioned[2:3,:,0,:]
            # noise_body2=noise_conditioned[0:1,:,1,:]+noise_conditioned[3:4,:,0,:]+noise_conditioned[4:5,:,0,:]
            # noise_body3=noise_conditioned[1:2,:,1,:]+noise_conditioned[3:4,:,1,:]+noise_conditioned[5:,:,0,:]
            # noise_body4=noise_conditioned[2:3,:,1,:]+noise_conditioned[4:5,:,1,:]+noise_conditioned[5:,:,1,:] #？

            # to get noise_body1 conditioned 1,2 1,3 1,and plus noise_body1_unconditioned
            # noise_body1=noise_body1_unconditioned  #[1,24,4] <----from Equation 18
            # noise_body2=noise_body2_unconditioned  
            # noise_body3=noise_body3_unconditioned  
            # noise_body4=noise_body4_unconditioned#[1,24,4]  ##-0.8 --》-0.9
    
            # noise_body1_2_3_4=torch.cat([noise_body1,noise_body2,noise_body3,noise_body4],dim=2) #[1,24,16]
            noise_body1_2_3=torch.cat([noise_body1,noise_body2,noise_body3],dim=2) #[batch_size,24,12]
            if t>400:
                return -1*scalar_for_gradient[t]*noise_body1_2_3
            else:
                return noise_body1_2_3


    @torch.no_grad()
    def sample_compose_multibodies(self,cond,N,L,n_bodies):
        img = torch.randn((cond.shape[0], self.rollout_steps, cond.shape[2]), device = cond.device)
        x=torch.cat([cond,img],dim=1)
        times = torch.tensor(range(N)).int().tolist()
        # pdb.set_trace()
        Mass_matrix_sqrt=self.betas
        Mass_matrix=Mass_matrix_sqrt**2
        mass_diag_sqrt = torch.ones_like(x)
        mass_diag=mass_diag_sqrt**2
        step_size=self.betas*0.1

        # caculate gradient scalar
        alphas_cumprod_inference=torch.cumprod(1.-self.betas_inference, dim=0)
        scalar_for_gradient=torch.sqrt(1 / (1 - alphas_cumprod_inference)).to(x.device)
        for i in tqdm(times,"Sampleing loop"):
            i=N-i-1 #reverse
            if i>400:
                # v= torch.randn_like(x) * 1
                # grad=self.gradient(x,i,4)
                # for j in range(L):
                #     v += 0.5 * step_size[i] * grad#gradient_target(x_k)  # half step in v
                #     x += step_size[i] * v / mass_diag  # Step in x
                #     grad = self.gradient(x,i,4)
                #     grad_mean_list.append(grad.mean().to("cpu"))
                #     v += 0.5 * step_size[i] * grad  # half step in v

                # if i<=1:
                #     plt.figure(figsize=(30,20))
                #     plt.plot(grad_mean_list,color="blue",label="grad_mean_list")
                #     # plt.plot(ss_list,color="red",label="ss_list")
                #     plt.title('grad_list')
                #     plt.grid(True)
                #     plt.savefig("grad_mean_list_UHMC.png")

                # use ULA sampler
                ts=torch.tensor([i]*x.shape[0])
                x=self.sample_step_ULA(x,ts,L,n_bodies,N,scalar_for_gradient)
            else:
                # use the normal p_samle
                # x_t=x #[1 24 16]
                # x_t=x_t.reshape((x_t.shape[0],x_t.shape[1],4,int(x_t.shape[2]/4))) #[1 24 4 4]
                # #to get single body's data
                # x_t_body1=x_t[:,:,0,:] #[1 24 4]
                # x_t_body2=x_t[:,:,1,:]
                # x_t_body3=x_t[:,:,2,:]
                # x_t_body4=x_t[:,:,3,:]
                # x_t=torch.cat([torch.cat([x_t_body1,x_t_body2],dim=2),torch.cat([x_t_body3,x_t_body4],dim=2)],dim=0) #[2,24,8]
                x[:,self.conditioned_steps:,:],_=self.p_sample(x[:,self.conditioned_steps:,:],x[:,:self.conditioned_steps,:],i)
                # x=torch.cat([x_t[:1,:,:],x_t[1:,:,:]],dim=2)
                if i<=2:
                    plt.figure(figsize=(30,20))
                    plt.plot(grad_mean_list,color="blue",label="grad_mean_list")
                    # plt.plot(ss_list,color="red",label="ss_list")
                    plt.title('grad_list')
                    plt.grid(True)
                    plt.savefig("grad_mean_list_p_sample_test.png")
        return x[:,self.conditioned_steps:,:]




    @torch.no_grad()
    def sample_step_ULA(self, x,ts, num_samples_per_step,n_bodies,N,scalar_for_gradient): # ts=[t]*batch_size
        
        step_sizes=self.betas_inference*0.035
        for i in range(num_samples_per_step):
            # pdb.set_trace()
            ss = step_sizes[ts[0]]
            std = (2 * ss) ** .5
            grad = self.gradient(x,ts[0],n_bodies,scalar_for_gradient)
            noise = torch.randn_like(grad) * std
            grad_mean_list.append(grad.mean().to("cpu"))
            ss_list.append(ss.to("cpu"))
            x = x + grad * ss + noise
            # if ts[0]<=2:
            #     plt.figure(figsize=(30,20))
            #     plt.plot(grad_mean_list,color="blue",label="grad_mean_list")
            #     plt.title('grad_list')
            #     plt.grid(True)
            #     plt.ylim([-2,2])
            #     plt.savefig("grad_mean_list_ULA.png")

            #     plt.figure(figsize=(30,20))
            #     plt.plot(ss_list,color="red",label="ss_list")
            #     plt.title('stepsize_list')
            #     plt.grid(True)
            #     plt.savefig("stepsize_list.png")
        return x

        
    @torch.no_grad()
    def p_sample_UHMC(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_UHMC(
        self,
        sampler,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive_UHMC(
            sampler,
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final

    def p_sample_loop_progressive_UHMC(
        self,
        sampler,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps_UHMC))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        if sampler==None:
            for i in indices:
                t = torch.tensor([i] * shape[0], device=device)
                # with th.no_grad():
                out = self.p_sample_UHMC(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                out = out["sample"]
                yield out
                img = out
        else:
            for i in indices:
                t = torch.tensor([i] * shape[0], device=device)
                # with th.no_grad():
                out = self.p_sample_UHMC(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                out = out["sample"]
            
                if i > 50 :
                    out=sampler.sample_step(out, i,t, model_kwargs)

                yield out
                img = out

    @torch.no_grad()
    def autoregress_time_compose_sample(self,batch_size,cond,n_composed,is_single_step_prediction=False,prediction_steps=40):
        clip_denoised=True
        # pdb.set_trace()
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # if self.conditioned_steps==0 and cond.shape[1]!=0:
        #     pass
        # else:
        # pdb.set_trace()
        if is_single_step_prediction:
            img_composed = torch.randn((cond.shape[0], prediction_steps, cond.shape[2]), device = device)
            x_start = None
            if prediction_steps%self.conditioned_steps==0:
                num_prediction=prediction_steps//self.conditioned_steps
            else:
                 num_prediction=prediction_steps//self.conditioned_steps
                 num_prediction=num_prediction+1
           
            for i in range(num_prediction):
                if i!=0:
                    cond=img[:,-self.conditioned_steps:,:] ##update the condition with the last prediction for the next prediction
                img = torch.randn((cond.shape[0], self.rollout_steps, cond.shape[2]), device = device)
                x_start=None
                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                    time_cond = torch.full((cond.shape[0],), time, device=device, dtype=torch.long) #time_cond.shape [25,256]
                    self_cond = x_start if self.self_condition else None
                    if self.conditioned_steps==0 and cond.shape[1]!=0:
                        img[:,:cond.shape[1],:]=cond#set first several time steps with groundtruth or named as condition
                    pred_noise, x_start, *_ = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = clip_denoised)##every step cost 0.02ms


                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]

                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()

                    noise = torch.randn_like(img)

                    img = x_start * alpha_next.sqrt() + \
                        c * pred_noise + \
                        sigma * noise
                    if time_next < 0:
                        img = x_start
                        continue
                if self.conditioned_steps==0 and cond.shape[1]!=0:
                    img_composed[:,i*self.rollout_steps:(i+1)*self.rollout_steps,:] = img[:,cond.shape[1]:,:]
                else:
                    img_composed[:,i*self.rollout_steps:(i+1)*self.rollout_steps,:] = img
            return img_composed
        else:
            img_composed = torch.randn((cond.shape[0], (n_composed+1)*self.rollout_steps, cond.shape[2]), device = device)
            x_start = None
            for i in range(n_composed+1):
                if i!=0:
                    cond=img[:,-self.conditioned_steps:,:] ##update the condition with the last prediction for the next prediction
                img = torch.randn((cond.shape[0], self.rollout_steps, cond.shape[2]), device = device)
                x_start=None
                for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
                    time_cond = torch.full((cond.shape[0],), time, device=device, dtype=torch.long) #time_cond.shape [25,256]
                    self_cond = x_start if self.self_condition else None
                    if self.conditioned_steps==0 and cond.shape[1]!=0:
                        img[:,:cond.shape[1],:]=cond#set first several time steps with groundtruth or named as condition
                    pred_noise, x_start, *_ = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = clip_denoised)##every step cost 0.02ms


                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]

                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()

                    noise = torch.randn_like(img)

                    img = x_start * alpha_next.sqrt() + \
                        c * pred_noise + \
                        sigma * noise
                    if time_next < 0:
                        img = x_start
                        continue
                if self.conditioned_steps==0 and cond.shape[1]!=0:
                    img_composed[:,i*self.rollout_steps:(i+1)*self.rollout_steps,:] = img[:,cond.shape[1]:,:]
                else:
                    img_composed[:,i*self.rollout_steps:(i+1)*self.rollout_steps,:] = img
            return img_composed

    @torch.no_grad()
    def sample(
        self,
        batch_size=16,
        cond=None,
        is_composing_time=False,
        n_composed=2,
        compose_start_step=4,
        compose_n_bodies=2,
        compose_mode="mean",
        design_fn=None,
        design_guidance="standard",
        initial_state_overwrite=None,
        initialization_mode=0,
        initialization_img=None,
    ):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        if self.is_ddim_sampling:
            # sample_fn = self.ddim_sample if not is_composing_time else self.composing_time_sample
            sample_fn = self.ddim_sample
            return sample_fn((batch_size, image_size,channels),
                             cond=cond,
                             n_composed=n_composed,
                             compose_start_step=compose_start_step,
                             compose_n_bodies=compose_n_bodies,
                             compose_mode=compose_mode,
                             design_fn=design_fn,
                             design_guidance=design_guidance,
                             initial_state_overwrite=initial_state_overwrite,
                            initialization_mode=initialization_mode,
                            initialization_img=initialization_img,
                            )
        else:
            sample_fn = self.p_sample_loop
            return sample_fn((batch_size, image_size, channels),
                             cond=cond,
                             n_composed=n_composed,
                             compose_start_step=compose_start_step,
                             compose_n_bodies=compose_n_bodies,
                             compose_mode=compose_mode,
                             design_fn=design_fn,
                             design_guidance=design_guidance,
                             initial_state_overwrite=initial_state_overwrite,
                            initialization_mode=initialization_mode,
                            initialization_img=initialization_img,
                            )


    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None): #forward process
        # x_start,shape=[B,FLAGS.rollout_steps,num_bodies*nun_feactures],t=[B]
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'loss_type3':
            lossfunction=CustomLoss()
            return lossfunction
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    def get_loss_weight(self,batch_size,conditioned_steps,rollout_steps,nbody_mul_features,loss_weight_discouont):
        if self.conditioned_steps!=0:
            weight_cond=torch.ones((batch_size,conditioned_steps,nbody_mul_features)).float()
            weight_steps=torch.ones((rollout_steps)).float()
            for i in range(0,rollout_steps):
                weight_steps[i]=weight_steps[i]*math.pow(loss_weight_discouont,i+1)
            copied=torch.stack([weight_steps]*nbody_mul_features, dim=0)
            copied2=torch.stack([copied]*batch_size,dim=0)
            weight_rollout=torch.transpose(copied2,1,2)
            return torch.cat([weight_cond,weight_rollout],dim=1)
        else:
            weight_steps=torch.ones((rollout_steps)).float()
            for i in range(0,rollout_steps):
                weight_steps[i]=weight_steps[i]*math.pow(loss_weight_discouont,i+1)
            copied=torch.stack([weight_steps]*nbody_mul_features, dim=0)
            copied2=torch.stack([copied]*batch_size,dim=0)
            weight_rollout=torch.transpose(copied2,1,2)
            return weight_rollout

    def p_losses(self, x_start, t, cond, noise = None):
        """
        Args:
            x_start: shape [B, C, N]
            t: shape [B]
            cond: shape [B,4,num_bodies*nun_feactures]
            noise: [B,FLAGS.rollout_steps,num_bodies*nun_feactures]
        """
        # pdb.set_trace()
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start)) #[32, FLAGS.rollout_steps, num_bodies*nun_feactures]
        if self.conditioned_steps!=0:
            noise_cond=torch.zeros_like(cond)

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        if self.conditioned_steps!=0:
            x = torch.cat([cond, x], dim=1)

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond) #x_self_cond：NONE

        # model_out = model_out[:, cond.size(1):]

        if self.objective == 'pred_noise':
            if self.conditioned_steps!=0:
                target=torch.cat([noise_cond,noise],dim=1)
            else:
                target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        if self.loss_type=="loss_type3":
            loss = self.loss_fn(model_out, target)
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none')
            loss_weight=self.get_loss_weight(x.shape[0],self.conditioned_steps,x.shape[1]-self.conditioned_steps,x.shape[2],self.loss_weight_discount).to(loss.device)
            # print("loss_weight",loss.shape)
            loss=torch.mul(loss,loss_weight)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
            # loss = loss * extract(loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, cond, *args, **kwargs):
        b, n, c,device, image_size, = *img.shape, img.device, self.image_size
        #img.shape [32, FLAGS.rollout_steps, num_bodies*nun_feactures]
        assert n == image_size, f'seq length must be {image_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(img, t, cond, *args, **kwargs)


class Trainer1D(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        conditioned_steps = 4,
        rollout_steps = 16,
        time_interval = 4,
        method_type=None,
        forward_model=None,
        train_dataset=None,
        dataset_path=None
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp
        self.dataset = dataset
        self.dataset_path=dataset_path
        # model
        self.method_type=method_type
        if self.method_type=="forward_model" or self.method_type=="Unet_rollout_one":
            self.model = forward_model
            self.model.to("cuda:0").float()
        elif self.method_type=="GNS" or self.method_type=="GNS_cond_one":
            self.model = forward_model
            self.model.to("cuda:0").float()
            self.metadata=train_dataset.metadata
            for key, value in self.metadata.items():
                self.metadata[key] = value.to('cuda')
        else:
            self.model = diffusion_model
            self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.conditioned_steps = conditioned_steps
        self.rollout_steps = rollout_steps
        self.time_interval = time_interval

        # dataset and dataloader

        if dataset.startswith("nbody"):
            self.ds = NBodyDataset(
                dataset=dataset,
                input_steps=conditioned_steps,
                output_steps=rollout_steps,
                time_interval=time_interval,
                is_y_diff=False,
                is_train=True,
                is_testdata=False,
                dataset_path=self.dataset_path
            )
        else:
            # TODO Add other datasets
            assert False
        s=CustomSampler(data=self.ds,batch_size=train_batch_size,noncollision_hold_probability=0,distance_threshold=40.5)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 6,sampler=s)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 6)
        # pdb.set_trace()
        # for data in dl:
        #     i_tem=i_tem+1
        # pdb.set_trace()
        # if self.method_type!="forward_model":
        if self.method_type=="GNS" or self.method_type=="GNS_cond_one" :
            self.ds=train_dataset
            dl = DataLoader(self.ds, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 6)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
            
        # optimizer

        self.opt = torch.optim.Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=40000, gamma=0.5)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        # if self.method_type!="forward_model":
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')

        mu = torch.mean(features, dim = 0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.channels == 1:
            real_samples, fake_samples = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch], (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):
        loss_list=torch.zeros((self.train_num_steps)).flatten()
        accelerator = self.accelerator
        device = accelerator.device

        ##just use the same test data to evaluate the model so that We can analyze it more clearly.
        n_bodies = eval(self.dataset.split("-")[1])
        dataset_test = NBodyDataset(
        dataset=f"nbody-{n_bodies}",
        input_steps=self.conditioned_steps,
        output_steps=self.rollout_steps,
        time_interval=self.time_interval,
        is_y_diff=False,
        is_train=False,
        is_testdata=False,
        dataset_path=self.dataset_path
        )
        dataloader = DataLoader(dataset_test, batch_size=1000, shuffle=False, pin_memory=True, num_workers=6)
        for data_test in dataloader:
            break
        data_test.to(self.device)

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            
            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    if self.method_type=="GNS" or self.method_type=="GNS_cond_one" :
                        data = next(self.dl)
                    else:
                        data = next(self.dl).to(device)
                    

                    if self.dataset == "chaotic_ellipse":
                        p.print("1", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
                        data = data['x'].reshape(-1, 126, 126, 4, 2) / 100.

                        data_pad = torch.zeros(data.shape[0], 8, 128, 128).to(data.device)
                        data = torch.flatten(data, -2, -1).permute(0, 3, 1, 2)
                        data_pad[:, :, 1:-1, 1:-1] = data
                        data = data_pad
                        p.print("2", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
                    elif self.dataset.startswith("nbody"):
                        if self.method_type=="GNS":
                            poss, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss = data
                            #only support Batch_size=1
                            poss=poss[0]
                            tgt_accs=tgt_accs[0]
                            tgt_vels=tgt_vels[0]
                            particle_type=particle_type[0]
                            nonk_mask=nonk_mask[0]
                            tgt_poss=tgt_poss[0]
                        elif self.method_type=="GNS_cond_one":
                            # pdb.set_trace()
                            poss, vel,tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss = data
                            #only support Batch_size=1
                            poss=poss[0]
                            vel=vel[0]
                            tgt_accs=tgt_accs[0]
                            tgt_vels=tgt_vels[0]
                            particle_type=particle_type[0]
                            nonk_mask=nonk_mask[0]
                            tgt_poss=tgt_poss[0]
                        else:
                            if self.conditioned_steps!=0:
                                cond = get_item_1d(data, "x")  # [B, conditioned_steps, n_bodies*feature_size]
                                data = get_item_1d(data, "y")  # [B, rollout_steps, n_bodies*feature_size]
                            else:
                                cond = None  # [B, conditioned_steps, n_bodies*feature_size]
                                data = get_item_1d(data, "y")  # [B, rollout_steps, n_bodies*feature_size]
                    else:
                        raise

                    with self.accelerator.autocast():
                        if self.method_type=="forward_model":
                            # pdb.set_trace()
                            # pdb.set_trace()
                            cond.cuda()
                            data=torch.cat([cond,data],dim=1)
                            pred = self.model(cond) #costs 0.08s cond [B,timesteps,8]
                            loss=F.l1_loss(pred,data,reduction = 'none')
                            loss = reduce(loss, 'b ... -> b (...)', 'mean')
                            loss=loss.mean()
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        elif self.method_type=="Unet_rollout_one":
                            # pdb.set_trace()
                            cond.cuda()
                            pred=torch.cat([cond.clone()]*self.rollout_steps,dim=1)
                            data=torch.cat([cond,data],dim=1)
                            for i in range(pred.shape[1]):
                                if i==0:
                                    pred[:,i:i+1]=self.model(cond)[:,-1:]
                                else:
                                    pred[:,i:i+1]=self.model(pred[:,i-1:i])[:,-1:]
                            pred=torch.cat([cond,pred],dim=1)
                            loss=F.l1_loss(pred,data,reduction = 'none')
                            loss = reduce(loss, 'b ... -> b (...)', 'mean')
                            loss=loss.mean()
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        elif self.method_type=="GNS_cond_one":
                            # pdb.set_trace()
                            num_rollouts = tgt_vels.shape[1]
                            # pdb.set_trace()
                            outputs = self.model(poss, vel,particle_type, self.metadata, nonk_mask, tgt_poss, num_rollouts=num_rollouts, phase='train')
                            labels = {
                                'accns': tgt_accs,
                                'poss': tgt_poss
                            }
                            # if tgt_accs.shape[1]!=20 or tgt_poss.shape[1]!=20:
                            #     pdb.set_trace()
                            # pdb.set_trace()
                            loss = F.l1_loss(outputs["pred_accns"], labels["accns"],reduction = 'none')+F.l1_loss(outputs["pred_poss"], labels["poss"],reduction = 'none')
                            loss = reduce(loss, 'b ... -> b (...)', 'mean')
                            loss=loss.mean()
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        elif self.method_type=="GNS":
                            # pdb.set_trace()
                            num_rollouts = tgt_vels.shape[1]
                            # pdb.set_trace()
                            outputs = self.model(poss, particle_type, self.metadata, nonk_mask, tgt_poss, num_rollouts=num_rollouts, phase='train')
                            labels = {
                                'accns': tgt_accs,
                                'poss': tgt_poss
                            }
                            # if tgt_accs.shape[1]!=20 or tgt_poss.shape[1]!=20:
                            #     pdb.set_trace()
                            # pdb.set_trace()
                            loss = F.l1_loss(outputs["pred_accns"], labels["accns"],reduction = 'none')+F.l1_loss(outputs["pred_poss"], labels["poss"],reduction = 'none')
                            loss = reduce(loss, 'b ... -> b (...)', 'mean')
                            loss=loss.mean()
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        else:
                            loss = self.model(data, cond) #costs 0.08s
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                    self.accelerator.backward(loss)
                loss_list[self.step]=total_loss
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'total_loss: {total_loss:.6f}'+f'  loss:{loss:.6f}')
                accelerator.wait_for_everyone()
                # if self.step%20000 !=0:
                #     self.opt.step()
                #     self.opt.zero_grad()
                # else:
                #     self.opt.step()
                #     self.opt.zero_grad()
                #     self.scheduler.step()
                # if self.step%20000 !=0:
                #     self.opt.step()
                #     self.opt.zero_grad()
                # else:
                self.opt.step()
                self.opt.zero_grad()
                if self.step>600000:
                    self.scheduler.step()


                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    # if True:
                        self.ema.ema_model.eval()

                        milestone = self.step // self.save_and_sample_every
                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                        #     if self.method_type=="forward_model":
                        #         if self.conditioned_steps!=0:
                        #             all_images_list = list(map(lambda n: self.ema.ema_model(cond=cond[:n]), batches))
                        #         else:
                        #             all_images_list = list(map(lambda n: self.ema.ema_model(cond=cond), batches))
                        #     else:
                        #         if self.conditioned_steps!=0:
                        #             all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, cond=cond[:n]), batches))
                        #         else:
                        #             all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n, cond=cond), batches))

                        # all_images = torch.cat(all_images_list, dim = 0)
                        # gt_images = data[:self.num_samples]

                        if self.dataset == "chaotic_ellipse":
                            channels = all_images.shape[1]
                            timestep = channels // 2

                            all_images = all_images.permute(0, 2, 3, 1)
                            s = all_images.size()

                            all_images = all_images.view(*s[:3], timestep, 2).permute(0, 3, 1, 2, 4)
                            all_images = torch.cat([all_images, torch.zeros_like(all_images)[..., :1]], dim=-1)

                            s = all_images.size()
                            panel_im = all_images.permute(0, 2, 1, 3, 4).reshape(s[0]*s[2], s[1]*s[3], s[4])
                            panel_im = panel_im.detach().cpu().numpy()

                            write_path = str(self.results_folder / f'sample-{milestone}.png')
                            imwrite(write_path, panel_im)

                            gt_images = gt_images.permute(0, 2, 3, 1)
                            s = gt_images.size()

                            gt_images = gt_images.view(*s[:3], timestep, 2).permute(0, 3, 1, 2, 4)
                            gt_images = torch.cat([gt_images, torch.zeros_like(gt_images)[..., :1]], dim=-1)

                            s = gt_images.size()
                            panel_im = gt_images.permute(0, 2, 1, 3, 4).reshape(s[0]*s[2], s[1]*s[3], s[4])
                            panel_im = panel_im.detach().cpu().numpy()

                            write_path = str(self.results_folder / f'gt-{milestone}.png')
                            imwrite(write_path, panel_im)
                        elif self.dataset.startswith("nbody") and self.method_type!="GNS" and self.method_type!="GNS_cond_one" :
                            # pdb.set_trace()
                            img_test = get_item_1d(data_test, "y").to(self.device)
                            if self.conditioned_steps!=0:
                                cond_test = get_item_1d(data_test, "x").to(self.device)  # [B, conditioned_steps, n_bodies*feature_size]
                            else:
                                cond_test=cond_test=img_test[:,:self.conditioned_steps,:]
                              # [B, rollout_steps, n_bodies*feature_size]
                            if self.method_type == "forward_model":
                                pred=self.model(cond_test)
                                pred=pred[:,self.conditioned_steps:,:]
                            elif self.method_type == "Unet_rollout_one":
                                pred=torch.cat([cond_test.clone()]*self.rollout_steps,dim=1)
                                for i in range(pred.shape[1]):
                                    if i==0:
                                        pred[:,i:i+1]=self.model(cond_test)[:,-1:]
                                    else:
                                        pred[:,i:i+1]=self.model(pred[:,i-1:i])[:,-1:]
                            else:
                                pred = self.model.sample(
                                    batch_size=1000,
                                    cond=cond_test,  # [B, conditioned_steps, n_bodies*feature_size]
                                )# [B, rollout_steps, n_bodies*feature_size]
                            img_test_split = img_test.view(img_test.shape[0],img_test.shape[1],n_bodies,img_test.shape[2]//n_bodies)
                            pred_split = pred.view(pred.shape[0],pred.shape[1],n_bodies,pred.shape[2]//n_bodies)
                            loss_mean = (pred_split[:,:,:,0:4] - img_test_split[:,:,:,0:4]).abs().mean().item() #when testing,we just analyze the accuracy of x,y,vx,y
                            pdf = matplotlib.backends.backend_pdf.PdfPages(self.results_folder/f"figure_milestone_{milestone}_steps_{self.step}.pdf")
                            fontsize = 16
                            for i in range(40):
                                i=i*15 #test batch_size pieces of data,but plot it every 15 data
                                fig = plt.figure(figsize=(18,15))
                                if self.conditioned_steps!=0:
                                    cond_reshape = cond_test.reshape(cond_test.shape[0], self.conditioned_steps, n_bodies, cond_test.shape[2]//n_bodies).to('cpu')
                                pred_reshape = pred.reshape(pred.shape[0], self.rollout_steps, n_bodies,pred.shape[2]//n_bodies).to('cpu').detach().numpy()
                                y_gt_reshape = img_test.reshape(img_test.shape[0], self.rollout_steps, n_bodies, img_test.shape[2]//n_bodies).to('cpu')
                                for j in range(n_bodies):
                                    # cond:
                                    if self.conditioned_steps!=0:
                                        marker_size_cond = np.linspace(1, 2, self.conditioned_steps) * 100
                                        plt.plot(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="--")
                                        plt.scatter(cond_reshape[i,:,j,0], cond_reshape[i,:,j,1], color=COLOR_LIST[j], marker="+", linestyle="--", s=marker_size_cond)
                                    # y_gt:
                                    marker_size_y_gt = np.linspace(2, 3, self.rollout_steps) * 100
                                    plt.plot(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-.")
                                    plt.scatter(y_gt_reshape[i,:,j,0], y_gt_reshape[i,:,j,1], color=COLOR_LIST[j], marker=".", linestyle="-.", s=marker_size_y_gt)
                                    # pred:
                                    marker_size_pred = np.linspace(2, 3, self.rollout_steps) * 100
                                    plt.plot(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], linestyle="-")
                                    plt.scatter(pred_reshape[i,:,j,0], pred_reshape[i,:,j,1], color=COLOR_LIST[j], marker="v", linestyle="-", s=marker_size_y_gt)
                                    plt.xlim([0,1])
                                    plt.ylim([0,1])
                                loss_item = (pred[i] - img_test[i]).abs().mean().item()
                                plt.title(f"loss_mean: {loss_mean:.6f}   loss_item: {loss_item:.6f}", fontsize=fontsize)
                                plt.tick_params(labelsize=fontsize)
                                pdf.savefig(fig)
                                # if is_jupyter:
                                #     plt.show()
                                i=i/15
                            pdf.close()



                        self.save(milestone)
                        numpy_data = loss_list.numpy()
                        # draw loss_list
                        plt.figure()
                        x=np.linspace(0,len(numpy_data),len(numpy_data))
                        plt.plot(x, numpy_data)
                        plt.xlabel('X')
                        plt.ylabel('Y')
                        plt.title('loss_list')
                        plt.grid(True)
                        plt.savefig(self.results_folder/"loss_list.png")
                        # plt.show()
                        np.save(self.results_folder/'loss_lis.npy', numpy_data)
                pbar.update(1)      
        accelerator.print('training complete')