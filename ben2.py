import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.layers  import DropPath, to_2tuple, trunc_normal_
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
import random
import cv2
import os 
import subprocess
import time
import tempfile
import ffmpeg




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed(9)


torch.set_float32_matmul_precision('highest')



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def forward(self, x):

        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed) # B Wh*Ww C

        outs = [x.contiguous()]
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)


        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)


            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)



        return tuple(outs)








def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "gelu":
        return F.gelu

    raise RuntimeError(F"activation should be gelu, not {activation}.")


def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.InstanceNorm2d(out_dim), nn.GELU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.InstanceNorm2d(out_dim), nn.GELU())


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2 )
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x



class PositionEmbeddingSine:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32)

    def __call__(self, b, h, w):
        device = self.dim_t.device
        mask = torch.zeros([b, h, w], dtype=torch.bool, device=device)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.temperature ** (2 * (self.dim_t.to(device) // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)



class PositionEmbeddingSine:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32)

    def __call__(self, b, h, w):
        device = self.dim_t.device
        mask = torch.zeros([b, h, w], dtype=torch.bool, device=device)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.temperature ** (2 * (self.dim_t.to(device) // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MCLM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8]):
        super(MCLM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('gelu')
        self.pool_ratios = pool_ratios
        self.p_poses = []
        self.g_pos = None
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        self.p_poses = []
        self.g_pos = None 
        b, c, h, w = l.size()
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = rearrange(l, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)

        pools = []
        for pool_ratio in self.pool_ratios:
             # b,c,h,w
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            if self.g_pos is None:
                pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3])
                pos_emb = rearrange(pos_emb, 'b c h w -> (h w) b c')
                self.p_poses.append(pos_emb)
        pools = torch.cat(pools, 0)
        if self.g_pos is None:
            self.p_poses = torch.cat(self.p_poses, dim=0)
            pos_emb = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
            self.g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')

        device = pools.device
        self.p_poses = self.p_poses.to(device)
        self.g_pos = self.g_pos.to(device)


        # attention between glb (q) & multisensory concated-locs (k,v)
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')


        g_hw_b_c = g_hw_b_c + self.dropout1(self.attention[0](g_hw_b_c + self.g_pos, pools + self.p_poses, pools)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        # attention between origin locs (q) & freashed glb (k,v)
        l_hw_b_c = rearrange(l, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c, "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng=2, nw=2)
        outputs_re = []
        for i, (_l, _g) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g, _g)[0])  # (h w) 1 c
        outputs_re = torch.cat(outputs_re, 1)  # (h w) 4 c

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # hw,b(5),c
        return rearrange(l, "(h w) b c -> b c h w", h=h, w=w)  ## (5,c,h*w)









class MCRM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(MCRM, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('gelu')
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios

    def forward(self, x):
        device = x.device
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w

        patched_glb = rearrange(glb, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)

        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map, size=patches2image(loc).shape[-2:], mode='nearest')
        loc = loc * rearrange(token_attention_map, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)

        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(rearrange(pool, 'nl c h w -> nl c (h w)'))  # nl(4),c,hw

        pools = rearrange(torch.cat(pools, 2), "nl c nphw -> nl nphw 1 c")
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')

        outputs = []
        for i, q in enumerate(loc_.unbind(dim=0)):  # traverse all local patches
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])

        outputs = torch.cat(outputs, 1)
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)
        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src), size=glb.shape[-2:], mode='nearest')  # freshed glb

        return torch.cat((src, glb), 0), token_attention_map



class BEN_Base(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12)
        emb_dim = 128
        self.sideout5 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout3 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8])
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8])
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8])

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            nn.InstanceNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.InstanceNorm2d(384),
            nn.GELU(),
            nn.Conv2d(384, emb_dim, kernel_size=3, padding=1)
        )

        self.shallow = nn.Sequential(nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.GELU) or isinstance(m, nn.Dropout):
                m.inplace = True

    

    @torch.inference_mode()
    @torch.autocast(device_type="cuda",dtype=torch.float16)
    def forward(self, x):
        real_batch = x.size(0)

        shallow_batch = self.shallow(x)
        glb_batch = rescale_to(x, scale_factor=0.5, interpolation='bilinear')



        final_input = None
        for i in range(real_batch):
            start = i * 4
            end   = (i + 1) * 4
            loc_batch = image2patches(x[i,:,:,:].unsqueeze(dim=0))
            input_ = torch.cat((loc_batch, glb_batch[i,:,:,:].unsqueeze(dim=0)), dim=0)  
            
            
            if final_input == None:
                final_input= input_
            else: final_input = torch.cat((final_input, input_), dim=0)

        features = self.backbone(final_input)
        outputs = []
        
        for i in range(real_batch):

            start = i * 5
            end   = (i + 1) * 5
            
            f4 = features[4][start:end, :, :, :]  # shape: [5, C, H, W]
            f3 = features[3][start:end, :, :, :]
            f2 = features[2][start:end, :, :, :]
            f1 = features[1][start:end, :, :, :]
            f0 = features[0][start:end, :, :, :]
            e5 = self.output5(f4)
            e4 = self.output4(f3)
            e3 = self.output3(f2)
            e2 = self.output2(f1)
            e1 = self.output1(f0)
            loc_e5, glb_e5 = e5.split([4, 1], dim=0)
            e5 = self.multifieldcrossatt(loc_e5, glb_e5)  # (4,128,16,16)


            e4, tokenattmap4 = self.dec_blk4(e4 + resize_as(e5, e4)) 
            e4 = self.conv4(e4) 
            e3, tokenattmap3 = self.dec_blk3(e3 + resize_as(e4, e3))
            e3 = self.conv3(e3)
            e2, tokenattmap2 = self.dec_blk2(e2 + resize_as(e3, e2))
            e2 = self.conv2(e2)
            e1, tokenattmap1 = self.dec_blk1(e1 + resize_as(e2, e1))
            e1 = self.conv1(e1)

            loc_e1, glb_e1 = e1.split([4, 1], dim=0)

            output1_cat = patches2image(loc_e1)  # (1,128,256,256)

            # add glb feat in
            output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
            # merge
            final_output = self.insmask_head(output1_cat)  # (1,128,256,256)
            # shallow feature merge
            shallow = shallow_batch[i,:,:,:].unsqueeze(dim=0)
            final_output = final_output + resize_as(shallow, final_output)
            final_output = self.upsample1(rescale_to(final_output))
            final_output = rescale_to(final_output + resize_as(shallow, final_output))
            final_output = self.upsample2(final_output)
            final_output = self.output(final_output)
            mask = final_output.sigmoid()
            outputs.append(mask)

        return torch.cat(outputs, dim=0)




    def loadcheckpoints(self,model_path):
        model_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.load_state_dict(model_dict['model_state_dict'], strict=True)
        del model_path

    def inference(self,image,refine_foreground=False):
        
        set_random_seed(9)
        # image = ImageOps.exif_transpose(image)
        if isinstance(image, Image.Image):            
            image, h, w,original_image =  rgb_loader_refiner(image)
            if torch.cuda.is_available():

                img_tensor = img_transform(image).unsqueeze(0).to(next(self.parameters()).device)
            else:
                img_tensor = img_transform32(image).unsqueeze(0).to(next(self.parameters()).device)

            
            with torch.no_grad():
                res = self.forward(img_tensor)

            # Show Results
            if refine_foreground == True:

                pred_pil = transforms.ToPILImage()(res.squeeze())
                image_masked = refine_foreground_process(original_image, pred_pil)
                
                image_masked.putalpha(pred_pil.resize(original_image.size))
                return image_masked

            else:
                alpha = postprocess_image(res, im_size=[w,h])
                pred_pil = transforms.ToPILImage()(alpha)
                mask = pred_pil.resize(original_image.size)
                original_image.putalpha(mask)
                # mask = Image.fromarray(alpha)

                return original_image


        else:
            # 实现真正的批处理
            batch_size = len(image)
            device = next(self.parameters()).device
            
            # 预处理所有图像
            preprocessed = []
            original_images = []
            sizes = []
            
            for img in image:
                img_p, h, w, original = rgb_loader_refiner(img)
                preprocessed.append(img_p)
                original_images.append(original)
                sizes.append((w, h))
            
            # 一次性转换整个批次
            if torch.cuda.is_available():
                batch_tensors = torch.stack([img_transform(img) for img in preprocessed]).to(device)
            else:
                batch_tensors = torch.stack([img_transform32(img) for img in preprocessed]).to(device)
            
            # 一次性推理整个批次
            with torch.no_grad():
                batch_results = self.forward(batch_tensors)
            
            # 处理每个结果
            foregrounds = []
            for i in range(batch_size):
                res = batch_results[i:i+1]  # 保持维度兼容性
                w, h = sizes[i]
                original_image = original_images[i]
                
                if refine_foreground:
                    pred_pil = transforms.ToPILImage()(res.squeeze())
                    image_masked = refine_foreground_process(original_image, pred_pil)
                    image_masked.putalpha(pred_pil.resize(original_image.size))
                    foregrounds.append(image_masked)
                else:
                    alpha = postprocess_image(res, im_size=[w, h])
                    pred_pil = transforms.ToPILImage()(alpha)
                    mask = pred_pil.resize(original_image.size)
                    original_image.putalpha(mask)
                    foregrounds.append(original_image)
            
            return foregrounds




    def segment_video(self, video_path, output_path="./", fps=0, refine_foreground=False, batch=1, print_frames_processed=True, webm=False, rgb_value=(0, 255, 0), output_mask=False, progress_callback=None):
    
        """
        Segments the given video to extract the foreground (with alpha) from each frame
        and saves the result as either a WebM video (with alpha channel) or MP4 (with a
        color background).

        Args:
            video_path (str):
                Path to the input video file.

            output_path (str, optional):
                Directory (or full path) where the output video and/or files will be saved.
                Defaults to "./".

            fps (int, optional):
                The frames per second (FPS) to use for the output video. If 0 (default), the
                original FPS of the input video is used. Otherwise, overrides it.

            refine_foreground (bool, optional):
                Whether to run an additional "refine foreground" process on each frame. 
                Defaults to False.

            batch (int, optional):
                Number of frames to process at once (inference batch size). Large batch sizes
                may require more GPU memory. Defaults to 1.

            print_frames_processed (bool, optional):
                If True (default), prints progress (how many frames have been processed) to 
                the console.

            webm (bool, optional):
                If True, exports a WebM video with alpha channel (VP9 / yuva420p).
                If False, exports an MP4 video composited over a solid color background.

            rgb_value (tuple, optional):
                The RGB background color (e.g., green screen) used to composite frames when
                saving to MP4. Defaults to (0, 255, 0).
                
            output_mask (bool, optional):
                If True, also outputs a separate grayscale video of the alpha masks.
                These masks can be used for custom compositing with any background.
                Defaults to False.
                
            progress_callback (callable, optional):
                A callback function to report progress. Should accept two parameters:
                current frame and total frames. Defaults to None.

        Returns:
            None. Writes the output video(s) to disk in the specified format.
        """

        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_fps = 30 if original_fps == 0 else original_fps
        fps = original_fps if fps == 0 else fps

        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("No frames found in the video.")
        height, width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        foregrounds = []
        masks = []  # Store masks separately if needed
        frame_idx = 0
        processed_count = 0
        batch_frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initial progress report
        if progress_callback:
            progress_callback(0, total_frames)

        while True:
            ret, frame = cap.read()
            if not ret:
                if batch_frames:
                    batch_results = self.inference(batch_frames, refine_foreground)
                    if isinstance(batch_results, Image.Image):
                        foregrounds.append(batch_results)
                    else:
                        foregrounds.extend(batch_results)
                    if print_frames_processed:
                        print(f"Processed frames {frame_idx-len(batch_frames)+1} to {frame_idx} of {total_frames}")
                    if progress_callback:
                        progress_callback(frame_idx, total_frames)
                break

            # Process every frame instead of using intervals
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            batch_frames.append(pil_frame)
            
            if len(batch_frames) == batch:
                batch_results = self.inference(batch_frames, refine_foreground)
                if isinstance(batch_results, Image.Image):
                    foregrounds.append(batch_results)
                else:
                    foregrounds.extend(batch_results)
                    
                    # Extract masks from the foreground images if needed
                    if output_mask:
                        for fg in batch_results:
                            # Extract alpha channel as mask
                            if fg.mode == 'RGBA':
                                mask = fg.split()[3]  # Alpha channel
                                masks.append(mask)
                            
                if print_frames_processed:
                    print(f"Processed frames {frame_idx-batch+1} to {frame_idx} of {total_frames}")
                if progress_callback:
                    # Call progress_callback after each batch with current frame number and total frames
                    progress_callback(frame_idx, total_frames)
                batch_frames = []
                processed_count += batch

            frame_idx += 1

        # Call progress callback one last time to indicate completion
        if progress_callback:
            progress_callback(total_frames, total_frames)

        # Video export phase
        if progress_callback:
            # Signal starting export phase
            progress_callback(total_frames, total_frames, "Exporting video")

        if webm:
            alpha_webm_path = os.path.join(output_path, "foreground.webm")
            pil_images_to_webm_alpha(foregrounds, alpha_webm_path, fps=original_fps)

        else:
            cap.release()
            fg_output = os.path.join(output_path, 'foreground.mp4')
            
            pil_images_to_mp4(foregrounds, fg_output, fps=original_fps, rgb_value=rgb_value)
            
            try:
                fg_audio_output = os.path.join(output_path, 'foreground_output_with_audio.mp4')
                add_audio_to_video(fg_output, video_path, fg_audio_output)
            except Exception as e:
                print("No audio found in the original video")
                print(e)
        
        # Save mask video if requested
        if output_mask and masks:
            mask_output = os.path.join(output_path, 'mask.mp4')
            # Convert grayscale PIL images to RGB for MP4 output
            rgb_masks = [Image.merge('RGB', (m, m, m)) for m in masks]
            pil_images_to_mp4(rgb_masks, mask_output, fps=original_fps)
            
            try:
                mask_audio_output = os.path.join(output_path, 'mask_output_with_audio.mp4')
                add_audio_to_video(mask_output, video_path, mask_audio_output)
            except Exception as e:
                print("No audio found in the original video")
                print(e)
                
        cv2.destroyAllWindows()


    def segment_video_v2(self, video_path, output_mask_path=None, output_composite_path=None, fps=0, refine_foreground=True, batch=4, rgb_value=(0, 255, 0), progress_callback=None):
        """
        优化后的视频分割函数，支持真正的批处理
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0:
                fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 获取视频尺寸
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 准备结果容器
            foregrounds = []
            masks = []
            
            # 批处理缓冲区
            batch_frames = []
            frame_idxs = []  # 记录帧索引，用于保持顺序
            
            # 处理计数
            frame_idx = 0
            processed_count = 0
            
            # 预分配，提高效率
            if total_frames > 0:
                # 预分配空间给结果，避免频繁重新分配内存
                foregrounds = [None] * total_frames
                if output_mask_path:
                    masks = [None] * total_frames
            
            # 主处理循环
            while True:
                ret, frame = cap.read()
                if not ret:
                    # 处理最后一批（可能不满batch_size）
                    if batch_frames:
                        batch_results = self.inference(batch_frames, refine_foreground)
                        
                        # 保持顺序放入结果
                        for i, result in enumerate(batch_results):
                            idx = frame_idxs[i]
                            foregrounds[idx] = result
                            
                            # 提取掩码
                            if output_mask_path and result.mode == 'RGBA':
                                masks[idx] = result.split()[3]  # Alpha channel
                        
                        if progress_callback:
                            progress_callback(frame_idx, total_frames)
                    break
                
                # 处理读取的帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # 添加到批处理队列
                batch_frames.append(pil_frame)
                frame_idxs.append(frame_idx)
                
                # 当积累足够的帧时进行批处理
                if len(batch_frames) >= batch:
                    # 批量推理，提高效率
                    batch_results = self.inference(batch_frames, refine_foreground)
                    
                    # 保持顺序放入结果
                    for i, result in enumerate(batch_results):
                        idx = frame_idxs[i]
                        foregrounds[idx] = result
                        
                        # 提取掩码
                        if output_mask_path and result.mode == 'RGBA':
                            masks[idx] = result.split()[3]  # Alpha channel
                    
                    # 清空批处理队列
                    batch_frames = []
                    frame_idxs = []
                    processed_count += batch
                    
                    # 报告进度
                    if progress_callback:
                        progress_callback(frame_idx, total_frames)
                
                frame_idx += 1
            
            cap.release()
            
            # 移除可能的None值（如果视频帧计数不准确）
            foregrounds = [f for f in foregrounds if f is not None]
            masks = [m for m in masks if m is not None]
            
            # 生成输出文件
            if output_mask_path and masks:
                self.save_frames_to_video(masks, output_mask_path, fps, is_mask=True)
            
            if output_composite_path and foregrounds:
                self.save_frames_with_bg(foregrounds, output_composite_path, fps, rgb_value, orig_width, orig_height)
        
        except Exception as e:
            print(f"视频处理错误: {e}")
            import traceback
            traceback.print_exc()

    
    def save_frames_to_video(self, masks, output_path, fps, is_mask=True):
        """
        将掩码序列保存为视频文件
        
        Args:
            masks: 掩码图像列表 (PIL Images)
            output_path: 输出视频文件路径
            fps: 帧率
            is_mask: 是否为掩码图像 (需要转RGB)
        """
        if not masks:
            print("掩码列表为空，无法保存视频")
            return
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 获取第一帧尺寸
        first_mask = masks[0]
        width, height = first_mask.size
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for mask in masks:
            # 如果是灰度掩码，需要转为RGB视频可以处理的格式
            if is_mask and mask.mode == 'L':
                # 将灰度掩码转为RGB (三通道相同)
                mask_rgb = Image.merge('RGB', (mask, mask, mask))
                frame = np.array(mask_rgb)
            else:
                frame = np.array(mask.convert('RGB'))
                
            # 转换为OpenCV格式 (BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 写入视频
            video_writer.write(frame)
        
        # 释放资源
        video_writer.release()
        print(f"已保存掩码视频: {output_path}")
        
    def save_frames_with_bg(self, foregrounds, output_path, fps, rgb_value, orig_width, orig_height):
        """
        将前景图像序列合成到背景色上并保存为视频
        
        Args:
            foregrounds: 前景图像列表 (带Alpha通道的PIL Images)
            output_path: 输出视频文件路径
            fps: 帧率
            rgb_value: 背景RGB颜色
            orig_width: 原始视频宽度
            orig_height: 原始视频高度
        """
        if not foregrounds:
            print("前景列表为空，无法保存视频")
            return
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
        
        for img in foregrounds:
            # 调整图像尺寸为原始视频尺寸
            if img.size != (orig_width, orig_height):
                img = img.resize((orig_width, orig_height), Image.LANCZOS)
                
            # 合成到背景色上
            if img.mode == 'RGBA':
                # 创建背景
                bg = Image.new('RGB', img.size, rgb_value)
                # 使用alpha通道合成
                img_composite = Image.alpha_composite(bg.convert('RGBA'), img).convert('RGB')
            else:
                # 若无Alpha通道，直接使用
                img_composite = img.convert('RGB')
                
            # 转换为OpenCV格式
            frame = cv2.cvtColor(np.array(img_composite), cv2.COLOR_RGB2BGR)
            
            # 写入视频
            video_writer.write(frame)
        
        # 释放资源
        video_writer.release()
        print(f"已保存合成视频: {output_path}")



def rgb_loader_refiner( original_image):
        h, w = original_image.size

        image = original_image
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image
        image = image.resize((1024, 1024), resample=Image.LANCZOS)

        return image.convert('RGB'), h, w,original_image

# Define the image transformation
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float16), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_transform32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])





def pil_images_to_mp4(images, output_path, fps=24, rgb_value=(0, 255, 0)):
    """
    Converts an array of PIL images to an MP4 video.
    
    Args:
        images: List of PIL images
        output_path: Path to save the MP4 file
        fps: Frames per second (default: 24)
        rgb_value: Background RGB color tuple (default: green (0, 255, 0))
    """
    if not images:
        raise ValueError("No images provided to convert to MP4.")

    width, height = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image in images:
        # If image has alpha channel, composite onto the specified background color
        if image.mode == 'RGBA':
            # Create background image with specified RGB color
            background = Image.new('RGB', image.size, rgb_value)
            background = background.convert('RGBA')
            # Composite the image onto the background
            image = Image.alpha_composite(background, image)
            image = image.convert('RGB')
        else:
            # Ensure RGB format for non-alpha images
            image = image.convert('RGB')

        # Convert to OpenCV format and write
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        video_writer.write(open_cv_image)
    
    video_writer.release()

def pil_images_to_webm_alpha(images, output_path, fps=30):
    """
    Converts a list of PIL RGBA images to a VP9 .webm video with alpha channel.

    NOTE: Not all players will display alpha in WebM. 
          Browsers like Chrome/Firefox typically do support VP9 alpha.
    """
    if not images:
        raise ValueError("No images provided for WebM with alpha.")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save frames as PNG (with alpha)
        for idx, img in enumerate(images):
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            out_path = os.path.join(tmpdir, f"{idx:06d}.png")
            img.save(out_path, "PNG")

        # Construct ffmpeg command
        # -c:v libvpx-vp9 => VP9 encoder
        # -pix_fmt yuva420p => alpha-enabled pixel format
        # -auto-alt-ref 0 => helps preserve alpha frames (libvpx quirk)
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True)

    print(f"WebM with alpha saved to {output_path}")

def add_audio_to_video(video_without_audio_path, original_video_path, output_path):
    """
    Check if the original video has an audio stream. If yes, add it. If not, skip.
    """
    # 1) Probe original video for audio streams
    probe_command = [
        'ffprobe', '-v', 'error', 
        '-select_streams', 'a:0', 
        '-show_entries', 'stream=index', 
        '-of', 'csv=p=0', 
        original_video_path
    ]
    result = subprocess.run(probe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # result.stdout is empty if no audio stream found
    if not result.stdout.strip():
        print("No audio track found in original video, skipping audio addition.")
        return
    
    print("Audio track detected; proceeding to mux audio.")
    # 2) If audio found, run ffmpeg to add it
    command = [
        'ffmpeg', '-y',
        '-i', video_without_audio_path,
        '-i', original_video_path,
        '-c', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',  # we know there's an audio track now
        output_path
    ]
    subprocess.run(command, check=True)
    print(f"Audio added successfully => {output_path}")





### Thanks to the source: https://huggingface.co/ZhengPeng7/BiRefNet/blob/main/handler.py
def refine_foreground_process(image, mask, r=90):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_2(image, mask, r=r)
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    return image_masked


def FB_blur_fusion_foreground_estimator_2(image, alpha, r=90):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator(image, image, image, alpha, r)
    return FB_blur_fusion_foreground_estimator(image, F, blur_B, alpha, r=6)[0]


def FB_blur_fusion_foreground_estimator(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * \
        (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B

    

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array




def rgb_loader_refiner( original_image):
        h, w = original_image.size
        # # Apply EXIF orientation

        image = ImageOps.exif_transpose(original_image)

        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')

        image = original_image
        # Convert to RGB if necessary

        # Resize the image
        image = image.resize((1024, 1024), resample=Image.LANCZOS)

        return image, h, w,original_image



