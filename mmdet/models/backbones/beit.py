# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path

from mmcv.cnn import trunc_normal_init, constant_init
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.utils import to_2tuple
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from ..utils.transformer import PatchEmbed

class DropPath(nn.Module):
	"""Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
	"""
	def __init__(self, drop_prob=None):
		super(DropPath, self).__init__()
		self.drop_prob = drop_prob

	def forward(self, x):
		return drop_path(x, self.drop_prob, self.training)
	
	def extra_repr(self) -> str:
		return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
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
		x = self.fc2(x)
		x = self.drop(x)
		return x

class Attention(nn.Module):
	def __init__(
			self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
			proj_drop=0., window_size=None, attn_head_dim=None):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		if attn_head_dim is not None:
			head_dim = attn_head_dim
		all_head_dim = head_dim * self.num_heads
		# NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
		self.scale = qk_scale or head_dim ** -0.5

		self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
		if qkv_bias:
			self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
			self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
		else:
			self.q_bias = None
			self.v_bias = None

		if window_size:
			self.window_size = window_size
			self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
			self.relative_position_bias_table = nn.Parameter(
				torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
			# cls to token & token 2 cls & cls to cls

			# get pair-wise relative position index for each token inside the window
			coords_h = torch.arange(window_size[0])
			coords_w = torch.arange(window_size[1])
			coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
			coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
			relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
			relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
			relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
			relative_coords[:, :, 1] += window_size[1] - 1
			relative_coords[:, :, 0] *= 2 * window_size[1] - 1
			relative_position_index = \
				torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
			relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
			relative_position_index[0, 0:] = self.num_relative_distance - 3
			relative_position_index[0:, 0] = self.num_relative_distance - 2
			relative_position_index[0, 0] = self.num_relative_distance - 1

			self.register_buffer("relative_position_index", relative_position_index)

			# trunc_normal_(self.relative_position_bias_table, std=.0)
		else:
			self.window_size = None
			self.relative_position_bias_table = None
			self.relative_position_index = None

		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(all_head_dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, rel_pos_bias=None):
		B, N, C = x.shape
		qkv_bias = None
		if self.q_bias is not None:
			qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
		# qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
		qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
		qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))

		if self.relative_position_bias_table is not None:
			relative_position_bias = \
				self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
					self.window_size[0] * self.window_size[1] + 1,
					self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
			relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
			attn = attn + relative_position_bias.unsqueeze(0)

		if rel_pos_bias is not None:
			attn = attn + rel_pos_bias
		
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class Block(nn.Module):

	def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
				 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
				 window_size=None, attn_head_dim=None):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(
			dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
			attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

		if init_values is not None:
			self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
			self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
		else:
			self.gamma_1, self.gamma_2 = None, None

	def forward(self, x, rel_pos_bias=None):
		if self.gamma_1 is None:
			x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
			x = x + self.drop_path(self.mlp(self.norm2(x)))
		else:
			x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
			x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
		return x

class RelativePositionBias(nn.Module):

	def __init__(self, window_size, num_heads):
		super().__init__()
		self.window_size = window_size
		self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
		self.relative_position_bias_table = nn.Parameter(
			torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
		# cls to token & token 2 cls & cls to cls

		# get pair-wise relative position index for each token inside the window
		coords_h = torch.arange(window_size[0])
		coords_w = torch.arange(window_size[1])
		coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
		coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
		relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
		relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
		relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
		relative_coords[:, :, 1] += window_size[1] - 1
		relative_coords[:, :, 0] *= 2 * window_size[1] - 1
		relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
		relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
		relative_position_index[0, 0:] = self.num_relative_distance - 3
		relative_position_index[0:, 0] = self.num_relative_distance - 2
		relative_position_index[0, 0] = self.num_relative_distance - 1

		self.register_buffer("relative_position_index", relative_position_index)

	def forward(self):
		relative_position_bias = \
			self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
				self.window_size[0] * self.window_size[1] + 1,
				self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
		return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


@BACKBONES.register_module()
class BEiT(BaseModule):
	""" Vision Transformer with support for patch or hybrid CNN input stage
	"""
	def __init__(self,
				 pretrain_img_size=224,
				 in_channels=3,
				 embed_dims=768,
				 patch_size=16,
				 depth=12,
				 num_heads=12,
				 mlp_ratio=4.,
				 qkv_bias=False,
				 qk_scale=None,
				 drop_rate=0.,
				 attn_drop_rate=0.,
				 drop_path_rate=0.,
				 norm_layer=None,
				 init_values=None,
				 use_checkpoint=False, 
				 use_abs_pos_emb=True,
				 use_rel_pos_bias=False,
				 use_shared_rel_pos_bias=False,
				 out_indices=[3, 5, 7, 11],
				 init_cfg = None):
		super(BEiT, self).__init__(init_cfg = init_cfg)
		norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
		self.num_features = self.embed_dims = embed_dims  # num_features for consistency with other models

		self.patch_embed = PatchEmbed(in_channels = in_channels, embed_dims = embed_dims, kernel_size = patch_size, stride = patch_size)
		num_patches = self.patch_embed.num_patches
		self.out_indices = out_indices

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
		if use_abs_pos_emb:
			self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
		else:
			self.pos_embed = None
		self.pos_drop = nn.Dropout(p=drop_rate)

		if use_shared_rel_pos_bias:
			self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
		else:
			self.rel_pos_bias = None

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
		self.use_rel_pos_bias = use_rel_pos_bias
		self.use_checkpoint = use_checkpoint
		self.blocks = nn.ModuleList([
			Block(
				dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
				drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
				init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
			for i in range(depth)])

		if self.pos_embed is not None:
			trunc_normal_init(self.pos_embed, std=.02)
		trunc_normal_init(self.cls_token, std=.02)
		self.out_indices = out_indices

		if patch_size == 16:
			self.fpn1 = nn.Sequential(
				nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
				nn.BatchNorm2d(embed_dims),
				nn.GELU(),
				nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
			)

			self.fpn2 = nn.Sequential(
				nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
			)

			self.fpn3 = nn.Identity()

			self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
		elif patch_size == 8:
			self.fpn1 = nn.Sequential(
				nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
			)

			self.fpn2 = nn.Identity()

			self.fpn3 = nn.Sequential(
				nn.MaxPool2d(kernel_size=2, stride=2),
			)

			self.fpn4 = nn.Sequential(
				nn.MaxPool2d(kernel_size=4, stride=4),
			)

	def init_weights(self):
		logger = get_root_logger()
		if self.init_cfg is None:
			logger.warn(f'No pre-trained weights for '
						f'{self.__class__.__name__}, '
						f'training start from scratch')
			if self.use_abs_pos_embed:
				trunc_normal_init(self.absolute_pos_embed, std=0.02)
			for m in self.modules():
				if isinstance(m, nn.Linear):
					trunc_normal_init(m.weight, std=.02)
					if m.bias is not None:
						constant_init(m.bias, 0)
				elif isinstance(m, nn.LayerNorm):
					constant_init(m.bias, 0)
					constant_init(m.weight, 1.0)
		else:
			assert 'checkpoint' in self.init_cfg, f'Only support ' \
												  f'specify `Pretrained` in ' \
												  f'`init_cfg` in ' \
												  f'{self.__class__.__name__} '
			ckpt = _load_checkpoint(self.init_cfg.checkpoint, logger=logger, map_location='cpu')
			if 'state_dict' in ckpt:
				_state_dict = ckpt['state_dict']
			elif 'model' in ckpt:
				_state_dict = ckpt['model']
			else:
				_state_dict = ckpt

			state_dict = OrderedDict()
			for k, v in _state_dict.items():
				if k.startswith('backbone.'):
					state_dict[k[9:].replace('proj.', 'projection.')] = v

			# strip prefix of state_dict
			if list(state_dict.keys())[0].startswith('module.'):
				state_dict = {k[7:]: v for k, v in state_dict.items()}

			# reshape absolute position embedding
			if state_dict.get('pos_embed') is not None:
				absolute_pos_embed = state_dict['pos_embed']
				N1, L, C1 = absolute_pos_embed.size()
				N2, C2, H, W = self.absolute_pos_embed.size()
				if N1 != N2 or C1 != C2 or L != H * W:
					logger.warning('Error in loading pos_embed, pass')
				else:
					state_dict['pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

			# interpolate position bias table if needed
			relative_position_bias_table_keys = [
				k for k in state_dict.keys()
				if 'relative_position_bias_table' in k
			]
			for table_key in relative_position_bias_table_keys:
				table_pretrained = state_dict[table_key]
				table_current = self.state_dict()[table_key]
				L1, nH1 = table_pretrained.size()
				L2, nH2 = table_current.size()
				if nH1 != nH2:
					logger.warning(f'Error in loading {table_key}, pass')
				elif L1 != L2:
					S1 = int(L1**0.5)
					S2 = int(L2**0.5)
					table_pretrained_resized = F.interpolate(
						table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
						size=(S2, S2),
						mode='bicubic')
					state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()

			# load state_dict
			self.load_state_dict(state_dict, False)

	def forward_features(self, x):
		B, C, H, W = x.shape
		x, (Hp, Wp) = self.patch_embed(x)
		batch_size, seq_len, _ = x.size()

		cls_tokens = self.cls_token.expand(batch_size, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		if self.pos_embed is not None:
			x = x + self.pos_embed
		x = self.pos_drop(x)

		rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
		features = []
		for i, blk in enumerate(self.blocks):
			if self.use_checkpoint:
				x = checkpoint.checkpoint(blk, x, rel_pos_bias)
			else:
				x = blk(x, rel_pos_bias)
			if i in self.out_indices:
				xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)
				features.append(xp.contiguous())

		ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
		for i in range(len(features)):
			features[i] = ops[i](features[i])

		return tuple(features)

	def forward(self, x):
		x = self.forward_features(x)
		return x
