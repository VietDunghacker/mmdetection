import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import ModulatedDeformConv2d
from mmcv.runner import BaseModule, ModuleList

from ..builder import NECKS

class ScaleAwareAttention(nn.Module):
	# Constructor
	def __init__(self, size):
		super(ScaleAwareAttention, self).__init__()
		self.L, self.S, self.C = size

		self.conv = nn.Conv2d(in_channels=self.L, out_channels=self.L, kernel_size=1)
		self.hard_sigmoid = nn.Hardsigmoid()
		self.relu = nn.ReLU()

	def forward(self, feat):
		B, L, S, C = feat.shape
		assert L == self.L and S == self.S and C == self.C 
		x = F.avg_pool2d(feat, (self.S, self.C)) #B, L
		x = self.conv(x) #B, L
		x = self.relu(x) #B, L
		scale_attention = self.hard_sigmoid(x)
		return scale_attention * feat

class SpatialAwareAttention(nn.Module):
	# Constructor
	def __init__(self, size, spatial_cfg):
		super(SpatialAwareAttention, self).__init__()
		self.L, self.S, self.C = size

		self.kernel_size = spatial_cfg['kernel_size']
		self.padding = spatial_cfg['padding']
		self.stride = spatial_cfg['stride']
		self.dilation = spatial_cfg['dilation']
		self.K = self.kernel_size[0] * self.kernel_size[1]
		self.groups = spatial_cfg['groups']

		# 3x3 Convolution with 3K out_channel output as described in Deform Conv2 paper
		self.conv_offset = nn.Conv2d(in_channels=1, out_channels=3*self.K, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
		
		self.deform_conv = ModulatedDeformConv2d(in_channels=self.L, out_channels=self.L, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
	
	def forward(self, feat):
		B, L, S, C = feat.shape
		assert L == self.L and S == self.S and C == self.C 
		out = self.conv_offset(feat[:, L // 2].unsqueeze(1))
		assert len(out.shape) == 4

		o1, o2, mask = torch.chunk(out, 3, dim=1)
		offset = torch.cat((o1, o2), dim=1)
		mask = torch.sigmoid(mask)

		spacial_output = self.deform_conv(feat, offset, mask)
		return spacial_output

class TaskAwareAttention(nn.Module):
	def __init__(self, size, task_cfg):
		super(TaskAwareAttention, self).__init__()
		self.L, self.S, self.C = size

		channel_reduction = task_cfg['channel_reduction']
		assert self.C % channel_reduction == 0

		self.fc1 = nn.Linear(self.C, self.C // channel_reduction)
		self.fc2 = nn.Linear(self.C // channel_reduction, 4)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Sigmoid()

		self.lambdas = task_cfg.get('lambdas', torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float))
		self.init_values = task_cfg.get('init_values', torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float))
	
	def forward(self, feat):
		B, L, S, C = feat.shape
		assert L == self.L and S == self.S and C == self.C 
		feat = feat.permute(0, 3, 1, 2)
		x = F.avgpool2d(feat, (self.L, self.S)).squeeze()
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		x = 2 * x.sigmoid() - 1

		thetas = self.init_values.to(feat.device) + self.lambdas.to(feat.device) * x
		alphas = thetas[:, :2]
		betas = thetas[:, 2:]

		output = torch.maximum((alphas[0] * feat + betas[0]), (alphas[1] * feat + betas[0]))
		output = output.permute(0, 2, 3, 1)
		return output

class DyHeadBlock(nn.Module):
	def __init__(self, size, spatial_cfg, task_cfg):
		L, S, C = size
		assert L % 2 == 1
		block = []
		block.append(ScaleAwareAttention(size))
		block.append(SpatialAwareAttention(size, spatial_cfg))
		block.append(TaskAwareAttention(size, task_cfg))
		self.blocks = nn.Sequential(*block)

	def forward(self, feat):
		return self.blocks(feat)

@NECKS.register_module()
class DyHead(BaseModule):
	"""SEPC (Scale-Equalizing Pyramid Convolution).
	https://arxiv.org/abs/2005.03101 https://github.com/jshilong/SEPC
	"""

	def __init__(self,
				 output_shape,
				 num_stacks=6,
				 spatial_cfg = dict(kernel_size=(3, 3), padding=1, stride=1, dilation=1, groups=1),
				 task_cfg = dict(channel_reduction = 8, lambdas = None, init_values = None),
				 init_cfg=dict(type='Xavier', layer=['Conv2d', 'Linear'], distribution='uniform')):
		L, S, C = output_shape
		assert L % 2 == 1
		super(DyHead, self).__init__(init_cfg)
		self.output_shape = output_shape

		self.blocks = ModuleList()
		for _ in range(num_stacks):
			self.blocks.append(DyHeadBlock(output_shape, spatial_cfg, task_cfg))

	def forward(self, feats):
		assert len(in_channels) == len(feats)
		B, C, H, W = feats[(len(feats) - 1) // 2].shape
		assert C == self.output_shape[2] and H * W == self.output_shape[1]
		L, S, C = self.output_shape
		median_level = (L - 1) // 2

		outputs = []
		for i, feat in enumerate(feats):
			if i < median_level:
				outputs.append(F.avg_pool2d(feat, kernel_size = 2 ** (median_level - i), stride = 2 ** (median_level - i)))
			elif i == median_level:
				outputs.append(feat)
			else:
				outputs.append(F.interpolate(feat, size = (H, W), mode = 'bilinear', align_corners = False))
		output = torch.stack(outputs, dim = 1)
		output = output.view(B, L, C, -1).permute(0, 1, 3, 2)
		return [self.blocks(output), ]