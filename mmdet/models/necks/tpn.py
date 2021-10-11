# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList

from ..builder import NECKS

class BottleneckBlock(nn.Module):
	def __init__(self, in_channels, hidden_channels, norm_cfg=dict(type='GN', num_groups=32, requires_grad=True), act_cfg=dict(type='ReLU')):
		self.conv1 = ConvModule(in_channels, hidden_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
		self.conv2 = ConvModule(hidden_channels, hidden_channels, 3, padding = 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
		self.conv3 = ConvModule(hidden_channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
	def forward(self, x):
		return x + self.conv3(self.conv2(self.conv1(x)))

@NECKS.register_module()
class TPN(BaseModule):
	"""Trident Pyramid Network.
	"""

	def __init__(self,
				 in_channels,
				 out_channels,
				 hidden_channels,
				 num_outs,
				 stack_times,
				 num_bottleneck_blocks,
				 start_level=0,
				 end_level=-1,
				 with_cp=True,
				 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
				 act_cfg=dict(type='ReLU'),
				 init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')):
		super(TPN, self).__init__(init_cfg)
		assert isinstance(in_channels, list)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_ins = len(in_channels)  # num of input feature levels
		self.num_outs = num_outs  # num of output feature levels
		self.stack_times = stack_times
		self.with_cp = False
		self.norm_cfg = norm_cfg

		if end_level == -1:
			self.backbone_end_level = self.num_ins
			assert num_outs >= self.num_ins - start_level
		else:
			# if end_level < inputs, no extra level is allowed
			self.backbone_end_level = end_level
			assert end_level <= len(in_channels)
			assert num_outs == end_level - start_level
		self.start_level = start_level
		self.end_level = end_level
		self.add_extra_convs = add_extra_convs

		# add lateral connections
		self.lateral_convs = nn.ModuleList()
		for i in range(self.start_level, self.backbone_end_level):
			l_conv = ConvModule(
				in_channels[i],
				out_channels,
				1,
				norm_cfg=norm_cfg,
				act_cfg=None)
			self.lateral_convs.append(l_conv)

		# add extra downsample layers (stride-2 pooling or conv)
		extra_levels = num_outs - self.backbone_end_level + self.start_level
		self.extra_downsamples = nn.ModuleList()
		for i in range(extra_levels):
			extra_conv = ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
			self.extra_downsamples.append(nn.Sequential(extra_conv, nn.MaxPool2d(2, 2)))

		self.down_conv_stages = ModuleList()
		self.up_conv_stages = ModuleList()
		self.parallel_conv_stages = ModuleList()
		for _ in range(self.stack_times):
			down_convs = ModuleList()
			up_convs = ModuleList()
			parallel_convs = ModuleList()

			for i in range(self.num_ins - 1):
				down_convs.append(ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
				up_convs.append(
					nn.Sequential(
						ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
						ConvModule(out_channels, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
						ConvModule(out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
					)
				)

			for i in range(self.num_ins):
				parallel_convs.append(BottleneckBlock(out_channels, hidden_channels, norm_cfg=norm_cfg, act_cfg=act_cfg))

			self.down_conv_stages.append(down_convs)
			self.up_conv_stages.append(up_convs)
			self.parallel_convs.append(parallel_convs)

	def forward(self, inputs):
		"""Forward function."""
		# build P3-P5
		feats = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
		# build P6-P7 on top of P5
		for downsample in self.extra_downsamples:
			feats.append(downsample(feats[-1]))

		for down_convs, up_convs, parallel_convs in zip(self.down_conv_stages, self.up_conv_stages, self.parallel_conv_stages):
			lateral_feats = [feats[-1]]
			for i in range(len(feats) - 1, 0, -1):
				prev_shape = feats[i - 1].shape
				down_feat = down_convs[len(feats) - 1 - i](lateral_feats[-1])
				down_feat = F.interpolate(down_feat, size=prev_shape, mode='nearest')
				lateral_feats.append(feats[i - 1] + down_feat)

			lateral_feats = lateral_feats[::-1]
			lateral_feats = [parallel_conv(lateral_feat) for lateral_feat, parallel_conv in zip(lateral_feats, parallel_convs)]

			out_feats = [lateral_feats[0]]
			for i in range(1, len(feats)):
				up_feat = up_convs[i - 1](out_feats[-1]) + lateral_feats[i]
				out_feats.append(up_feat)
			feats = out_feats

		return tuple(feats)
