import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.runner import BaseModule

from ..builder import BACKBONES

class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x).sigmoid()
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class RepVGGplusBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,
				 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
				 norm_cfg=dict(type='BN', requires_grad=True),
				 use_post_se=False):
		super(RepVGGplusBlock, self).__init__()
		self.groups = groups
		self.in_channels = in_channels

		assert kernel_size == 3
		assert padding == 1

		self.nonlinearity = nn.ReLU()

		if use_post_se:
			self.post_se = SEBlock(out_channels, internal_neurons=out_channels // 4)
		else:
			self.post_se = nn.Identity()

		if out_channels == in_channels and stride == 1:
			self.rbr_identity = build_norm_layer(norm_cfg, out_channels)[1]
		else:
			self.rbr_identity = None
		self.rbr_dense = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, norm_cfg=norm_cfg, act_cfg=None)
		padding_11 = padding - kernel_size // 2
		self.rbr_1x1 = ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups, norm_cfg=norm_cfg, act_cfg=None)

	def forward(self, x):
		if self.rbr_identity is None:
			id_out = 0
		else:
			id_out = self.rbr_identity(x)
		out = self.rbr_dense(x) + self.rbr_1x1(x) + id_out
		out = self.post_se(self.nonlinearity(out))
		return out

class RepVGGplusStage(nn.Module):
	def __init__(self, in_planes, planes, num_blocks, stride, norm_cfg, with_cp, use_post_se=False):
		super().__init__()
		strides = [stride] + [1] * (num_blocks - 1)
		blocks = []
		self.in_planes = in_planes
		for stride in strides:
			cur_groups = 1
			blocks.append(RepVGGplusBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, norm_cfg=norm_cfg, use_post_se=use_post_se))
			self.in_planes = planes
		self.blocks = nn.ModuleList(blocks)
		self.with_cp = with_cp

	def forward(self, x):
		for block in self.blocks:
			if self.with_cp and x.requires_grad:
				x = checkpoint.checkpoint(block, x)
			else:
				x = block(x)
		return x

@BACKBONES.register_module()
class RepVGGplus(BaseModule):
	def __init__(self, num_blocks,
				 width_multiplier,
				 norm_cfg=dict(type='BN', requires_grad=True),
				 override_groups_map=None,
				 use_post_se=False,
				 out_indices=(1,2,3,4),
				 with_cp=False,
				 init_cfg=None):
		super().__init__(init_cfg=init_cfg)

		self.override_groups_map = override_groups_map or dict()
		self.use_post_se = use_post_se
		self.with_cp = with_cp
		self.out_indices = out_indices

		self.in_planes = min(64, int(64 * width_multiplier[0]))
		self.stage0 = RepVGGplusBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, use_post_se=use_post_se)
		self.cur_layer_idx = 1
		self.stage1 = RepVGGplusStage(self.in_planes, int(64 * width_multiplier[0]), num_blocks[0], stride=2, norm_cfg=norm_cfg, with_cp=with_cp, use_post_se=use_post_se)
		self.stage2 = RepVGGplusStage(int(64 * width_multiplier[0]), int(128 * width_multiplier[1]), num_blocks[1], stride=2, norm_cfg=norm_cfg, with_cp=with_cp, use_post_se=use_post_se)
		#   split stage3 so that we can insert an auxiliary classifier
		self.stage3_first = RepVGGplusStage(int(128 * width_multiplier[1]), int(256 * width_multiplier[2]), num_blocks[2] // 2, stride=2, norm_cfg=norm_cfg, with_cp=with_cp, use_post_se=use_post_se)
		self.stage3_second = RepVGGplusStage(int(256 * width_multiplier[2]), int(256 * width_multiplier[2]), num_blocks[2] // 2, stride=1, norm_cfg=norm_cfg, with_cp=with_cp, use_post_se=use_post_se)
		self.stage4 = RepVGGplusStage(int(256 * width_multiplier[2]), int(512 * width_multiplier[3]), num_blocks[3], stride=2, norm_cfg=norm_cfg, with_cp=with_cp, use_post_se=use_post_se)

	def forward(self, x):
		outs = []
		out = self.stage0(x)
		if 0 in self.out_indices:
			outs.append(out)
		out = self.stage1(out)
		if 1 in self.out_indices:
			outs.append(out)
		out = self.stage2(out)
		if 2 in self.out_indices:
			outs.append(out)
		out = self.stage3_first(out)
		out = self.stage3_second(out)
		if 3 in self.out_indices:
			outs.append(out)
		out = self.stage4(out)
		if 4 in self.out_indices:
			outs.append(out)
		return outs