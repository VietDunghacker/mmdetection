import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, xavier_init, kaiming_init
from mmcv.runner import BaseModule, auto_fp16
from ..builder import NECKS

from .sepc import PConvModule
from mmdet.models.utils import ModulatedSEPCConv

OPS = {
	'none': lambda in_channels, out_channels: None_(),
	'skip_connect': lambda in_channels, out_channels: Skip_(),
	'TD': lambda in_channels, out_channels: TopDown(in_channels, out_channels),
	'BU': lambda in_channels, out_channels: BottomUp(in_channels, out_channels),
	'FS': lambda in_channels, out_channels: FuseSplit(in_channels, out_channels),
	'SE': lambda in_channels, out_channels: PConvModule(
		in_channels,
		out_channels,
		ibn=True,
		norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
		norm_eval=False,
		part_deform=True),
}

class TopDown(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 ):
		super(TopDown, self).__init__()
		self.tdm_convs = nn.ModuleList()
		for i in range(4):
			tdm_conv = ModulatedSEPCConv(in_channels, out_channels, 3, padding=1, part_deform=True)
			self.tdm_convs.append(tdm_conv)

	def forward(self, inputs):
		# build top-down path
		topdown = []
		topdownconv = self.tdm_convs[-1](1, inputs[-1])
		if topdownconv.shape[2:] != inputs[-1].shape:
			topdownconv = F.interpolate(topdownconv, size=inputs[-1].shape[2:], mode='nearest')

		topdown.append(topdownconv)
		for i in range(3, 0, -1):
			temp = self.tdm_convs[i - 1](i - 1, inputs[i - 1] + F.interpolate(topdownconv.clone(), size=inputs[i - 1].shape[2:], mode='nearest'))
			topdown.insert(0, temp)
			topdownconv = temp
		return topdown

class BottomUp(nn.Module):
	def __init__(self,
				 in_channels,
				 out_channels,
				 ):
		super(BottomUp, self).__init__()
		self.bun_convs = nn.ModuleList()
		for i in range(4):
			bun_conv = ModulatedSEPCConv(in_channels, out_channels, 3, padding=1, part_deform=True)
			self.bun_convs.append(bun_conv)

	def forward(self, inputs):
		# build bottom-up path
		bottomup = []
		for i in range(4):
			if i == 0:
				bum = inputs[0]
			elif i == 3:
				bb = F.max_pool2d(bottomup[-1].clone(), 2, stride=2)
				if bb.shape[2:] != inputs[-1].shape[2:]:
					bb = F.interpolate(bb, size=inputs[-1].shape[2:], mode='nearest')
				bum = bb + inputs[-1]
			else:
				bum = inputs[i] + F.max_pool2d(bottomup[i - 1].clone(), 2, stride=2)
			bottomup.append(self.bun_convs[i](i, bum))

		return bottomup

class FuseSplit(nn.Module):
	def __init__(self, in_channels, out_channels, ):
		super(FuseSplit, self).__init__()
		self.fuse = nn.ModuleList([ModulatedSEPCConv(out_channels * 2, out_channels, 3, padding=1, part_deform=True)] * 2)
		self.in_channels = in_channels
		self.out_channels = out_channels

	def forward(self, inputs):
		# build fusing-splitting path
		fusesplit = []
		fuse1 = inputs[1] + F.max_pool2d(inputs[0], 2, stride=2)
		fuse2 = F.interpolate(inputs[-1], size=inputs[2].shape[2:], mode='nearest') + inputs[2]
		fuseconv1 = self.fuse[0](1, torch.cat([fuse1.clone(), F.interpolate(fuse2.clone(), size=fuse1.shape[2:], mode='nearest')], 1))
		fuseconv2 = self.fuse[1](1, torch.cat([F.max_pool2d(fuse1.clone(), 2, stride=2), fuse2.clone()], 1))

		fusesplit.append(F.interpolate(fuseconv1.clone(), size=inputs[0].shape[2:], mode='nearest'))
		fusesplit.append(fuseconv1)
		fusesplit.append(fuseconv2)
		fusesplit.append(F.max_pool2d(fuseconv2.clone(), 2, stride=2, ceil_mode=False))
		if fusesplit[-1].shape[2:] != inputs[-1].shape[2:]:
			fusesplit[-1] = F.interpolate(fusesplit[-1].clone(), size=inputs[-1].shape[2:], mode='nearest')
		return fusesplit

class None_(nn.Module):
	def __init__(self,):
		super(None_, self).__init__()
		self.size =0
		self.fp = 0
			
	def forward(self, inputs):
		outs = []
		for x in inputs:
			outs.append(x.new_zeros(x.shape))
		return outs

class Skip_(nn.Module):
	def __init__(self):
		super(Skip_, self).__init__()
		self.size = 0
		self.fp = 0

	def forward(self, inputs):
		return inputs

@NECKS.register_module()
class OPA_FPN(BaseModule):
	def __init__(self,
				 in_channels,
				 out_channels,
				 num_outs,
				 num_stacks=5,
				 start_level=0,
				 end_level=-1,
				 add_extra_convs=False,
				 relu_before_extra_convs=False,
				 no_norm_on_lateral=False,
				 conv_cfg=None,
				 norm_cfg=None,
				 act_cfg=None,
				 primitives = ['none', 'skip_connect', 'TD', 'BU', 'FS', 'SE'],
				 paths=None,
				 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')):
		super(OPA_FPN, self).__init__(init_cfg=init_cfg)
		assert isinstance(in_channels, list)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_ins = len(in_channels)
		self.num_outs = num_outs
		self.act_cfg = act_cfg
		self.relu_before_extra_convs = relu_before_extra_convs
		self.no_norm_on_lateral = no_norm_on_lateral
		self.fp16_enabled = False
		
		self.num_stacks = num_stacks
		self.primitives = primitives
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

		self.lateral_convs = nn.ModuleList()
		self.fpn_convs = nn.ModuleList()
		self.information_path = nn.ModuleList()

		self.features = nn.ModuleList()
		self.paths = paths
		for path in self.paths:
			self.features.append(OPS[path](out_channels, out_channels))

		self.topcontext = nn.Sequential(
			ConvModule(
				out_channels,
				out_channels,
				1,
				padding=0,
				stride=1,
				conv_cfg=conv_cfg,
				norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
				act_cfg=self.act_cfg,
				inplace=False),
			nn.AdaptiveAvgPool2d(1))

		for i in range(self.start_level, self.backbone_end_level):
			l_conv = ConvModule(
				in_channels[i],
				out_channels,
				1,
				conv_cfg=conv_cfg,
				norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
				act_cfg=self.act_cfg,
				inplace=False)
			self.lateral_convs.append(l_conv)

		# add extra conv layers (e.g., RetinaNet)
		extra_levels = num_outs - self.backbone_end_level + self.start_level
		if self.add_extra_convs and extra_levels >= 1:
			for i in range(extra_levels):
				extra_fpn_conv = ConvModule(
					out_channels,
					out_channels,
					3,
					stride=2,
					padding=1,
					conv_cfg=conv_cfg,
					norm_cfg=norm_cfg,
					act_cfg=act_cfg,
					inplace=False)
				self.fpn_convs.append(extra_fpn_conv)
	# default init_weights for conv(msra) and norm in ConvModule
	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				xavier_init(m, distribution='uniform')

	@auto_fp16()
	def forward(self, inputs, architecture=None):
		assert len(inputs) == len(self.in_channels)
		# build laterals

		laterals = [
			lateral_conv(inputs[i + self.start_level])
			for i, lateral_conv in enumerate(self.lateral_convs)
		]
		if self.add_extra_convs:
			laterals.append(self.fpn_convs[-2](laterals[-1]))
		used_backbone_levels = len(laterals)
		top = F.interpolate(self.topcontext(laterals[-1]), size=laterals[-1].shape[2:], mode='nearest')
		laterals[-1] = top + laterals[-1]

		info_paths = []
		info_paths.append(laterals)

		for step in range(self.num_stacks):
			_step = step * (step + 1) // 2
			laterals_mid = [laterals[i].new_zeros(laterals[i].shape) for i in range(4)]
			for j in range(step+1):
				temp = self.features[_step + j](info_paths[j])

				for i in range(4):
					laterals_mid[i] += temp[i]
			info_paths.append(laterals_mid)

		outs = info_paths[-1]

		for i in range(1, len(info_paths)-1):
			out = info_paths[i]
			for j in range(4):
				outs[j] += out[j]

		# part 2: add extra levels
		if self.num_outs > len(outs):
			# use max pool to get more levels on top of outputs
			# (e.g., Faster R-CNN, Mask R-CNN)
			if not self.add_extra_convs:
				for i in range(self.num_outs - used_backbone_levels):
					outs.append(F.max_pool2d(outs[-1], 1, stride=2))
			# add conv layers on top of original feature maps (RetinaNet)
			else:
				if self.relu_before_extra_convs:
					outs.append(self.fpn_convs[-1](F.relu(outs[-1])))
				else:
					outs.append(self.fpn_convs[-1](outs[-1]))
		return tuple(outs)
