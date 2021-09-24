import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck
from .resnext import Bottleneck as BottleneckX
from .res2net import Bottle2neck

class Root(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, residual):
		super(Root, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=False, padding=(kernel_size - 1) // 2)
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.residual = residual

	def forward(self, *x):
		children = x
		x = self.conv(torch.cat(x, 1))
		x = self.bn(x)
		if self.residual:
			x += children[0]
		x = self.relu(x)

		return x


class Tree(nn.Module):
	def __init__(self, levels, block, in_channels, out_channels, stride=1,
				 level_root=False, root_dim=0, root_kernel_size=1,
				 dilation=1, root_residual=False, with_cp = False):
		super(Tree, self).__init__()
		if root_dim == 0:
			root_dim = 2 * out_channels
		if level_root:
			root_dim += in_channels
		if levels == 1:
			self.tree1 = block(in_channels, out_channels, stride, dilation=dilation, with_cp = with_cp)
			self.tree2 = block(out_channels, out_channels, 1, dilation=dilation, with_cp = with_cp)
		else:
			self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
							  stride, root_dim=0,
							  root_kernel_size=root_kernel_size,
							  dilation=dilation, root_residual=root_residual, with_cp = with_cp)
			self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
							  root_dim=root_dim + out_channels,
							  root_kernel_size=root_kernel_size,
							  dilation=dilation, root_residual=root_residual, with_cp = with_cp)
		if levels == 1:
			self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
		self.level_root = level_root
		self.root_dim = root_dim
		self.downsample = None
		self.project = None
		self.levels = levels
		if stride > 1:
			self.downsample = nn.MaxPool2d(stride, stride=stride)
		if in_channels != out_channels:
			self.project = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x, residual=None, children=None):
		children = [] if children is None else children
		bottom = self.downsample(x) if self.downsample else x
		residual = self.project(bottom) if self.project else bottom
		if self.level_root:
			children.append(bottom)
		x1 = self.tree1(x, residual)
		if self.levels == 1:
			x2 = self.tree2(x1)
			x = self.root(x2, x1, *children)
		else:
			children.append(x1)
			x = self.tree2(x1, children=children)
		return x

@BACKBONES.register_module()
class DLA(BaseModule):
	arch_settings = {
		'dla34': ([1, 1, 1, 2, 2, 1], BasicBlock, [16, 32, 64, 128, 256, 512]),
		'dla60': ([1, 1, 1, 2, 3, 1], Bottleneck, [16, 32, 128, 256, 512, 1024]),
		'dla60x': ([1, 1, 1, 2, 3, 1], BottleneckX, [16, 32, 128, 256, 512, 1024]),
		'dla-to-60': ([1, 1, 1, 2, 3, 1], Bottle2neck, [16, 32, 128, 256, 512, 1024]),
		'dla102': ([1, 1, 1, 3, 4, 1], Bottleneck, [16, 32, 128, 256, 512, 1024]),
		'dla102x': ([1, 1, 1, 3, 4, 1], BottleneckX, [16, 32, 128, 256, 512, 1024]),
		'dla-to-102': ([1, 1, 1, 3, 4, 1], Bottle2neck, [16, 32, 128, 256, 512, 1024]),
		'dla169': ([1, 1, 2, 3, 5, 1], Bottleneck, [16, 32, 128, 256, 512, 1024]),
	}
	def __init__(self,
				 arch,
				 residual_root=False,
				 return_levels=False,
				 pool_size=7,
				 linear_root=False,
				 with_cp = False,
				 out_indices = (0, 1, 2, 3),
				 init_cfg=[
					dict(type='Kaiming', layer='Conv2d'),
					dict(
						type='Constant',
						val=1,
						layer=['_BatchNorm', 'GroupNorm'])
				]):
		super(DLA, self).__init__(init_cfg)
		if arch not in self.arch_settings:
			raise KeyError(f'invalid architecture {arch}')
		levels, block, channels = self.arch_settings[arch]
		if arch != 'dla34':
			block.expansion = 2

		self.out_indices = out_indices

		self.channels = channels
		self.return_levels = return_levels
		self.base_layer = nn.Sequential(
			nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
			nn.BatchNorm2d(channels[0]),
			nn.ReLU(inplace=True))
		self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
		self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
		self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root, with_cp = with_cp)
		self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root, with_cp = with_cp)
		self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root, with_cp = with_cp)
		self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root, with_cp = with_cp)

	def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
		modules = []
		for i in range(convs):
			modules.extend([
				nn.Conv2d(inplanes, planes, kernel_size=3,
						  stride=stride if i == 0 else 1,
						  padding=dilation, bias=False, dilation=dilation),
				nn.BatchNorm2d(planes),
				nn.ReLU(inplace=True)])
			inplanes = planes
		return nn.Sequential(*modules)

	def forward(self, x):
		y = []
		x = self.base_layer(x)
		for i in range(6):
			x = getattr(self, 'level{}'.format(i))(x)
			if i in self.out_indices:
				y.append(x)
		return y

