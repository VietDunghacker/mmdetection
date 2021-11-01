import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import ModulatedDeformConv2d
from mmcv.runner import BaseModule

from ..builder import NECKS

def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

	if new_v < 0.9 * v:
		new_v += divisor
	return new_v

class DYReLU(nn.Module):
	def __init__(self, in_channels, out_channels, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False, init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
		super(DYReLU, self).__init__()
		self.out_channels = out_channels
		self.lambda_a = lambda_a * 2
		self.K2 = K2
		self.avg_pool = nn.AdaptiveAvgPool2d(1)

		self.use_bias = use_bias
		if K2:
			self.exp = 4 if use_bias else 2
		else:
			self.exp = 2 if use_bias else 1
		self.init_a = init_a
		self.init_b = init_b

		# determine squeeze
		if reduction == 4:
			squeeze = in_channels // reduction
		else:
			squeeze = _make_divisible(in_channels // reduction, 4)

		self.fc = nn.Sequential(
			nn.Linear(in_channels, squeeze),
			nn.ReLU(inplace=True),
			nn.Linear(squeeze, out_channels * self.exp),
			nn.Hardsigmoid(in_place=True)
		)
		if use_spatial:
			self.spa = nn.Sequential(
				nn.Conv2d(in_channels, 1, kernel_size=1),
				nn.BatchNorm2d(1),
			)
		else:
			self.spa = None

	def forward(self, x):
		if isinstance(x, list):
			x_in = x[0]
			x_out = x[1]
		else:
			x_in = x
			x_out = x

		b, c, h, w = x_in.size()
		y = self.avg_pool(x_in).view(b, c)
		y = self.fc(y).view(b, self.out_channels * self.exp, 1, 1)

		if self.exp == 4:
			a1, b1, a2, b2 = torch.split(y, self.out_channels, dim=1)
			a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
			a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

			b1 = b1 - 0.5 + self.init_b[0]
			b2 = b2 - 0.5 + self.init_b[1]
			out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
		elif self.exp == 2:
			if self.use_bias:  # bias but not PL
				a1, b1 = torch.split(y, self.out_channels, dim=1)
				a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
				b1 = b1 - 0.5 + self.init_b[0]
				out = x_out * a1 + b1

			else:
				a1, a2 = torch.split(y, self.out_channels, dim=1)
				a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
				a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
				out = torch.max(x_out * a1, x_out * a2)
		elif self.exp == 1:
			a1 = y
			a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
			out = x_out * a1

		if self.spa:
			ys = self.spa(x_in).view(b, -1)
			ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
			ys = F.hardtanh(ys, 0, 3, inplace=True)/3
			out = out * ys

		return out

class DyConv(nn.Module):
	def __init__(self, in_channels=256, out_channels=256):
		super(DyConv, self).__init__()

		self.DyConv = nn.ModuleList()
		self.DyConv.append(
			nn.Sequential(
				ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=1),
				nn.GroupNorm(out_channels, num_channels=16)
			)
		)
		self.DyConv.append(
			nn.Sequential(
				ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=1),
				nn.GroupNorm(out_channels, num_channels=16)
			)
		)
		self.DyConv.append(
			nn.Sequential(
				ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=2),
				nn.GroupNorm(out_channels, num_channels=16)
			)
		)

		self.AttnConv = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels, 1, kernel_size=1),
			nn.ReLU(inplace=True))

		self.relu = DYReLU(in_channels, out_channels)
		self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		next_x = []
		for level, feature in enumerate(x):
			offset_mask = self.offset(feature)
			offset = offset_mask[:, :18, :, :]
			mask = offset_mask[:, 18:, :, :].sigmoid()

			temp_fea = [self.DyConv[1](feature, offset=offset, mask=mask)]
			if level > 0:
				temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], offset=offset, mask=mask))
			if level < len(x) - 1:
				temp_fea.append(F.upsample_bilinear(self.DyConv[0](x[feature_names[level + 1]], offset=offset, mask=mask), size=[feature.size(2), feature.size(3)]))
			attn_fea = []
			res_fea = []
			for fea in temp_fea:
				res_fea.append(fea)
				attn_fea.append(self.AttnConv(fea))

			res_fea = torch.stack(res_fea)
			spa_pyr_attn = F.hardsigmoid(torch.stack(attn_fea), inplace=True)
			mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
			next_x.append(self.relu(mean_fea))

		return next_x

@NECKS.register_module()
class DyHead(BaseModule):
	def __init__(self,
				 in_channels,
				 out_channels,
				 num_convs,
				 conv_cfg=None,
				 norm_cfg=None,
				 act_cfg=None,
				 init_cfg=dict(type='Normal', layer=['Conv2d', 'ModulatedDeformConv2d'], mean=0, std=0.01)):
		super(DyHead, self).__init__(init_cfg=init_cfg)
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.dyhead_tower = []
		for i in range(num_convs):
			self.dyhead_tower.append(
				DyConv(
					in_channels if i == 0 else out_channels,
					out_channels,
				)
			)

	def forward(self, x):
		dyhead_tower = self.dyhead_tower(x)
		return dyhead_tower
