from .GFL import GFLHead
from ..builder import HEADS

class ScaleAwareAttention(nn.Module):
	# Constructor
	def __init__(self, in_channels):
		super(ScaleAwareAttention, self).__init__()

		self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)
		self.hard_sigmoid = nn.Hardsigmoid()
		self.relu = nn.ReLU()

	def forward(self, feat):
		B, L, S, C = feat.shape
		x = feat.permute(0, 2, 1, 3).contiguous()
		x = self.pool(x)
		x = self.conv(x).permute(0, 2, 1, 3).contiguous()
		x = self.relu(x)
		scale_attention = self.hard_sigmoid(x)
		return scale_attention * feat

@HEADS.register_module()
class DyGFLHead(GFLHead):
	def __init__(self,
				 num_stacks,
				 init_cfg=dict(
					 type='Normal',
					 layer='Conv2d',
					 std=0.01,
					 override=dict(
						 type='Normal',
						 name='gfl_cls',
						 std=0.01,
						 bias_prob=0.01)),
				 **kwargs):
		super(DyGFLHead, self).__init__(init_cfg=init_cfg, **kwargs)
		self.num_stacks = num_stacks
