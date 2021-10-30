from ..builder import NECKS
from .fpn import FPN
from .pafpnx import PAFPNX

@NECKS.register_module()
class CBFPN(FPN):
	"""FPN for CBNetV2.
	This neck supports multiple outputs from CBNet backbones. The same neck
	weights are shared between different backbones.
	https://arxiv.org/abs/2107.00420
	"""

	def forward(self, inputs):
		"""Forward function."""
		if not isinstance(inputs[0], (list, tuple)):
			inputs = [inputs]

		if self.training:
			# forward features from assisting backbones and lead backbone
			outs = [super(CBFPN, self).forward(x) for x in inputs]
			return outs
		else:
			# forward features from lead backbone (inputs[-1]) only
			out = super().forward(inputs[-1])
			return out

@NECKS.register_module()
class CBPAFPNX(PAFPNX):
	"""FPN for CBNetV2.
	This neck supports multiple outputs from CBNet backbones. The same neck
	weights are shared between different backbones.
	https://arxiv.org/abs/2107.00420
	"""

	def forward(self, inputs):
		"""Forward function."""
		if not isinstance(inputs[0], (list, tuple)):
			inputs = [inputs]

		if self.training:
			# forward features from assisting backbones and lead backbone
			outs = [super(CBPAFPNX, self).forward(x) for x in inputs]
			return outs
		else:
			# forward features from lead backbone (inputs[-1]) only
			out = super().forward(inputs[-1])
			return out
