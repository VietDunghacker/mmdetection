# Copyright (c) OpenMMLab. All rights reserved.
import timm
from mmengine.model import BaseModule

from mmdet.registry import MODELS

@MODELS.register_module()
class TimmModel(BaseModule):
    def __init__(self,
                 model_name,
                 out_indices=(0, 1, 2, 3),
                 init_cfg=None):
        super(TimmModel, self).__init__(init_cfg=init_cfg)

        self.model = timm.create_model(model_name, pretrained=True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.permute(0, 3, 1, 2)
        return outputs

