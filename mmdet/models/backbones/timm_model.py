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

        self.model = timm.create_model(model_name, features_only=True, pretrained=True, out_indices=out_indices)
        self.model.set_grad_checkpointing()

    def forward(self, x):
        outputs = self.model(x)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

