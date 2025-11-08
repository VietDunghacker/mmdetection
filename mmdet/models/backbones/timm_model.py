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

        self.model = timm.create_model(model_name, features_only=True, pretrained=True)
        assert False, self.model.feature_info.channels()

    def forward(self, x):
        return self.model(x)

