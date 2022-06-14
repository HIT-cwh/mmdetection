# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch
import matplotlib.pyplot as plt
import numpy as np

from .fpn import FPN

from ..builder import NECKS


@NECKS.register_module()
class FPNWarpper(BaseModule):
    def __init__(self,
                 # stu_channels=None,
                 # warp_norm_cfg=dict(type='BN', requires_grad=True),
                 *args, **kwargs):
        super(FPNWarpper, self).__init__()
        self.fpn = FPN(*args, **kwargs)
        # out_channels = self.fpn.out_channels
        #
        # stu_channels = stu_channels or out_channels
        # self.align_convs = nn.ModuleList()
        # for i in range(self.fpn.num_outs):
        #     align_module = ConvModule(
        #         out_channels,
        #         stu_channels,
        #         1,
        #         conv_cfg=None,
        #         norm_cfg=warp_norm_cfg,
        #         act_cfg=None,
        #         inplace=False)
        #     self.align_convs.append(align_module)

    def init_weights(self):
        self.fpn.init_weights()
        self._is_init = True

    def forward(self, inputs):
        fpn_outs = self.fpn(inputs)
        sizes = [out.shape[-2:] for out in fpn_outs]
        out_sizes = sizes[1:]
        p6 = sizes[-1]
        p7 = ((p6[0] - 1) // 2 + 1, (p6[1] - 1) // 2 + 1)
        out_sizes += [p7]
        outs = []
        for out, out_size in zip(fpn_outs, out_sizes):
            outs.append(F.interpolate(out, out_size, mode='bilinear'))
        # for i, fpn_out in enumerate(fpn_outs):
        #     out = self.align_convs[i](fpn_out)
        #     outs.append(out)
        return tuple(outs)
