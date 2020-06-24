import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import (ConvModule, build_upsample_layer, xavier_init, constant_init)

from mmdet.core import auto_fp16
from ..builder import NECKS

@NECKS.register_module()
class YOLONeck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(
                     type="bilinear",
                     scale_factor=2
                 )):
        super(YOLONeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.upsample_cfg = upsample_cfg.copy()

        self.lateral_convs = nn.ModuleList()
        self.upsample_modules = nn.ModuleList()
        self.final_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = nn.Sequential(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

            final_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )

            if i > 0:
                upsample_cfg_ = self.upsample_cfg.copy()
                # suppress warnings
                align_corners = (None
                                 if self.upsample_cfg == 'nearest' else False)
                upsample_cfg_.update(
                    scale_factor=2,
                    mode=self.upsample_cfg,
                    align_corners=align_corners)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
            else:
                self.upsample_modules.append(None)

            self.lateral_convs.append(l_conv)
            self.final_convs.append(final_conv)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.upsample_modules[i](laterals[i])

        # build outputs
        outs = [
            self.final_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return outs





