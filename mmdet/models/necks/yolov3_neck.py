import torch
from torch import layer_norm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import ConvModule, build_upsample_layer, xavier_init, constant_init

from ..builder import NECKS


class RefineBlock(nn.Module):
    """ RefineBlock, contains 4 conv layers
        in => Conv_3x3_2*in => Conv_1x1_in => Conv_3x3_2*in => Conv_1x1_in
    Args:
        channels (int): input channels
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for norm layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
    """
    def __init__(self, 
                 channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None) -> None:
        super(RefineBlock, self).__init__()
        layers = []
        layer_nums = 4
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        for i in range(layer_nums):
            if i % 2 == 0:
                layers.append(ConvModule(channels, channels * 2, 3, padding=1, **cfg))
            else:
                layers.append(ConvModule(channels * 2, channels, 1, **cfg))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


@NECKS.register_module()
class YOLOV3Neck(nn.Module):
    """ YOLOV3 Neck

    C3 -> cat(C4_Refine_Upsample, C3) -> C3_Lateral -> C3_Refine -> C3_Final -> C3_Output
    C4 -> cat(C5_Refine_Upsample, C4) -> C4_Lateral -> C4_Refine -> C4_Final -> C4_Output 
    C5 -> C5_lateral -> C5_Refine -> C5_Final -> C5_Output

    Args:
        in_channels (List(int)): List of input channels
        out_channels (List(int)): List of output channels
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for norm layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.      
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=None) -> None:
        super(YOLOV3Neck, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, list)
        assert len(in_channels) == len(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.upsample_cfg = upsample_cfg.copy()

        # build lateral convs
        self.lateral_convs = nn.ModuleList()
        self.upsample_modules = nn.ModuleList()
        self.refine_modules = nn.ModuleList()
        self.final_convs = nn.ModuleList()

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        for i in range(self.num_ins):
            lateral_inc = in_channels[i] + (out_channels[i + 1] if i + 1 < self.num_ins else 0)
            l_conv = ConvModule(lateral_inc, out_channels[i], 1, **cfg)
            
            if i > 0:
                upsample_cfg_ = self.upsample_cfg.copy()
                # suppress warnings
                align_corners = (None
                                 if self.upsample_cfg.type == 'nearest' else False)
                upsample_cfg_.update(
                    scale_factor=2,
                    mode=self.upsample_cfg,
                    align_corners=align_corners)
                upsample_module = build_upsample_layer(upsample_cfg_)
                self.upsample_modules.append(upsample_module)
            else:
                self.upsample_modules.append(None)

            refine_block = RefineBlock(out_channels[i], **cfg)

            final_conv = ConvModule(out_channels[i], out_channels[i] * 2, 3, padding=1, **cfg)

            self.lateral_convs.append(l_conv)
            self.refine_modules.append(refine_block)
            self.final_convs.append(final_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)   

    def forward(self, inputs):
        assert len(inputs) == self.num_ins

        laterals = []
        # build top-down path, ie. high-level => low-level
        # calculate highest level feats
        feats = self.refine_modules[-1](self.lateral_convs[-1](inputs[-1]))
        laterals.append(feats)

        for i in range(self.num_ins - 1, 0, -1):
            upsample_feats = self.upsample_modules[i](feats)
            cat_feats = torch.cat((upsample_feats, inputs[i - 1]), dim=1)
            feats = self.refine_modules[i - 1](self.lateral_convs[i - 1](cat_feats))
            laterals.append(feats)
        
        # Reverse feats orders
        laterals = laterals[::-1]

        outs = [final_conv(laterals[i]) for i, final_conv in enumerate(self.final_convs)]
        return tuple(outs)
