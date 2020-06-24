import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np

from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmcv.cnn import xavier_init, constant_init, ConvModule

from mmdet.core import (build_assigner, build_sampler, multi_apply, force_fp32, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead


@HEADS.register_module()
class YOLOV3Head(BaseDenseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 mlvl_sizes=[[(10, 13), (16, 30), (33, 23)],
                             [(30, 61), (62, 45), (59, 119)],
                             [(116, 90), (156, 198), (373, 326)]],
                 mlvl_strides=[8, 16, 32],
                 ignore_iou_thr=0.5,
                 eps=1e-6,
                 background_label=None,
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, reduction="sum", loss_weight=1.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, reduction="sum", loss_weight=1.0),
                 loss_center=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, reduction="sum", loss_weight=1.0),    
                 loss_scale=dict(
                     type='MSELoss', reduction="sum", loss_weight=1.0),             
                 **kwargs
                 ):
        super(YOLOV3Head, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(mlvl_sizes, list)
        assert isinstance(mlvl_strides, list)
        assert len(mlvl_strides) == len(in_channels) == len(mlvl_strides)
        assert loss_cls.get('use_sigmoid', False)
        assert loss_obj.get('use_sigmoid', False)

        self.num_levels = len(in_channels)
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.background_label = (
            num_classes if background_label is None else background_label)
        # background_label should be either 0 or num_classes
        assert (self.background_label == 0
                or self.background_label == num_classes)

        self.out_channels = num_classes + 5  # xywh + objectness + num_classes
        self.ignore_iou_thr = ignore_iou_thr

        self.loss_cls = build_loss(loss_cls)
        self.loss_obj = build_loss(loss_obj)
        self.loss_center = build_loss(loss_center)
        self.loss_scale = build_loss(loss_scale)

        self.mlvl_anchors = self._generate_mlvl_anchors(mlvl_sizes)
        self.mlvl_strides = mlvl_strides

        iou_calculator = dict(type="BboxOverlaps2D")
        self.iou_calculator = build_iou_calculator(iou_calculator)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.eps = eps
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        num_anchors = [anchor.size(0) for anchor in self.mlvl_anchors]
        bridge_convs = []
        final_convs = []

        for i in range(self.num_levels):
            output_dim = num_anchors[i] * self.out_channels
            bridge_convs.append(
                ConvModule(self.in_channels[i], self.in_channels[i],
                           kernel_size=3, stride=1, padding=1,
                           norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )
            final_convs.append(
                ConvModule(self.in_channels[i], output_dim,
                           kernel_size=1, stride=1, padding=0,
                           norm_cfg=None, act_cfg=None)
            )
        self.bridge_convs = nn.ModuleList(bridge_convs)
        self.final_convs = nn.ModuleList(final_convs)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, feats):
        preds = []
        for feat, bridge_conv, final_conv in zip(feats, self.bridge_convs, self.final_convs):
            preds.append(final_conv(bridge_conv(feat)))
        return (tuple(preds), )


    def _generate_mlvl_anchors(self, mlvl_sizes, device="cuda"):
        mlvl_anchors = [torch.Tensor(size).to(device) for size in mlvl_sizes]
        return mlvl_anchors

    def _generate_mlvl_grids(self, featmap_sizes, device="cuda"):
        num_levels = len(featmap_sizes)
        mlvl_grids = []

        for i in range(num_levels):
            feat_h, feat_w = featmap_sizes[i]
            grid_x = torch.arange(feat_w)
            grid_y = torch.arange(feat_h)
            grid_xx = grid_x.repeat(len(grid_y))
            grid_yy = grid_y.reshape(-1, 1).repeat(1, len(grid_x)).view(-1)     

            mlvl_grids.append(torch.stack([grid_xx, grid_yy], dim=-1).to(device))    

        return mlvl_grids  

    @force_fp32(apply_to=('mlvl_preds', ))
    def get_bboxes(self,
                   mlvl_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        num_levels = len(mlvl_preds)
        device = mlvl_preds[0].device
        featmap_sizes = [mlvl_preds[i].shape[-2:] for i in range(num_levels)]
        mlvl_grids = self._generate_mlvl_grids(featmap_sizes, device=device)
        
        result_list = []

        for img_id in range(len(img_metas)):
            # mlvl pred for each image
            single_mlvl_preds = [
                mlvl_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(single_mlvl_preds, self.mlvl_anchors, mlvl_grids, 
                                                self.mlvl_strides,
                                                img_shape, scale_factor, cfg, rescale=rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           mlvl_preds,
                           mlvl_anchors,
                           mlvl_grids,
                           mlvl_strides,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False
                           ):
        """Transform outputs for a single batch item into labeled boxes.

        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(mlvl_preds) == len(mlvl_anchors) == len(mlvl_grids)
        mlvl_scores = []
        mlvl_bboxes = []
        mlvl_objs = []

        for pred, anchor, grid, stride in zip(mlvl_preds, mlvl_anchors, mlvl_grids, mlvl_strides):
            feat_h, feat_w = pred.size()[-2:]
            num_anchors = anchor.size(0)
            # preds is num_anchors * (4 + 1 + num_classes) * h * w
            pred = pred.view(num_anchors, -1, feat_h, feat_w).permute(2, 3, 0, 1)
 
            # bboxes
            xy_pred = (torch.sigmoid(pred[..., :2]).view(-1, 2) + grid.repeat(1, num_anchors).view(-1, 2)) * stride
            wh_pred = (torch.exp(pred[..., 2:4])* anchor.expand_as(pred[..., 2:4])).reshape(-1, 2)
            bbox_pred = torch.cat([xy_pred - wh_pred / 2, xy_pred + wh_pred / 2], dim=1).view(-1, 4)
 
            cls_score = torch.sigmoid(pred[..., 5:]).view(-1, self.num_classes)
            obj_score = torch.sigmoid(pred[..., 4]).view(-1)

            # Filtering out all predictions with conf < conf_thr
            obj_thr = cfg.get('conf_thr', -1)
            obj_inds = obj_score.ge(obj_thr).nonzero().flatten()
            bbox_pred = bbox_pred[obj_inds, :]
            cls_score = cls_score[obj_inds, :]
            obj_score = obj_score[obj_inds]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and obj_score.shape[0] > nms_pre:
                _, topk_inds = obj_score.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]
                obj_score = obj_score[topk_inds]

            mlvl_scores.append(cls_score)
            mlvl_bboxes.append(bbox_pred)
            mlvl_objs.append(obj_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_objs = torch.cat(mlvl_objs)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img,
                                                score_factors=mlvl_objs)
        return det_bboxes, det_labels


    def _get_target_single(self,
                           mlvl_preds,
                           mlvl_grids,
                           mlvl_anchors,
                           mlvl_strides,
                           image_meta,
                           gt_bboxes,
                           gt_labels):
        """ Get target of each image

        Args:
            mlvl_preds (list[Tensor]): List of multi level predictions in single image
            mlvl_grids (list[Tensor]): List of multi level grids 
            mlvl_anchors (list[Tensor]): List of multi level anchors 
            mlvl_strides (list[int/float]): List of multi level strides
            image_meta (dict): Image meta infomation including input image shape
            gt_bboxes (Tensor): Ground truth bboxes in each image
            gt_labels (Tensor): Ground truth labels in each image

        Returns:
            bbox_targets List[Tensor]): 
            reg_weights (List[Tensor]):
            assigned_gt_inds (List[Tensor]):  -1 ignored, 0 negtive, >1 positive 
            assigned_labels (List[Tensor]): -1 ignored, >=0 label(0-based)
        """
        assert isinstance(mlvl_anchors, (tuple, list))
        assert isinstance(mlvl_preds, (tuple, list))
        assert isinstance(mlvl_grids, (tuple, list))
        assert len(mlvl_preds) == len(mlvl_anchors) == len(mlvl_grids) == self.num_levels

        device = gt_bboxes.device

        # Origin input image width and height
        pad_shape = image_meta['pad_shape']
        pad_h, pad_w, _ = pad_shape

        mlvl_featmap_sizes = [featmap.size()[-2:] for featmap in mlvl_preds]

        # multi level anchors 
        mlvl_anchors_num = torch.Tensor([anchors.size(0) for anchors in mlvl_anchors]).long().to(device)
        mlvl_anchors_cusum = torch.cumsum(mlvl_anchors_num, dim=0).to(device)
        mlvl_anchors_cusum_ = torch.cat([torch.Tensor([0]).long().to(device), mlvl_anchors_cusum])

        # multi level grids
        mlvl_grids_num = torch.Tensor([grids.size(0) for grids in mlvl_grids]).long().to(device)

        num_gts = gt_bboxes.size(0)

        # concat all level anchors to a single tensor
        flat_anchors = torch.cat(mlvl_anchors)

        # caclulate scale overlaps between anchors and gt_bboxes
        gt_cxy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:4]) / 2
        gt_wh = gt_bboxes[:, 2:4] - gt_bboxes[:, :2]
        gt_xywh = torch.cat([gt_cxy, gt_wh], dim=1)
        pesudo_gt_bboxes = torch.cat([-0.5 * gt_wh, 0.5*gt_wh], dim=1)
        pesudo_anchors = torch.cat([-0.5 * flat_anchors, 0.5 * flat_anchors], dim=1)
        overlaps = self.iou_calculator(pesudo_gt_bboxes, pesudo_anchors)

        # return results
        assigned_gt_inds = []
        bbox_targets = []
        reg_weights = []
        assigned_labels = []

        if num_gts == 0:
            for level_idx in range(self.num_levels):
                grids_num_level = mlvl_grids_num[level_idx]
                anchors_num_level = mlvl_anchors_num[level_idx]
                assigned_gt_inds_level = overlaps.new_full((grids_num_level, anchors_num_level), 0, dtype=torch.long)
                bbox_targets_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
                reg_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 2), 0)
                assigned_labels_level = overlaps.new_full((grids_num_level, anchors_num_level), -1, dtype=torch.long)

                assigned_gt_inds.append(assigned_gt_inds_level)
                bbox_targets.append(bbox_targets_level)
                reg_weights.append(reg_weights_level)
                assigned_labels.append(assigned_labels_level) 

            return bbox_targets, reg_weights, assigned_gt_inds, assigned_labels

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        _, gt_argmax_overlaps = overlaps.max(dim=1)
        argmax_level = torch.stack([torch.nonzero(mlvl_anchors_cusum > argmax, as_tuple=False)[0][0] for argmax in gt_argmax_overlaps])
        gt_inds = torch.arange(0, num_gts, dtype=torch.long).to(device)

        # calculate assigner for each level
        for level_idx in range(self.num_levels):
            stride = mlvl_strides[level_idx]
            feat_h, feat_w = mlvl_featmap_sizes[level_idx]

            grids_num_level = mlvl_grids_num[level_idx]
            anchors_num_level = mlvl_anchors_num[level_idx]

            # initialize assigned gt inds by assume all sample is negtive
            assigned_gt_inds_level = overlaps.new_full((grids_num_level, anchors_num_level), 
                                                       0, 
                                                       dtype=torch.long)

            # initialize bbox_targets
            # initialize reg_weights
            bbox_targets_level = overlaps.new_full((grids_num_level, anchors_num_level, 4), 0)
            reg_weights_level = overlaps.new_full((grids_num_level, anchors_num_level, 2), 0)
            assigned_labels_level = overlaps.new_full((grids_num_level, anchors_num_level), 
                                                       -1, 
                                                       dtype=torch.long)

            # whether to ignore the sample which is overlaped with groud truth bboxes
            if self.ignore_iou_thr > 0:
                anchors = mlvl_anchors[level_idx]
                grids = mlvl_grids[level_idx]
                grid_anchors = torch.cat((grids[:, None, :] * stride - anchors[None, :, :] / 2 + stride / 2,
                                          grids[:, None, :] * stride + anchors[None, :, :] / 2 + stride / 2), dim=-1).view(-1, 4)
                ovelaps_level = self.iou_calculator(gt_bboxes, grid_anchors)
                # if torch.all(ovelaps_level == 0):
                #     import pdb; pdb.set_trace()
                # for each anchor, which gt best overlaps with it
                # for each anchor, the max iou of all gts
                max_overlaps, _ = ovelaps_level.max(dim=0)
                assigned_gt_inds_level = assigned_gt_inds_level.view(-1)

                # assigne gt inds with -1 when max overlaps between sample and gt bboxes > igore_iou_thr
                assigned_gt_inds_level[max_overlaps > self.ignore_iou_thr] = -1
                assigned_gt_inds_level = assigned_gt_inds_level.view(grids_num_level, anchors_num_level)

            # assinged gt inds 
            matched_gt_inds = torch.nonzero(argmax_level == level_idx, as_tuple=False).squeeze(1)
            if matched_gt_inds.numel() > 0:
                matched_anchor_inds = gt_argmax_overlaps[matched_gt_inds] - mlvl_anchors_cusum_[level_idx]
                matched_gt_xywhs = gt_xywh[matched_gt_inds]
                matched_gt_locx = (matched_gt_xywhs[:, 0] / stride).clamp(min=0).long()
                matched_gt_locy = (matched_gt_xywhs[:, 1] / stride).clamp(min=0).long()
                matched_grid_index = matched_gt_locy * feat_w + matched_gt_locx
                assigned_gt_inds_level[matched_grid_index, matched_anchor_inds] = gt_inds[matched_gt_inds] + 1
                bbox_targets_level[matched_grid_index, matched_anchor_inds, 0] = (matched_gt_xywhs[:, 0] / stride - matched_gt_locx).clamp(self.eps, 1 - self.eps)
                bbox_targets_level[matched_grid_index, matched_anchor_inds, 1] = (matched_gt_xywhs[:, 1] / stride - matched_gt_locy).clamp(self.eps, 1 - self.eps)
                matched_gt_bbox_wh = matched_gt_xywhs[:, 2:4]
                matched_anchor = mlvl_anchors[level_idx][matched_anchor_inds]
                bbox_targets_level[matched_grid_index, matched_anchor_inds, 2:4] = torch.log((matched_gt_bbox_wh / matched_anchor).clamp(min=self.eps))
                reg_weights_level[matched_grid_index, matched_anchor_inds, 0] = 2.0 - matched_gt_bbox_wh.prod(1) / pad_w / pad_h
                reg_weights_level[matched_grid_index, matched_anchor_inds, 1] = 2.0 - matched_gt_bbox_wh.prod(1) / pad_w / pad_h
                assigned_labels_level[matched_grid_index, matched_anchor_inds] = gt_labels[matched_gt_inds]

            assigned_gt_inds.append(assigned_gt_inds_level)
            bbox_targets.append(bbox_targets_level)
            reg_weights.append(reg_weights_level)
            assigned_labels.append(assigned_labels_level) 

        return bbox_targets, reg_weights, assigned_gt_inds, assigned_labels

    def get_targets(self, 
                    mlvl_preds_list,
                    mlvl_grids,
                    mlvl_anchors,
                    mlvl_strides,
                    image_metas,
                    gt_bboxes_list,
                    gt_labels_list):
        """
        Args:
            mlvl_preds_list (list[list[Tensor]]): List of multi level predictions in batched images
            mlvl_grids (list[Tensor]): List of multi level grids 
            mlvl_anchors (list[Tensor]): List of multi level anchors 
            mlvl_strides (list[Tuple]): List of multi level strides
            image_metas (list[dict]): List of image meta infomation in batched images
            gt_bboxes_list (list[Tensor]): List of ground truth bboxes in batched image
            gt_labels_list (list[Tensor]): List of ground truth labels in batched image
        Returns:

        """
        num_imgs = len(image_metas)
        mlvl_grids_list = [mlvl_grids] * num_imgs
        mlvl_anchors_list = [mlvl_anchors] * num_imgs
        mlvl_strides_list = [mlvl_strides] * num_imgs

        (all_bbox_targets, all_reg_weights, 
        all_assigned_gt_inds, all_assigned_labels) = multi_apply(
            self._get_target_single,
            mlvl_preds_list,
            mlvl_grids_list,
            mlvl_anchors_list,
            mlvl_strides_list,
            image_metas,
            gt_bboxes_list,
            gt_labels_list
        )

        return all_bbox_targets, all_reg_weights, all_assigned_gt_inds, all_assigned_labels

    @force_fp32(apply_to=('preds_list', ))
    def loss(self,
             preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """ Calculate loss of YOLO

        Args:
            preds_list (list[Tensor]): List of predicted results in multiple feature maps
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image
            gt_label_list (list[Tensor]): Ground truth labels of each box
            img_metas (list[dict]): Meta info of each image
            gt_bboxes_ignore (list[Tensor]): Ground truth labels of each box.
        """
        device = preds_list[0].device
        num_levels = len(preds_list)

        featmap_sizes = [featmap.size()[-2:] for featmap in preds_list]
        mlvl_grids = self._generate_mlvl_grids(featmap_sizes, device=device)
        mlvl_anchors_num = [anchors.size(0) for anchors in self.mlvl_anchors]

        mlvl_preds_list = []
        for img_id in range(len(img_metas)):
            mlvl_preds_list.append([preds_list[level][img_id] for level in range(num_levels)])

        all_bbox_targets, all_reg_weights, all_assigned_gt_inds, all_assigned_labels = \
            self.get_targets(mlvl_preds_list, mlvl_grids, self.mlvl_anchors,
            self.mlvl_strides, img_metas, gt_bboxes_list, gt_labels_list)

        ft = torch.cuda.FloatTensor if preds_list[0].is_cuda else torch.Tensor
        lcls, lcenter, lscale, lobj = ft([0]), ft([0]), ft([0]), ft([0])

        for mlvl_preds, mlvl_bbox_targets, mlvl_reg_weights, mlvl_assigned_gt_inds, mlvl_assigned_labels in \
            zip(mlvl_preds_list, all_bbox_targets, all_reg_weights, all_assigned_gt_inds, all_assigned_labels):

            for level_idx in range(self.num_levels):
                preds = mlvl_preds[level_idx].view(mlvl_anchors_num[level_idx], self.out_channels, -1).permute(2, 0, 1)
                bbox_targets = mlvl_bbox_targets[level_idx]
                reg_weights = mlvl_reg_weights[level_idx]
                assigned_gt_inds = mlvl_assigned_gt_inds[level_idx]
                assigned_labels = mlvl_assigned_labels[level_idx]

                preds_cxy = preds[..., :2]
                preds_wh = preds[..., 2:4]
                preds_obj = preds[..., 4]
                preds_cls = preds[..., 5:]

                pos_inds = assigned_gt_inds[assigned_gt_inds > 0]
                pos_nums = pos_inds.numel()
                if pos_nums > 0:
                    lcenter += self.loss_center(preds_cxy, bbox_targets[..., :2], weight=reg_weights, avg_factor=None)
                    lscale += self.loss_scale(preds_wh, bbox_targets[..., 2:4], weight=reg_weights, avg_factor=None)

                    # construct classification target, and expand binary label into multi label
                    cls_weights = torch.zeros_like(assigned_labels, dtype=torch.long)
                    cls_weights[assigned_labels > -1] = 1
                    cls_weights = cls_weights[..., None].expand_as(preds_cls)
                    cls_targets = assigned_labels.new_full(preds_cls.size(), 0, dtype=torch.long)
                    inds = torch.nonzero(assigned_labels > 0, as_tuple=False)
                    cls_targets[inds[:, 0], inds[:, 1],  assigned_labels[inds[:, 0], inds[:, 1]]] = 1
                    lcls += self.loss_cls(preds_cls, cls_targets, weight=cls_weights, avg_factor=None)

                obj_weights = torch.zeros_like(preds_obj)
                obj_weights[assigned_gt_inds != -1] = 1
                obj_targets = assigned_gt_inds.clamp(min=0, max=1)
                lobj += self.loss_obj(preds_obj, obj_targets, weight=obj_weights, avg_factor=None)
        return dict(loss_center=lcenter, loss_scale=lscale, loss_object=lobj, loss_cls=lcls,)
