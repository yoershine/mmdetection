import torch
import numpy as np
from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class YOLOAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    - -1: ignored sample

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 ignore_iou_thr=0.5,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        super(YOLOAssigner).__init__()
        self.ignore_iou_thr = ignore_iou_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        return super().assign(bboxes, gt_bboxes, gt_bboxes_ignore=gt_bboxes_ignore, gt_labels=gt_labels)
    
    def assign(self, 
               mlvl_anchors, mlvl_grids, mlvl_strides, mlvl_featmap_sizes,
               gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """
        Args:
            mlvl_anchors (List[Tensor]): List of anchors(w, h) in each level, shape [(A, 2), ]
            mlvl_grids (List[Tensor]): List of grids(x, y) in each level, shape [(n, 2)]
            mlvl_strides (List[float]): List of stride in each level
            mlvl_featmap_sizes (List[Tensor]): List of featmap size in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        assert len(mlvl_anchors) == len(mlvl_grids) == len(mlvl_strides) == len(mlvl_featmap_sizes)
        device = gt_bboxes.device
        num_levels = len(mlvl_anchors)

        # multi level anchors 
        mlvl_anchors_num = torch.Tensor([anchors.size(0) for anchors in mlvl_anchors]).to(device)
        mlvl_anchors_cusum = torch.cumsum(mlvl_anchors_num, dim=0).to(device)
        mlvl_anchors_cusum_ = torch.cat([torch.Tensor([0]).to(device), mlvl_anchors_cusum])

        # multi level grids
        mlvl_grids_num = torch.Tensor([grids.size(0) for grids in mlvl_grids]).to(device)

        # concat all level anchors to a single tensor
        flat_anchors = torch.cat(mlvl_anchors)

        # caclulate scale overlaps between anchors and gt_bboxes
        gt_wh = gt_bboxes[:, 2:4] - gt_bboxes[:, :2]
        pesudo_gt_bboxes = torch.cat([-0.5 * gt_wh, 0.5*gt_wh], dim=1)
        pesudo_anchors = torch.cat([-0.5 * flat_anchors, 0.5 * flat_anchors], dim=1)
        overlaps = self.iou_calculator(pesudo_gt_bboxes, pesudo_anchors)

        num_gts = gt_bboxes.size(0)
        num_bboxes = torch.sum(mlvl_anchors_num * mlvl_grids_num)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = None
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
            
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
        argmax_level = torch.cat([torch.nonzero(mlvl_anchors_cusum > argmax, as_tuple=False)[0][0] for argmax in gt_argmax_overlaps])
        gt_inds = torch.range(num_gts, dtype=torch.long).to(device)

        assigned_gt_inds = []

        # calculate assigner for each level
        for level_idx in range(num_levels):
            stride = mlvl_strides[level_idx]
            feat_w, feat_h = mlvl_featmap_sizes[level_idx]

            # initialize assigned gt inds by assume all sample is negtive
            assigned_gt_inds_level = overlaps.new_full((mlvl_grids_num[level_idx], mlvl_anchors_num[level_idx]), 
                                                       0, 
                                                       dtype=torch.long)
            # assinged gt inds 
            matched_gt_inds = torch.nonzero(argmax_level == level_idx, as_tuple=False).squeeze(0)
            matched_anchor_inds = gt_argmax_overlaps[matched_gt_inds] - mlvl_anchors_cusum_[level_idx]
            matched_gt_bboxes = gt_bboxes[matched_gt_inds]
            matched_gt_locx = (matched_gt_bboxes[:, 0] / stride).clamp(min=0).int()
            matched_gt_locy = (matched_gt_bboxes[:, 1] / stride).clamp(min=0).int()
            matched_grid_index = matched_gt_locy * feat_w + matched_gt_locx
            assigned_gt_inds_level[matched_grid_index, matched_anchor_inds] = gt_inds[matched_gt_inds] + 1

            # whether to ignore the sample which is overlaped with groud truth bboxes
            if self.ignore_iou_thr > 0:
                anchors = mlvl_anchors[level_idx]
                grids = mlvl_grids[level_idx]
                grid_anchors = torch.cat((grids[:, None, :] - anchors[None, :, :] / 2 + stride / 2,
                                        grids[:, None, :] + anchors[None, :, :] / 2 + stride / 2), dim=-1).view(-1, 4)

                ovelaps_level = self.iou_calculator(gt_bboxes, grid_anchors)
                # for each anchor, which gt best overlaps with it
                # for each anchor, the max iou of all gts
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)
                assigned_gt_inds_level = assigned_gt_inds_level.view(-1)

                # assigne gt inds with -1 when max overlaps between sample and gt bboxes > igore_iou_thr
                assigned_gt_inds_level[max_overlaps > self.ignore_iou_thr] = -1

            assigned_gt_inds.append(assigned_gt_inds_level)
        assigned_gt_inds = torch.cat(assigned_gt_inds)

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)