# model settings
input_size = 608
model = dict(
    type='SingleStageDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='YOLONeck',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(
            type='bilinear',
            scale_factor=2
        )
    ),
    bbox_head=dict(
        type='YOLOV3Head',
        in_channels=[256, 256, 256],
        num_classes=80,
        anchor_generator=dict(
            type="YOLOAnchorGenerator",
            strides=[8, 16, 32],
            mlvl_sizes=[[(10, 13), (16, 30), (33, 23)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(116, 90), (156, 198), (373, 326)]]),
        iou_calculator=dict(
            type="BboxOverlaps2D"
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_obj=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
cudnn_benchmark = True
train_cfg = dict(
    object_thr=0.2)
test_cfg = dict(
    nms=dict(type='nms', iou_thr=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)

