_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = '/home/vgc/users/lwj/detection_hivit/detection/checkpoint-600.pth'  # noqa
# pretrained = 'D:\MIM\hivit_SAR_tidu_trainfromimagenet\self_supervised\\mae_hivit_base_1600ep.pth'
dataset_type = 'CocoDataset'
data_root = '/home/vgc/users/lwj/detection_hivit/detection/data/SSDD/'
# data_root = '/home/vgc/users/lwj/detection_hivit/detection/data/SAR_AIRcraft/'
# data_root = '/home/vgc/users/lwj/detection_hivit/detection/data/SIVED/'
# data_root = '/home/vgc/users/lwj/detection_hivit/detection/data/SADD/'


model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='HiViT',
        img_size=224,
        patch_size=16,
        embed_dim=512,
        frozen_stages=-1,
        depths=[2, 2, 20],
        num_heads=8,
        mlp_ratio=4.,
        rpe=False,
        drop_path_rate=0.2,
        with_fpn=True,
        out_indices=['H', 'M', 19, 19],
        use_checkpoint=True,
        global_indices=[4, 9, 14, 19],
        window_size=14,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 512],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            # norm_cfg=dict(type='BN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0))),
)

img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=False),  # 512
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        # img_prefix=data_root + 'images/train/',
        img_prefix=data_root + 'images/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/test/',
        pipeline=test_pipeline))


optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='HiViTLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=20, layer_decay_rate=0.9),
)

lr_config = dict(step=[27, 33])
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
runner = dict(type='EpochBasedRunner', max_epochs=36)
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

evaluation = dict(interval=1, metric='bbox', save_best='auto')