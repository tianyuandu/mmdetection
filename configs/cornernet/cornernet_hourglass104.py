_base_ = [
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='CornerNet',
    backbone=dict(
        type='Hourglass',
        n=5,
        nstack=2,
        dims=[256, 256, 384, 384, 384, 512],
        modules=[2, 2, 2, 2, 2, 4],
        out_dim=80,
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CornerHead',
        num_classes=81,
        in_channels=256,
        emb_dim=1,
        off_dim=2,
        loss_hmp=dict(type='FocalLoss2D', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_emb=dict(type='AELoss', pull_weight=0.25, push_weight=0.25),
        loss_off=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)))
# data settings
dataset_type = 'CocoDataset'
data_root = '/mnt/lustre/share/DSK/datasets/mscoco2017/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(511, 511),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        train_mode=True),
    dict(type='Resize', img_scale=(511, 511), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        ratio=1.0,
        flip=False,
        transforms=[
            dict(
                type='RandomCenterCropPad',
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                train_mode=False,
                pad_mode=['logical-or', 127]),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'img_norm_cfg', 'border')),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# training and testing settings
train_cfg = None
test_cfg = dict(
    nms_topk=100,
    nms_pool_kernel=3,
    ae_threshold=0.5,
    score_thr=0.05,
    max_per_img=100)
# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[180])
total_epochs = 210
find_unused_parameters = True
