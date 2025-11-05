_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

class_name = ['Alexandra Lecciones Doig',
 'Audrey Marie Anderson',
 'Caity Lotz',
 'Candice Patton',
 'Ciara Renée Harper',
 'Danielle Nicole Panabaker',
 'Elizabeth Melise Jow',
 'Emily Bett Rickards',
 'Jessica Elise De Gouw',
 'Juliana Jay Harkavy',
 'Katherine Evelyn Anita Cassidy',
 'Katherine Grace McNamara',
 'Katrina Law',
 'Kelly Ann Hu',
 'Lư Tĩnh San',
 'Melissa Marie Benoist',
 'Susanna Thompson',
 'Willa Joanna Chance Holland']
num_classes = len(class_name)

model = dict(
    type='CornerNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CentripetalHead',
        num_classes=80,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=0,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1),
        loss_guiding_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=0.05),
        loss_centripetal_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms=dict(type='nms', iou_threshold=0.6, method='gaussian')))

# optimizer
base_lr = 0.0001
optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True, type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=None)

dataset_type = 'CocoDataset'
data_root = '/workspace/compressed_data_yolo/'
metainfo = {
    'classes': class_name,
}

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None
albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0, scale_limit=0.1, rotate_limit=1, interpolation=1, p=0.5, border_mode=0),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    dict(type='RGBShift', r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
    dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
    dict(
        type='OneOf',
        transforms=[
            dict(type='ChannelShuffle', p=1.0),
            dict(type='ToGray', p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='Mosaic', center_ratio_range=(0.95, 1.05), img_scale=(1280, 1280), pad_val=0.0, prob=0.1),
    dict(type='RandomResize', scale=[(960, 960), (1280, 1280)], keep_ratio=True),
    dict(
        type='CutOut',
        n_holes=(5, 25),
        cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32)]),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type="MultiImageMixDataset",
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=backend_args),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            ann_file='celebrity_detection_coco_train.json',
            data_prefix=dict(img='images/train/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=10),
            backend_args=backend_args,
        ),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='celebrity_detection_coco_val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'celebrity_detection_coco_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator


# learning policy
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=10000, val_interval=500)

default_hooks = dict(
    logger=dict(interval=25),
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
)

custom_hooks = [dict(type='Fp16CompresssionHook')]

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.12,
        begin=500,
        by_epoch=False,
        T_max=10000,
    )
]
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth'
log_processor = dict(type='LogProcessor', by_epoch=False)