_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

class_name = ['Audrey Marie Anderson',
 'Caity Lotz',
 'Emily Bett Rickards',
 'Jessica Elise De Gouw',
 'Juliana Jay Harkavy',
 'Katherine Evelyn Anita Cassidy',
 'Katrina Law',
 'Kelly Ann Hu',
 'Lư Tĩnh San',
 'Susanna Thompson',
 'Willa Joanna Chance Holland']
num_classes = len(class_name)

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'  # noqa

model = dict(
    type='VFNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        # TODO: verify stochastic depth rate {0.1, 0.2, 0.3, 0.4}
        drop_path_rate=0.4,
        layer_scale_init_value=0.,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.6),
        max_per_img=100))

# optimizer
base_lr = 0.0001
optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg=dict(
        bias_lr_mult=2., bias_decay_mult=0.,
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True, type='AdamW', lr=base_lr, weight_decay=0.0001),
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
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=3),
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
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_x101_64x4d_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-b5f6da5e.pth'
log_processor = dict(type='LogProcessor', by_epoch=False)