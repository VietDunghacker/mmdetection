_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

class_name = ['Allison Mack',
 'Annette Toole',
 'Audrey Marie Anderson',
 'Azie Tesfai',
 'Brianne Sidonie Desaulniers',
 'Caity Lotz',
 'Candice Patton',
 'Chyler Leigh Potts',
 'Danielle Diggs',
 'Danielle Nicole Panabaker',
 'Elizabeth Chase Olsen',
 'Emily Bett Rickards',
 'Erica Durance',
 'Gwyneth Kate Paltrow',
 'Jessica Lisa Camacho',
 'Jessica Parker Kennedy',
 'Juliana Jay Harkavy',
 'Karen Sheila Gillan',
 'Katherine Evelyn Anita Cassidy',
 'Katherine Grace McNamara',
 'Katie McGrath',
 'Katrina Law',
 'Kayla Compton',
 'Kristin Laura Kreuk',
 'Melissa Marie Benoist',
 'Nicole Amber Maines',
 'Nicole Evangeline Lilly',
 'Odette Juliette Yustman',
 'Pom Alexandra Klementieff',
 'Ruby Rose Langenheim',
 'Scarlett Ingrid Johansson',
 'Susanna Thompson',
 'Trần Diệu Anh',
 'Trần Thu Trang',
 'Willa Joanna Chance Holland',
 'Zoë Yadira Saldaña Nazario']
num_classes = len(class_name)
num_queries = 48
num_levels = 4
model = dict(
    type='DDQDETR',
    num_queries=num_queries,  # num_matching_queries
    # ratio of num_dense queries to num_queries
    dense_topk_ratio=1.5,
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=num_levels,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='TimmModel',
        model_name='vit_base_patch16_dinov3.lvd1689m',
        features_only=True,
        pretrained=True,
        with_cp=True,
        out_indices=(5, 8, 11)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768, 768, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),
    encoder=dict(
        num_layers=6,
        num_cp=3,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    # decoder class name: DDQTransformerDecoder
    decoder=dict(
        # `num_layers` >= 2, because attention masks of the last
        #   `num_layers` - 1 layers are used for distinct query selection
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=-0.5,  # -0.5 for DeformDETR
        temperature=10000),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DDQDETRHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=0.5,
        group_cfg=dict(dynamic=False, num_groups=10)),
    dqs_cfg=dict(type='nms', iou_threshold=0.8),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=num_queries, score_threshold=0.2))  # 100 for DeformDETR

# optimizer
base_lr = 0.0002
optim_wrapper = dict(
    type='AmpOptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
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
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(type='Mosaic', center_ratio_range=(0.99, 1.01), img_scale=(1280, 1280), pad_val=0.0, prob=0.5),
                dict(type='RandomResize', scale=[(960, 960), (1280, 1280)], keep_ratio=True),
                dict(
                    type='CutOut',
                    n_holes=(5, 25),
                    cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16)]
                ),
            ],
            [
                dict(type='Mosaic', center_ratio_range=(0.99, 1.01), img_scale=(1280, 1280), pad_val=0.0, prob=0.0),
                dict(type='RandomResize', scale=[(800, 800), (1280, 1280)], keep_ratio=True),
                dict(
                    type='CutOut',
                    n_holes=(5, 25),
                    cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32)]
                ),
            ],
            [
                dict(type='Mosaic', center_ratio_range=(0.99, 1.01), img_scale=(1280, 1280), pad_val=0.0, prob=0.0),
                dict(type='RandomResize', scale=[(800, 800), (1280, 1280)], keep_ratio=True),
                dict(
                    type='CutOut',
                    n_holes=(5, 25),
                    cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32)]
                ),
            ],
            [
                dict(type='Mosaic', center_ratio_range=(0.99, 1.01), img_scale=(1280, 1280), pad_val=0.0, prob=0.0),
                dict(type='RandomResize', scale=[(1024, 1024), (1280, 1280)], keep_ratio=True),
                dict(
                    type='CutOut',
                    n_holes=(5, 25),
                    cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32)]
                ),
            ],
            [
                dict(type='Mosaic', center_ratio_range=(0.99, 1.01), img_scale=(1280, 1280), pad_val=0.0, prob=0.0),
                dict(type='RandomResize', scale=[(1024, 1024), (1280, 1280)], keep_ratio=True),
                dict(
                    type='CutOut',
                    n_holes=(5, 25),
                    cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32)]
                ),
            ],
        ]
    ),
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
    num_workers=8,
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
            filter_cfg=dict(filter_empty_gt=False),
            backend_args=backend_args,
        ),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=16,
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
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='OlrpMetric',
    ann_file=data_root + 'celebrity_detection_coco_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    backend_args=backend_args)
test_evaluator = val_evaluator


# learning policy
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=10000, val_interval=500)
val_cfg = dict(type='ValLoop', fp16=True)

default_hooks = dict(
    logger=dict(interval=25),
    checkpoint=dict(by_epoch=False, interval=500, max_keep_ckpts=3, save_best='olrp/bbox_oLRP', rule='less'),
)


param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=500,
        by_epoch=False,
        T_max=10000,
    )
]
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/ddq/ddq_detr_swinl_30e.pth'
log_processor = dict(type='LogProcessor', by_epoch=False)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]