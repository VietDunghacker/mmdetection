_base_ = [
    '/workspace/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/workspace/mmdetection/configs/_base_/schedules/schedule_1x.py', '/workspace/mmdetection/configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.CO-DETR.codetr'], allow_failed_imports=False)

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
num_levels = 5
num_dec_layer = 6
loss_lambda = 2.0
model = dict(
    type='CoDETR',
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),
    query_head=dict(
        type='CoDINOHead',
        num_query=num_queries,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=0.5,
            group_cfg=dict(dynamic=False, num_groups=10)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=num_levels,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=4,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=num_levels,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=num_levels,
                            dropout=0.0),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(  # Different from the DINO
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * loss_lambda),
        loss_bbox=dict(
            type='L1Loss', loss_weight=1.0 * num_dec_layer * loss_lambda)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=num_queries,
            # NMS can improve the mAP by 0.2.
            nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=num_queries)),
        dict(
            # atss bbox head:
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=num_queries),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])

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
    dict(type='Mosaic', center_ratio_range=(0.99, 1.01), img_scale=(1280, 1280), pad_val=0.0, prob=0.1),
    dict(type='RandomResize', scale=[(960, 960), (1280, 1280)], keep_ratio=True),
    dict(
        type='CutOut',
        n_holes=(5, 25),
        cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16)]),
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
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
log_processor = dict(type='LogProcessor', by_epoch=False)