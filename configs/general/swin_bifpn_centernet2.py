_base_ = [
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
	type='GFL',
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
		type='BiFPN',
		in_channels=[256, 512, 1024],
		out_channels=256,
		input_indices=(1, 2, 3),
		num_outs=5,
		strides=[8, 16, 32],
		num_layers=1,
		weight_method='fast_attn',
		act_cfg='silu',
		separable_conv=True,
		epsilon=0.0001
	),
	rpn_head=dict(
		type='CustomCenterNetHead',
		num_classes=36,
		norm='BN',
		in_channel=256,
		num_features=5,
		num_cls_convs=4,
		num_box_convs=4,
		num_share_convs=0,
		only_proposal=True,
		loss_center_heatmap=dict(type='CustomGaussianFocalLoss', alpha=0.25, ignore_high_fp=-1, loss_weight=0.5),
		loss_bbox=dict(type='CIoULoss', loss_weight=1.0)
		),
	roi_head=dict(
		type='CustomCascadeRoIHead',
		num_stages=3,
		stage_loss_weights=[1, 1, 1],
		bbox_roi_extractor=dict(
			type='SingleRoIExtractor',
			roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
			out_channels=256,
			featmap_strides=[8, 16, 32, 64, 128]),
		bbox_head=[
			dict(
				type='Shared2FCBBoxHead',
				in_channels=256,
				fc_out_channels=1024,
				roi_feat_size=7,
				num_classes=36,
				bbox_coder=dict(
					type='DeltaXYWHBBoxCoder',
					target_means=[0., 0., 0., 0.],
					target_stds=[0.1, 0.1, 0.2, 0.2]),
				reg_class_agnostic=True,
				loss_cls=dict(
					type='CrossEntropyLoss',
					use_sigmoid=False,
					loss_weight=1.0),
				loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
			dict(
				type='Shared2FCBBoxHead',
				in_channels=256,
				fc_out_channels=1024,
				roi_feat_size=7,
				num_classes=36,
				bbox_coder=dict(
					type='DeltaXYWHBBoxCoder',
					target_means=[0., 0., 0., 0.],
					target_stds=[0.05, 0.05, 0.1, 0.1]),
				reg_class_agnostic=True,
				loss_cls=dict(
					type='CrossEntropyLoss',
					use_sigmoid=False,
					loss_weight=1.0),
				loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
			dict(
				type='Shared2FCBBoxHead',
				in_channels=256,
				fc_out_channels=1024,
				roi_feat_size=7,
				num_classes=36,
				bbox_coder=dict(
					type='DeltaXYWHBBoxCoder',
					target_means=[0., 0., 0., 0.],
					target_stds=[0.033, 0.033, 0.067, 0.067]),
				reg_class_agnostic=True,
				loss_cls=dict(
					type='CrossEntropyLoss',
					use_sigmoid=False,
					loss_weight=1.0),
				loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
		]),
	# model training and testing settings
	train_cfg=dict(
		rpn=dict(),
		rpn_proposal=dict(
			nms_pre=2000,
			max_per_img=2000,
			nms=dict(type='nms', iou_threshold=0.9),
			min_bbox_size=0),
		rcnn=[
			dict(
				assigner=dict(
					type='MaxIoUAssigner',
					pos_iou_thr=0.6,
					neg_iou_thr=0.6,
					min_pos_iou=0.6,
					match_low_quality=False,
					ignore_iof_thr=-1),
				sampler=dict(
					type='RandomSampler',
					num=512,
					pos_fraction=0.25,
					neg_pos_ub=-1,
					add_gt_as_proposals=True),
				pos_weight=-1,
				debug=False),
			dict(
				assigner=dict(
					type='MaxIoUAssigner',
					pos_iou_thr=0.7,
					neg_iou_thr=0.7,
					min_pos_iou=0.7,
					match_low_quality=False,
					ignore_iof_thr=-1),
				sampler=dict(
					type='RandomSampler',
					num=512,
					pos_fraction=0.25,
					neg_pos_ub=-1,
					add_gt_as_proposals=True),
				pos_weight=-1,
				debug=False),
			dict(
				assigner=dict(
					type='MaxIoUAssigner',
					pos_iou_thr=0.8,
					neg_iou_thr=0.8,
					min_pos_iou=0.8,
					match_low_quality=False,
					ignore_iof_thr=-1),
				sampler=dict(
					type='RandomSampler',
					num=512,
					pos_fraction=0.25,
					neg_pos_ub=-1,
					add_gt_as_proposals=True),
				pos_weight=-1,
				debug=False)
		]),
	test_cfg=dict(
		rpn=dict(
			nms_pre=1000,
			max_per_img=1000,
			nms=dict(type='nms', iou_threshold=0.9),
			min_bbox_size=0),
		rcnn=dict(
			score_thr=0.05,
			nms=dict(type='nms', iou_threshold=0.7),
			max_per_img=100)))

# data setting
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
	dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.1, rotate_limit=2, interpolation=1, p=0.5, border_mode = 0),
	dict(type='RandomBrightnessContrast', brightness_limit=0.1, contrast_limit=0.1),
	dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
	dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
	dict(
		type='OneOf',
		transforms=[
			dict(type='ChannelShuffle', p=1.0),
			dict(type='ToGray', p = 1.0)
		],
		p=0.1),
]

train_pipeline = [
	dict(
		type = 'AutoAugment',
		policies = [
			[
				dict(type='Mosaic', center_ratio_range=(0.9, 1.1), img_scale=(720, 720), pad_val=0.0),
				dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
			],
			[
				dict(type='Mosaic', center_ratio_range=(0.95, 1.05), img_scale=(720, 720), pad_val=0.0),
				dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
			],
			[
				dict(
					type='Albu',
					transforms=[dict(type = "Crop", x_min = 0, y_min = 400, x_max = 800, y_max = 800)],
					bbox_params=dict(
						type='BboxParams',
						format='pascal_voc',
						label_fields=['gt_labels'],
						min_visibility=0.8,
						filter_lost_elements=True),
					keymap={
						'img': 'image',
						'gt_bboxes': 'bboxes'
					},
					update_pad_shape=False,
					skip_img_without_anno=False),
				dict(type = 'Pad', size_divisor = 800),
			],
			[
				dict(
					type='Albu',
					transforms=[
						dict(
							type = "OneOf",
							transforms=[dict(type = "Crop", x_min = 0, y_min = i, x_max = 800, y_max = 800) for i in range(400, 700, 10)],
							p=1.0),							
						],
					bbox_params=dict(
						type='BboxParams',
						format='pascal_voc',
						label_fields=['gt_labels'],
						min_visibility=0.8,
						filter_lost_elements=True),
					keymap={
						'img': 'image',
						'gt_bboxes': 'bboxes'
					},
					update_pad_shape=False,
					skip_img_without_anno=False),
				dict(type = 'Pad', size_divisor = 800),
				dict(
					type='MixUp',
					img_scale=(800, 800),
					ratio_range=(1.0, 1.0),
					pad_val=0.0),
			],
			[
				dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.9, 0.9), allow_negative_crop = True),
				dict(type='Resize', img_scale=[(640, 640), (800, 800)], multiscale_mode='range', keep_ratio=True),
			],
			[
				dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.9, 0.9), allow_negative_crop = True),
				dict(type='Resize', img_scale=[(640, 640), (800, 800)], multiscale_mode='range', keep_ratio=True),
			]
		]
	),
	dict(
		type='CutOut',
		n_holes=(5, 10),
		cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
					  (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32),
					  (32, 48), (48, 32), (48, 48)]),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(
		type='Albu',
		transforms=albu_train_transforms,
		bbox_params=dict(
			type='BboxParams',
			format='pascal_voc',
			label_fields=['gt_labels'],
			min_visibility=0.1,
			filter_lost_elements=True),
		keymap={
			'img': 'image',
			'gt_bboxes': 'bboxes'
		},
		update_pad_shape=False,
		skip_img_without_anno=True),	
	dict(type='Pad', size_divisor=32),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(800, 800),
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=32),
			dict(type='DefaultFormatBundle'),
			dict(type='Collect', keys=['img']),
		])
]
data = dict(
	workers_per_gpu=4,
	train=dict(pipeline=train_pipeline),
	val=dict(pipeline=test_pipeline),
	test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
	_delete_ = True,
	type='AdamW',
	lr=0.0001,
	betas=(0.9, 0.999),
	weight_decay=0.05,
	paramwise_cfg=dict(
		custom_keys={
			'absolute_pos_embed': dict(decay_mult=0.),
			'relative_position_bias_table': dict(decay_mult=0.),
			'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
log_config = dict(interval = 10)
# learning policy
lr_config = dict(
	_delete_ = True,
	policy='CosineAnnealing',
	min_lr_ratio = 0.12,
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=1.0 / 3,
	)
runner = dict(type='IterBasedRunner', max_iters=10000, max_epochs = None)
checkpoint_config = dict(interval = 100)
evaluation = dict(interval = 100, metric = 'bbox')

fp16 = dict(loss_scale = 512.)

# runtime
load_from = None
resume_from = None
workflow = [('train', 1)]
