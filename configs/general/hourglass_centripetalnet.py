_base_ = [
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
]

# model settings
max_per_img = 64
model = dict(
	type='CornerNet',
	backbone=dict(
		type='HourglassNet',
		downsample_times=5,
		num_stacks=2,
		stage_channels=[256, 256, 384, 384, 384, 512],
		stage_blocks=[2, 2, 2, 2, 2, 4],
		norm_cfg=dict(type='BN', requires_grad=True),
		with_cp = True),
	neck=None,
	bbox_head=dict(
		type='CentripetalHead',
		num_classes=34,
		in_channels=256,
		num_feat_levels=2,
		corner_emb_channels=0,
		loss_heatmap=dict(type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
		loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1),
		loss_guiding_shift=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.05),
		loss_centripetal_shift=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1)),
	train_cfg=None,
	test_cfg=dict(
		corner_topk=max_per_img,
		local_maximum_kernel=3,
		distance_threshold=0.5,
        score_thr=0.01,
        max_per_img=max_per_img,
        nms=dict(type='soft_nms', iou_threshold=0.6)))

# data setting
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
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
				dict(type='Resize', img_scale=(768, 768), keep_ratio=True),
			],
			[
				dict(type='Mosaic', center_ratio_range=(0.8, 1.2), img_scale=(720, 720), pad_val=0.0),
				dict(type='Resize', img_scale=(768, 768), keep_ratio=True),
			],
			[
				dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.9, 0.9), allow_negative_crop = True),
				dict(type='Resize', img_scale=(768, 768), keep_ratio=True),
			]
		]
	),
	dict(type='Pad', size_divisor=768),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(
		type='CutOut',
		n_holes=(5, 10),
		cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
					  (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32),
					  (32, 48), (48, 32), (48, 48)]),
	dict(
		type='Albu',
		transforms=albu_train_transforms,
		bbox_params=dict(
			type='BboxParams',
			format='pascal_voc',
			label_fields=['gt_labels'],
			min_visibility=0.0,
			filter_lost_elements=True),
		keymap={
			'img': 'image',
			'gt_bboxes': 'bboxes'
		},
		update_pad_shape=False,
		skip_img_without_anno=False),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
	dict(type='LoadImageFromFile', to_float32=True),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(800, 800),
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=True),
			dict(
				type='RandomCenterCropPad',
				ratios=None,
				border=None,
				mean=[0, 0, 0],
				std=[1, 1, 1],
				to_rgb=True,
				test_mode=True,
				test_pad_mode=['size_divisor', 32]),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='DefaultFormatBundle'),
			dict(type='Collect', meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'border'), keys=['img']),
		])
]
data = dict(
	workers_per_gpu=4,
	train=dict(pipeline=train_pipeline),
	val=dict(pipeline=test_pipeline),
	test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(_delete_ = True, type = 'AdamW', lr=0.0005, weight_decay = 0.0001)
optimizer_config = dict(grad_clip = None)
# learning policy
lr_config = dict(_delete_=True,
	policy='CosineAnnealing',
	min_lr_ratio = 0.12,
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=1.0 / 3,
	)
runner = dict(type='IterBasedRunner', max_iters=5000, max_epochs = None)

checkpoint_config = dict(interval = 5000)
evaluation = dict(interval = 5000, metric = 'bbox')

fp16 = dict(loss_scale=512.)

# runtime
resume_from = None
workflow = [('train', 1)]