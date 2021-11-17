_base_ = [
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
	type='YOLOF',
	backbone=dict(
		type='SwinTransformer',
		embed_dims=192,
		depths=[2, 2, 18, 2],
		num_heads=[6, 12, 24, 48],
		window_size=12,
		mlp_ratio=4,
		qkv_bias=True,
		qk_scale=None,
		drop_rate=0.,
		attn_drop_rate=0.,
		drop_path_rate=0.3,
		patch_norm=True,
		out_indices=(3, ),
		convert_weights=True,
		with_cp=True,
		init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth')),
	neck=dict(
		type='DilatedEncoder',
		in_channels=1536,
		out_channels=512,
		block_mid_channels=128,
		num_residual_blocks=4),
	bbox_head=dict(
		type='YOLOFHead',
		num_classes=40,
		in_channels=512,
		reg_decoded_bbox=True,
		anchor_generator=dict(
			type='AnchorGenerator',
			ratios=[1.0],
			scales=[1, 2, 4, 8, 16],
			strides=[32]),
		bbox_coder=dict(
			type='DeltaXYWHBBoxCoder',
			target_means=[.0, .0, .0, .0],
			target_stds=[1., 1., 1., 1.],
			add_ctr_clamp=True,
			ctr_clamp=32),
		loss_cls=dict(
			type='FocalLoss',
			use_sigmoid=True,
			gamma=2.0,
			alpha=0.25,
			loss_weight=1.0),
		loss_bbox=dict(type='CIoULoss', loss_weight=1.0)),
	train_cfg=dict(
		assigner=dict(type='UniformAssigner', pos_ignore_thr=0.15, neg_ignore_thr=0.7),
		allowed_border=-1,
		pos_weight=-1,
		debug=False),
	test_cfg = dict(
		nms_pre=1000,
		min_bbox_size=0,
		score_thr=0.05,
		nms=dict(type='soft-nms', iou_threshold=0.6),
		max_per_img=100)
	)

# data setting
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
	dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0, rotate_limit=0, interpolation=1, p=0.5, border_mode = 0),
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
				dict(type='Mosaic', center_ratio_range=(0.9, 1.1), img_scale=(960, 960), pad_val=0.0),
				dict(type='Resize', img_scale=[(800, 800), (960, 960)], multiscale_mode='range', keep_ratio=True),
			],
			[
				dict(type='Mosaic', center_ratio_range=(0.8, 1.2), img_scale=(960, 960), pad_val=0.0),
				dict(type='Resize', img_scale=[(800, 800), (960, 960)], multiscale_mode='range', keep_ratio=True),
			],
			[
				dict(type='Resize', img_scale=(960, 960), keep_ratio=True),
				dict(type='Pad', size_divisor=960),
				dict(
					type='Albu',
					transforms=[
						dict(type = "Crop", x_min = 0, y_min = 480, x_max = 960, y_max = 960),
						dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, p=0.5, border_mode = 0)],
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
				dict(type='Resize', img_scale=[(640, 640), (960, 960)], multiscale_mode='range', keep_ratio=True, override=True),
			],
			[
				dict(type='Resize', img_scale=(960, 960), keep_ratio=True),
				dict(type='Pad', size_divisor=960),
				dict(
					type='Albu',
					transforms=[
						dict(
							type = "OneOf",
							transforms=[
								dict(type = "Crop", x_min = 0, y_min = i, x_max = 960, y_max = 960) for i in range(480, 720, 10)
								],
							p=1.0),
						dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, p=0.5, border_mode = 0),					
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
				dict(type = 'Pad', size_divisor = 960),
				dict(
					type='MixUp',
					img_scale=(960, 960),
					ratio_range=(1.0, 1.0),
					pad_val=0.0),
			],
			[
				dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.9, 0.9), allow_negative_crop = True),
				dict(type='Resize', img_scale=[(640, 640), (960, 960)], multiscale_mode='range', keep_ratio=True),
			],
			[
				dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.9, 0.9), allow_negative_crop = True),
				dict(type='Resize', img_scale=[(640, 640), (960, 960)], multiscale_mode='range', keep_ratio=True),
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
			min_visibility=0.0,
			filter_lost_elements=True),
		keymap={
			'img': 'image',
			'gt_bboxes': 'bboxes'
		},
		update_pad_shape=False,
		skip_img_without_anno=False),	
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=1),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=(960, 960),
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

custom_hooks = [
	dict(type="HeadHook")
]
