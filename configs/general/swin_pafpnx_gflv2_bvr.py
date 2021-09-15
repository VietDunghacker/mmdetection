_base_ = [
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
	type='BVR',
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
		out_indices=(0, 1, 2, 3, ),
		with_cp=True,
		init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth')),
	neck= dict(
		type='PAFPNX',
		in_channels=[128, 256, 512, 1024],
		out_channels=256,
		start_level=2,
		add_extra_convs='on_output',
		num_outs=4,
		relu_before_extra_convs=True,
		pafpn_conv_cfg=dict(type='DCNv2'),
		norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
	bbox_head=dict(
		type='BVRHead',
		bbox_head_cfg=dict(
			type='GFLHead',
			num_classes=34,
			in_channels=256,
			stacked_convs=4,
			feat_channels=256,
			anchor_generator=dict(
				type='AnchorGenerator',
				ratios=[1.0],
				octave_base_scale=8,
				scales_per_octave=1,
				strides=[16, 32, 64, 128]),
			loss_cls=dict(type='QualityFocalLoss', use_sigmoid=False, beta=2.0, loss_weight=1.0),
			loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
			use_dgqp = True,
			loss_bbox=dict(type='CIoULoss', loss_weight=2.0)),
		keypoint_pos='input',
		keypoint_head_cfg=dict(
			type='KeypointHead',
			num_classes=34,
			in_channels=256,
			stacked_convs=2,
			strides=[16, 32, 64, 128],
			shared_stacked_convs=0,
			logits_convs=1,
			head_types=['top_left_corner', 'bottom_right_corner', 'center'],
			corner_pooling=False,
			loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
			loss_cls=dict(type='GaussianFocalLoss', loss_weight=0.25)),
		cls_keypoint_cfg=dict(
			keypoint_types=['center'],
			with_key_score=False,
			with_relation=True),
		reg_keypoint_cfg=dict(
			keypoint_types=['top_left_corner', 'bottom_right_corner'],
			with_key_score=False,
			with_relation=True),
		keypoint_cfg=dict(max_keypoint_num=20, keypoint_score_thr=0.0),
		feature_selection_cfg=dict(
			selection_method='index',
			cross_level_topk=50,
			cross_level_selection=True),
		num_attn_heads=8,
		scale_position=False,
		pos_cfg=dict(base_size=[300, 300], log_scale=True, num_layers=2),
		shared_positional_encoding_outer=True),
	train_cfg=dict(
		bbox=dict(
			assigner=dict(type='ATSSAssigner', topk=13),
			allowed_border=-1,
			pos_weight=-1,
			debug=False),
		keypoint=dict(
			assigner=dict(type='PointKptAssigner'),
			allowed_border=-1,
			pos_weight=-1,
			debug=False)),
	test_cfg = dict(
		nms_pre=1000,
		min_bbox_size=0,
		score_thr=0.05,
		nms=dict(type='soft_nms', iou_threshold=0.6, method = 'gaussian'),
		max_per_img=100)
	)

# data setting
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
	dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=0.5),
	dict(
		type='OneOf',
		transforms=[
			dict(type='Blur', blur_limit=3, p=1.0),
			dict(type='MedianBlur', blur_limit=3, p=1.0)
		],
		p=0.1),
]

train_pipeline = [
	dict(type='LoadImageFromFile', to_float32=True),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(
		type='RandomCrop',
		crop_type='relative_range',
		crop_size=(0.9, 0.9)),
	dict(
		type='Resize',
		img_scale=(800, 800),
		multiscale_mode='range',
		keep_ratio=True),
	dict(
		type='PhotoMetricDistortion',
		brightness_delta=10,
		contrast_range=(0.9, 1.1),
		saturation_range=(0.9, 1.1),
		hue_delta=9
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
		skip_img_without_anno=True),	
	dict(type='Pad', size_divisor=800),
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
		bypass_duplicate = True,
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