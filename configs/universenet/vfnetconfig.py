_base_ = [
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
	'../_base_/swa.py'
]

# model settings
model = dict(
	type='GFL',
	pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
	backbone=dict(
		type='SwinTransformer',
		embed_dim=128,
		depths=[2, 2, 18, 2],
		num_heads=[4, 8, 16, 32],
		window_size=7,
		mlp_ratio=4.,
		qkv_bias=True,
		qk_scale=None,
		drop_rate=0.,
		attn_drop_rate=0.,
		ape=False,
		drop_path_rate=0.3,
		patch_norm=True,
		out_indices=(0, 1, 2, 3),
		use_checkpoint=True,
	),
	neck=[
		dict(
			type='PAFPNX',
			in_channels=[128, 256, 512, 1024],
			out_channels=256,
			start_level=1,
			add_extra_convs='on_output',
			num_outs=5,
			relu_before_extra_convs=True,
			pafpn_conv_cfg=dict(type='DCNv2'),
			no_norm_on_lateral=True,
			norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
		dict(
			type='SEPC',
			in_channels=[256] * 5,
			out_channels=256,
			stacked_convs=4,
			pconv_deform=True,
			lcconv_deform=True,
			ibn=True,  # please set imgs/gpu >= 4
			pnorm_eval=False,
			lcnorm_eval=False,
			lcconv_padding=1,
			pnorm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
			lcnorm_cfg=dict(type='GN', num_groups=32, requires_grad=True))
	],
	bbox_head=dict(
		type='VFNetHead',
		num_classes=24,
		in_channels=256,
		stacked_convs=0,
		feat_channels=256,
		strides=[8, 16, 32, 64, 128],
		regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)),
		anchor_generator=dict(
			type='AnchorGenerator',
			ratios=[1.0],
			octave_base_scale=8,
			scales_per_octave=1,
			center_offset=0.0,
			strides=[8, 16, 32, 64, 128]),
		center_sampling=False,
		dcn_on_last_conv=True,
		use_atss=True,
		use_vfl=True,
		loss_cls=dict(
			type='VarifocalLoss',
			use_sigmoid=True,
			alpha=0.75,
			gamma=2.0,
			iou_weighted=True,
			loss_weight=1.0),
		loss_bbox=dict(type='CIoULoss', loss_weight=1.5),
		loss_bbox_refine=dict(type='CIoULoss', loss_weight=2.0),
		norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
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
		nms=dict(type='nms', iou_threshold=0.7),
		max_per_img=100))

# data setting
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
	dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=0.5),
	dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.2),
	dict(
		type='OneOf',
		transforms=[
			dict(
				type='RGBShift',
				r_shift_limit=10,
				g_shift_limit=10,
				b_shift_limit=10,
				p=1.0),
			dict(
				type='HueSaturationValue',
				hue_shift_limit=20,
				sat_shift_limit=30,
				val_shift_limit=20,
				p=1.0)
		],
		p=0.1),
	dict(type='ImageCompression', quality_lower=85, quality_upper=95, p=0.2),
	dict(type='ChannelShuffle', p=0.1),
	dict(
		type='OneOf',
		transforms=[
			dict(type='Blur', blur_limit=3, p=1.0),
			dict(type='MedianBlur', blur_limit=3, p=1.0)
		],
		p=0.1),
]

train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(
		type='RandomCrop',
		crop_type='relative_range',
		crop_size=(0.8, 0.8)),
	dict(
		type='Resize',
		img_scale=[(640, 640), (800, 800)],
		multiscale_mode='range',
		keep_ratio=True),
	dict(
		type='CutOut',
		n_holes=(5, 10),
		cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
					  (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32),
					  (32, 48), (48, 32), (48, 48)]),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Pad', size_divisor=32),
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
	samples_per_gpu=12,
	workers_per_gpu=4,
	train=dict(type = dataset_type,
		ann_file = data_root + '/annotations/instances_train2017.json',
		img_prefix = 'train_images/',
		pipeline=train_pipeline),
	val=dict(type = dataset_type,
		ann_file = data_root + '/annotations/instances_val2017.json',
		img_prefix = 'val_images/',
		pipeline=test_pipeline,
		samples_per_gpu = 24),
	test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0., custom_keys={'absolute_pos_embed': dict(decay_mult=0.), 'relative_position_bias_table': dict(decay_mult=0.), 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip = None)
# learning policy
lr_config = dict(_delete_=True,
	policy='CosineAnnealing',
	min_lr_ratio = 0.12,
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=0.1,
	)
runner = dict(type='IterBasedRunner', max_iters=480000, max_epochs = None)

checkpoint_config = dict(interval = 5000)
evaluation = dict(interval = 5000, metric = 'bbox')

fp16 = dict(loss_scale=512.)

# runtime
resume_from = None
workflow = [('train', 1)]
