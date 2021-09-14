_base_ = [
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
	type='RepPointsV2Detector',
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
		out_indices=(0, 1, 2, 3),
		with_cp=True,
		init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth')),
	neck=[
		dict(
			type='PAFPNX',
			in_channels=[128, 256, 512, 1024],
			out_channels=256,
			start_level=1,
			add_extra_convs='on_input',
			num_outs=5,
			relu_before_extra_convs=True,
			pafpn_conv_cfg=dict(type='DCNv2'),
			norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
		dict(
			type='SEPC',
			in_channels=[256] * 5,
			out_channels=256,
			stacked_convs=3,
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
		type='RepPointsV2Head',
		num_classes=34,
		in_channels=256,
		feat_channels=256,
		point_feat_channels=256,
		stacked_convs=0,
		shared_stacked_convs=1,
		first_kernel_size=3,
		kernel_size=1,
		corner_dim=64,
		num_points=9,
		gradient_mul=0.1,
		point_strides=[8, 16, 32, 64, 128],
		point_base_scale=4,
		norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
		loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
		loss_bbox_init=dict(type='CIoULoss', loss_weight=1.0),
		loss_bbox_refine=dict(type='CIoULoss', loss_weight = 2.0),
		loss_heatmap=dict(type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=0.25),
		loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
		loss_sem=dict(type='SEPFocalLoss', gamma=2.0, alpha=0.25, loss_weight=0.1),
		transform_method='exact_minmax'),
	train_cfg = dict(
		init=dict(
			assigner=dict(type='PointAssignerV2', scale=4, pos_num=1),
			allowed_border=-1,
			pos_weight=-1,
			debug=False),
		heatmap=dict(
			assigner=dict(type='PointHMAssigner', gaussian_bump=True, gaussian_iou=0.7),
			allowed_border=-1,
			pos_weight=-1,
			debug=False),
		refine=dict(
			assigner=dict(type='ATSSAssigner', topk=9),
			allowed_border=-1,
			pos_weight=-1,
			debug=False)),
	test_cfg = dict(
		nms_pre=1000,
		min_bbox_size=0,
		score_thr=0.05,
		nms=dict(type='soft_nms', iou_thr=0.6, method = 'gaussian'),
		max_per_img=100)
	)

# data setting
dataset_type = 'CocoDataset'
data_root = '/content/data/'
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
		crop_size=(0.9, 0.9)),
	dict(
		type='Resize',
		img_scale=[(640, 640), (960, 960)],
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
	dict(type='LoadRPDV2Annotations', num_classes=34),
	dict(type='RPDV2FormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights']),
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
optimizer = dict(
	type='AdamW',
	lr=0.0001,
	betas=(0.9, 0.999),
	weight_decay=0.0001,
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
	warmup_ratio=0.1,
	)
runner = dict(type='IterBasedRunner', max_iters=10000, max_epochs = None)
checkpoint_config = dict(interval = 100)
evaluation = dict(interval = 100, metric = 'bbox')

fp16 = dict(loss_scale = 512.)

# runtime
load_from = None
resume_from = None
workflow = [('train', 1)]