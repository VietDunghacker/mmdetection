_base_ = [
	'../_base_/datasets/coco_detection.py',
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
]

# model settings
max_per_img = 100
model = dict(
	type='CenterNet',
	pretrained = 'https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth',
	backbone=dict(
		type='HRNet',
		extra=dict(
			stage1=dict(
				num_modules=1,
				num_branches=1,
				block='BOTTLENECK',
				num_blocks=(4, ),
				num_channels=(64, )),
			stage2=dict(
				num_modules=1,
				num_branches=2,
				block='BASIC',
				num_blocks=(4, 4),
				num_channels=(48, 96)),
			stage3=dict(
				num_modules=4,
				num_branches=3,
				block='BASIC',
				num_blocks=(4, 4, 4),
				num_channels=(48, 96, 192)),
			stage4=dict(
				num_modules=3,
				num_branches=4,
				block='BASIC',
				num_blocks=(4, 4, 4, 4),
				num_channels=(48, 96, 192, 384))),
		norm_eval = False,
		with_cp = True,),
	neck=dict(
		type='FCNHead',
		in_channels=[48, 96, 192, 384],
		in_index=(0, 1, 2, 3),
		channels=256,
		input_transform='resize_concat',
		kernel_size=1,
		num_convs=1,
		concat_input=False,
		dropout_ratio=-1,
		norm_cfg=dict(type='BN', requires_grad=True),
		align_corners=False),
	bbox_head=dict(
		type='CentripetalHead',
		num_classes=34,
		in_channels=256,
		num_feat_levels=1,
		corner_emb_channels=0,
		loss_heatmap=dict(type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
		loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1),
		loss_guiding_shift=dict(type='SmoothL1Loss', beta=1.0, loss_weight=0.05 * 511 / 800),
		loss_centripetal_shift=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)),
	train_cfg=None,
	test_cfg=dict(
		corner_topk=max_per_img,
		local_maximum_kernel=3,
		distance_threshold=0.5,
        score_thr=0,
        max_per_img=max_per_img,
        nms=dict(type='soft_nms', iou_threshold=0.6, method='gaussian')))

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
	dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
	dict(
		type='RandomCenterCropPad',
		crop_size=(800, 800),
		ratios=(0.9, 0.925, 0.95, 0.975, 1.0, 1.05, 1.1, 1.15, 1.2),
		border = 380,
		mean=[0, 0, 0],
		std=[1, 1, 1],
		to_rgb=True,
		test_pad_mode=None),
	dict(type='Resize', img_scale=(800, 800), keep_ratio=True, override = True),
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
		skip_img_without_anno=True),	
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