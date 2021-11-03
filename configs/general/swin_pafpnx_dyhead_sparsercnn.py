_base_ = [
	'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
num_stages = 6
num_proposals = 128
model = dict(
	type='SparseRCNN',
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
		init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth')),
	neck=[
		dict(
			type='PAFPNX',
			in_channels=[128, 256, 512, 1024],
			out_channels=256,
			start_level=0,
			add_extra_convs='on_output',
			num_outs=4,
			relu_before_extra_convs=True,
			pafpn_conv_cfg=dict(type='DCNv2'),
			norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
		),
		dict(
			type='DyHead',
			in_channels=256,
			out_channels=256,
			num_convs=6,
			with_cp=True)
	],
	rpn_head=dict(
		type='EmbeddingRPNHead',
		num_proposals=num_proposals,
		proposal_feature_channel=256),
	roi_head=dict(
		type='SparseRoIHead',
		num_stages=num_stages,
		stage_loss_weights=[1] * num_stages,
		proposal_feature_channel=256,
		bbox_roi_extractor=dict(
			type='SingleRoIExtractor',
			roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
			out_channels=256,
			featmap_strides=[4, 8, 16, 32]),
		bbox_head=[
			dict(
				type='DIIHead',
				num_classes=37,
				num_ffn_fcs=2,
				num_heads=8,
				num_cls_fcs=1,
				num_reg_fcs=3,
				feedforward_channels=2048,
				in_channels=256,
				dropout=0.0,
				ffn_act_cfg=dict(type='ReLU', inplace=True),
				dynamic_conv_cfg=dict(
					type='DynamicConv',
					in_channels=256,
					feat_channels=64,
					out_channels=256,
					input_feat_shape=7,
					act_cfg=dict(type='ReLU', inplace=True),
					norm_cfg=dict(type='LN')),
				loss_bbox=dict(type='SmoothL1Loss', beta=0.01, loss_weight=5.0),
				loss_iou=dict(type='CIoULoss', loss_weight=2.0),
				loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
				bbox_coder=dict(
					type='DeltaXYWHBBoxCoder',
					clip_border=False,
					target_means=[0., 0., 0., 0.],
					target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
		]),
	# training and testing settings
	train_cfg=dict(
		rpn=None,
		rcnn=[
			dict(
				assigner=dict(
					type='HungarianAssigner',
					cls_cost=dict(type='FocalLossCost', weight=2.0),
					reg_cost=dict(type='BBoxL1Cost', smooth=True, beta=0.01, weight=5.0),
					iou_cost=dict(type='IoUCost', iou_mode='ciou', weight=2.0)),
				sampler=dict(type='PseudoSampler'),
				pos_weight=1) for _ in range(num_stages)
		]),
	test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals, score_threshold = 0.05, nms = dict(type='nms', iou_threshold=0.6))))

# data setting
dataset_type = 'CocoDataset'
data_root = '/content/data/'
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
	lr=0.000025,
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