_base_ = [
	'../_base_/default_runtime.py'
]
max_per_img = 32
model = dict(
	type='DETR',
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
		out_indices=(3, ),
		with_cp=True,
		init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth')),
	bbox_head=dict(
		type='DETRHead',
		num_classes=1,
		num_query = max_per_img,
		in_channels=1024,
		transformer=dict(
			type='Transformer',
			encoder=dict(
				type='DetrTransformerEncoder',
				num_layers=6,
				transformerlayers=dict(
					type='BaseTransformerLayer',
					attn_cfgs=[
						dict(
							type='MultiheadAttention',
							embed_dims=256,
							num_heads=8,
							dropout=0.1)
					],
					feedforward_channels=2048,
					ffn_dropout=0.1,
					operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
			decoder=dict(
				type='DetrTransformerDecoder',
				return_intermediate=True,
				num_layers=6,
				transformerlayers=dict(
					type='DetrTransformerDecoderLayer',
					attn_cfgs=dict(
						type='MultiheadAttention',
						embed_dims=256,
						num_heads=8,
						dropout=0.1),
					feedforward_channels=2048,
					ffn_dropout=0.1,
					operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
			)),
		positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
 	 	loss_cls=dict(type='CrossEntropyLoss', bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0),
		loss_bbox=dict(type='L1Loss', loss_weight=5.0),
		loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
	# training and testing settings
	train_cfg=dict(
		assigner=dict(
			type='HungarianAssigner',
			cls_cost=dict(type='ClassificationCost', weight=1.),
			reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
			iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
	test_cfg=dict(
		max_per_img=max_per_img,
		score_threshold = 0.05,
		nms = dict(type='nms', iou_threshold=0.6)))

# data setting
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
				dict(type='Mosaic', center_ratio_range=(0.9, 1.1), img_scale=(720, 720), pad_val=0.0),
				dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
			],
			[
				dict(type='Mosaic', center_ratio_range=(0.8, 1.2), img_scale=(720, 720), pad_val=0.0),
				dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
			],
			[
				dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.9, 0.9), allow_negative_crop = True),
				dict(type='Resize', img_scale=[(480, 480), (800, 800)], multiscale_mode='range', keep_ratio=True),
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
	type='AdamW',
	lr=0.0001,
	betas=(0.9, 0.999),
	weight_decay=0.0001,
	paramwise_cfg=dict(
		custom_keys={
			'absolute_pos_embed': dict(decay_mult=0.),
			'relative_position_bias_table': dict(decay_mult=0.),
			'norm': dict(decay_mult=0.),}))
optimizer_config = dict(grad_clip = None)
# learning policy
lr_config = dict(
	policy='CosineAnnealing',
	min_lr_ratio = 0.12,
	warmup='linear',
	warmup_iters=500,
	warmup_ratio=1.0/3,
	)
runner = dict(type='IterBasedRunner', max_iters=15000, max_epochs = None)

checkpoint_config = dict(interval = 500)
evaluation = dict(interval = 500, metric = 'bbox')

fp16 = dict(loss_scale=512.)

# runtime
resume_from = None
workflow = [('train', 1)]