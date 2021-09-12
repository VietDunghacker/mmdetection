_base_ = [
	'../_base_/datasets/coco_detection.py', '../_base_/default_runtime.py'
]
model = dict(
	type='DeformableDETR',
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
		init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth')),
	neck=dict(
		type='ChannelMapper',
		in_channels=[256, 512, 1024],
		kernel_size=1,
		out_channels=256,
		act_cfg=None,
		norm_cfg=dict(type='GN', num_groups=32),
		num_outs=4),
	bbox_head=dict(
		type='DeformableDETRHead',
		num_query=512,
		num_classes=1,
		in_channels=256,
		as_two_stage=True,
		with_box_refine=True,
		transformer=dict(
			type='DeformableDetrTransformer',
			encoder=dict(
				type='DetrTransformerEncoder',
				num_layers=6,
				transformerlayers=dict(
					type='BaseTransformerLayer',
					attn_cfgs=dict(
						type='MultiScaleDeformableAttention', embed_dims=256),
					feedforward_channels=1024,
					ffn_dropout=0.1,
					operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
			decoder=dict(
				type='DeformableDetrTransformerDecoder',
				num_layers=6,
				return_intermediate=True,
				transformerlayers=dict(
					type='DetrTransformerDecoderLayer',
					attn_cfgs=[
						dict(
							type='MultiheadAttention',
							embed_dims=256,
							num_heads=8,
							dropout=0.1),
						dict(
							type='MultiScaleDeformableAttention',
							embed_dims=256)
					],
					feedforward_channels=1024,
					ffn_dropout=0.1,
					operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
									 'ffn', 'norm')))),
		positional_encoding=dict(
			type='SinePositionalEncoding',
			num_feats=128,
			normalize=True,
			offset=-0.5),
		loss_cls=dict(
			type='CrossEntropyLoss',
			bg_cls_weight=0.1,
			use_sigmoid=False,
			loss_weight=2.0,
			class_weight=1.0),
		loss_bbox=dict(type='L1Loss', loss_weight=5.0),
		loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
	# training and testing settings
	train_cfg=dict(
		assigner=dict(
			type='HungarianAssigner',
			cls_cost=dict(type='ClassificationCost', weight=2.),
			reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
			iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
	test_cfg=dict(
		max_per_img=32,
		nms_max_per_img = 32,
		nms = dict(type='soft_nms', iou_threshold=0.6)))

# data setting
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
	dict(type='Normalize', **img_norm_cfg),
	dict(type='Pad', size_divisor=1),
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
optimizer = dict(
	type='AdamW',
	lr=0.0001,
	betas=(0.9, 0.999),
	weight_decay=0.0001,
	paramwise_cfg=dict(
		custom_keys={
			'absolute_pos_embed': dict(decay_mult=0.),
			'relative_position_bias_table': dict(decay_mult=0.),
			'norm': dict(decay_mult=0.),
			'sampling_offsets': dict(lr_mult=0.1),
			'reference_points': dict(lr_mult=0.1)}))
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