import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
						images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.models.utils import BRPool, TLPool
from mmcv.ops.nms import batched_nms
from mmcv.ops import DeformConv2d, ModulatedDeformConv2dPack
from mmcv.runner import force_fp32
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from .corner_head import BiCornerPool

class DCNConvModule(nn.Module):
	def __init__(
		self,
		in_channels=256,
		out_channels=256,
		kernel_size=3,
		dilation=1,
		num_groups=1,
		dcn_pad=1
	):
		super(DCNConvModule, self).__init__()

		self.conv = ModulatedDeformConv2dPack(in_channels, out_channels, kernel_size, 1, dcn_pad)
		self.bn = nn.GroupNorm(num_groups, out_channels)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.relu(self.bn(self.conv(x)))
		return x


@HEADS.register_module()
class PyCenterNetHead(AnchorFreeHead):
	def __init__(self,
				 num_classes,
				 in_channels,
				 point_feat_channels=256,
				 shared_stacked_convs=1,
				 first_kernel_size=3,
				 kernel_size=1,
				 corner_dim=64,
				 num_points=9,
				 gradient_mul=0.1,
				 conv_module_style='norm', #norm or dcn, norm is faster
				 point_strides=[8, 16, 32, 64, 128],
				 point_base_scale=4,
				 background_label=None,
				 loss_cls=dict(type='FocalLoss',
					 use_sigmoid=True,
					 gamma=2.0,
					 alpha=0.25,
					 loss_weight=1.0),
				 loss_bbox_init=dict(
					 type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
				 loss_bbox_refine=dict(
					 type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
				 loss_heatmap=dict(
					 type='GaussianFocalLoss',
					 alpha=2.0,
					 gamma=4.0,
					 loss_weight=0.25),
				 loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
				 loss_sem=dict(type='SEPFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, 
							   loss_weight=0.1),
				 **kwargs):

		self.num_points = num_points
		self.point_feat_channels = point_feat_channels
		self.shared_stacked_convs = shared_stacked_convs
		self.first_kernel_size = first_kernel_size
		self.kernel_size = kernel_size
		self.corner_dim = corner_dim
		self.conv_module_style = conv_module_style

		self.background_label = (
			  num_classes if background_label is None else background_label)
		# background_label should be either 0 or num_classes
		assert(self.background_label == 0
			   or self.background_label == num_classes)

		# we use deformable conv to extract points features
		self.dcn_kernel = int(np.sqrt(num_points))
		self.dcn_pad = int((self.dcn_kernel - 1) / 2)
		assert self.dcn_kernel * self.dcn_kernel == num_points, \
			'The points number should be a square number.'
		assert self.dcn_kernel % 2 == 1, \
			'The points number should be an odd square number.'
		dcn_base = np.arange(-self.dcn_pad,
							 self.dcn_pad + 1).astype(np.float64)
		dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
		dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
		dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
		self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

		super().__init__(num_classes, in_channels, **kwargs)

		self.gradient_mul	 = gradient_mul
		self.point_base_scale = point_base_scale
		self.point_strides	= point_strides
		self.point_generators = [PointGenerator() for _ in self.point_strides]

		if self.train_cfg:
			self.init_assigner = build_assigner(self.train_cfg.init.assigner)
			self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)
			self.hm_assigner = build_assigner(self.train_cfg.heatmap.assigner)
			# use PseudoSampler when sampling is False
			sampler_cfg = dict(type='PseudoSampler')
			self.sampler = build_sampler(sampler_cfg, context=self)

		self.cls_out_channels = self.num_classes

		self.loss_tl_cls		 = build_loss(loss_cls)
		self.loss_br_cls		 = build_loss(loss_cls)
		self.loss_tl_bbox_init   = build_loss(loss_bbox_init)
		self.loss_tl_bbox_refine = build_loss(loss_bbox_refine)
		self.loss_br_bbox_init   = build_loss(loss_bbox_init)
		self.loss_br_bbox_refine = build_loss(loss_bbox_refine)
		self.loss_heatmap = build_loss(loss_heatmap)
		self.loss_offset = build_loss(loss_offset)
		self.loss_sem = build_loss(loss_sem)

	def _init_layers(self):
		"""Initialize layers of the head."""
		self.relu = nn.ReLU(inplace=True)
		self.tl_cls_convs = nn.ModuleList()
		self.br_cls_convs = nn.ModuleList()
		self.tl_reg_convs = nn.ModuleList()
		self.br_reg_convs = nn.ModuleList()
		self.shared_convs = nn.ModuleList()
		for i in range(self.stacked_convs):
			chn = self.in_channels if i == 0 else self.feat_channels
			if self.conv_module_style == 'norm':
				self.tl_cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
										conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
				self.br_cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
										conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
				self.tl_reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
										conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
				self.br_reg_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
										conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
			else:
				self.tl_cls_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1, self.norm_cfg.num_groups, self.dcn_pad))
				self.br_cls_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1, self.norm_cfg.num_groups, self.dcn_pad))
				self.tl_reg_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1, self.norm_cfg.num_groups, self.dcn_pad))
				self.br_reg_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1, self.norm_cfg.num_groups, self.dcn_pad))

		for i in range(self.shared_stacked_convs):
			self.shared_convs.append(ConvModule(self.feat_channels, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))


		self.hem_tl = TLPool(self.feat_channels, self.conv_cfg, self.norm_cfg, first_kernel_size=self.first_kernel_size, kernel_size=self.kernel_size, corner_dim=self.corner_dim)
		self.hem_br = BRPool(self.feat_channels, self.conv_cfg, self.norm_cfg,  first_kernel_size=self.first_kernel_size, kernel_size=self.kernel_size, corner_dim=self.corner_dim)
		self.hem_ct = ConvModule(self.feat_channels, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

		pts_out_dim = 2 * self.num_points

		cls_in_channels = self.feat_channels + 9
		self.reppoints_tl_cls_conv = DeformConv2d(cls_in_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
		self.reppoints_tl_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)

		self.reppoints_br_cls_conv = DeformConv2d(cls_in_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
		self.reppoints_br_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)

		self.reppoints_tl_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
		self.reppoints_tl_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)
		
		self.reppoints_br_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
		self.reppoints_br_pts_init_out = nn.Conv2d(self.point_feat_channels,
												  pts_out_dim, 1, 1, 0)
		pts_in_channels = self.feat_channels + 9
		self.reppoints_tl_pts_refine_conv = DeformConv2d(pts_in_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
		self.reppoints_tl_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

		self.reppoints_br_pts_refine_conv = DeformConv2d(pts_in_channels, self.point_feat_channels, self.dcn_kernel, 1, self.dcn_pad)
		self.reppoints_br_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

		self.reppoints_hem_tl_score_out = nn.Conv2d(self.feat_channels, 1, 3, 1, 1)
		self.reppoints_hem_br_score_out = nn.Conv2d(self.feat_channels, 1, 3, 1, 1)
		self.reppoints_hem_ct_score_out = nn.Conv2d(self.feat_channels, 1, 3, 1, 1)
		self.reppoints_hem_tl_offset_out = nn.Conv2d(self.feat_channels, 2, 3, 1, 1)
		self.reppoints_hem_br_offset_out = nn.Conv2d(self.feat_channels, 2, 3, 1, 1)
		self.reppoints_hem_ct_offset_out = nn.Conv2d(self.feat_channels, 2, 3, 1, 1)

		self.reppoints_sem_out = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1, 1, 0)
		self.reppoints_sem_embedding = ConvModule(self.feat_channels, self.feat_channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg)

	def init_weights(self):
		"""Initialize weights of the head."""
		for m in self.tl_cls_convs:
			normal_init(m.conv, std=0.01)
		for m in self.br_cls_convs:
			normal_init(m.conv, std=0.01)
		for m in self.tl_reg_convs:
			normal_init(m.conv, std=0.01)
		for m in self.br_reg_convs:
			normal_init(m.conv, std=0.01)
		for m in self.shared_convs:
			normal_init(m.conv, std=0.01)
		bias_cls = bias_init_with_prob(0.01)
		normal_init(self.reppoints_tl_cls_conv, std=0.01)
		normal_init(self.reppoints_tl_cls_out, std=0.01, bias=bias_cls)
		normal_init(self.reppoints_br_cls_conv, std=0.01)
		normal_init(self.reppoints_br_cls_out, std=0.01, bias=bias_cls)

		normal_init(self.reppoints_tl_pts_init_conv, std=0.01)
		normal_init(self.reppoints_tl_pts_init_out, std=0.01)
		normal_init(self.reppoints_br_pts_init_conv, std=0.01)
		normal_init(self.reppoints_br_pts_init_out, std=0.01)
		normal_init(self.reppoints_tl_pts_refine_conv, std=0.01)
		normal_init(self.reppoints_tl_pts_refine_out, std=0.01)
		normal_init(self.reppoints_br_pts_refine_conv, std=0.01)
		normal_init(self.reppoints_br_pts_refine_out, std=0.01)
		normal_init(self.reppoints_hem_tl_score_out, std=0.01, bias=bias_cls)
		normal_init(self.reppoints_hem_tl_offset_out, std=0.01)
		normal_init(self.reppoints_hem_br_score_out, std=0.01, bias=bias_cls)
		normal_init(self.reppoints_hem_br_offset_out, std=0.01)
		normal_init(self.reppoints_hem_ct_score_out, std=0.01, bias=bias_cls)
		normal_init(self.reppoints_hem_ct_offset_out, std=0.01)
		normal_init(self.reppoints_sem_out, std=0.01, bias=bias_cls)

	def points2bbox(self, pts, y_first=True):
		"""Converting the points set into bounding box.
		:param pts: the input points sets (fields), each points
			set (fields) is represented as 2n scalar.
		:param y_first: if y_fisrt=True, the point set is represented as
			[y1, x1, y2, x2 ... yn, xn], otherwise the point set is
			represented as [x1, y1, x2, y2 ... xn, yn].
		:return: each points set is converting to a bbox [x1, y1, x2, y2].
		"""
		pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
		pts_reshape = pts_reshape[:, :2, ...]
		pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
		pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
		bbox_left = pts_x[:, 0:1, ...]
		bbox_right = pts_x[:, 1:2, ...]
		bbox_up = pts_y[:, 0:1, ...]
		bbox_bottom = pts_y[:, 1:2, ...]
		bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
		return bbox

	def forward(self, feats):
		return multi_apply(self.forward_single, feats)

	@force_fp32()
	def forward_single(self, x):
		""" Forward feature map of a single FPN level."""
		if isinstance(x, list):
			tl_cls_feat = x[0]
			br_cls_feat = x[0]
			tl_pts_feat = x[1]
			br_pts_feat = x[1]
			dcn_base_offset = self.dcn_base_offset.type_as(x[0])
		else:
			tl_cls_feat = x
			br_cls_feat = x
			tl_pts_feat = x
			br_pts_feat = x

			dcn_base_offset = self.dcn_base_offset.type_as(x)
		
		for tl_cls_conv in self.tl_cls_convs:
			tl_cls_feat = tl_cls_conv(tl_cls_feat)
		for br_cls_conv in self.br_cls_convs:
			br_cls_feat = br_cls_conv(br_cls_feat)
		for tl_reg_conv in self.tl_reg_convs:
			tl_pts_feat = tl_reg_conv(tl_pts_feat)
		for br_reg_conv in self.br_reg_convs:
			br_pts_feat = br_reg_conv(br_pts_feat)

		shared_feat = tl_pts_feat + br_pts_feat
		for shared_conv in self.shared_convs:
			shared_feat = shared_conv(shared_feat)

		sem_feat = shared_feat
		hem_feat = shared_feat

		sem_scores_out = self.reppoints_sem_out(sem_feat)
		sem_feat = self.reppoints_sem_embedding(sem_feat)

		tl_cls_feat = tl_cls_feat + sem_feat
		br_cls_feat = br_cls_feat + sem_feat
		tl_pts_feat = tl_pts_feat + sem_feat
		br_pts_feat = br_pts_feat + sem_feat
		hem_feat	= hem_feat + sem_feat

		# generate heatmap and offset
		hem_tl_feat = self.hem_tl(hem_feat)
		hem_br_feat = self.hem_br(hem_feat)
		hem_ct_feat = self.hem_ct(hem_feat)

		hem_tl_score_out = self.reppoints_hem_tl_score_out(hem_tl_feat)
		hem_tl_offset_out = self.reppoints_hem_tl_offset_out(hem_tl_feat)
		hem_br_score_out = self.reppoints_hem_br_score_out(hem_br_feat)
		hem_br_offset_out = self.reppoints_hem_br_offset_out(hem_br_feat)
		hem_ct_score_out = self.reppoints_hem_ct_score_out(hem_ct_feat)
		hem_ct_offset_out = self.reppoints_hem_ct_offset_out(hem_ct_feat)

		hem_score_out = torch.cat([hem_tl_score_out, hem_br_score_out, hem_ct_score_out], dim=1)
		hem_offset_out = torch.cat([hem_tl_offset_out, hem_br_offset_out, hem_ct_offset_out], dim=1)

		# initialize 
		pts_tl_out_init = self.reppoints_tl_pts_init_out(self.relu(self.reppoints_tl_pts_init_conv(tl_pts_feat)))
		pts_br_out_init = self.reppoints_br_pts_init_out(self.relu(self.reppoints_br_pts_init_conv(br_pts_feat)))
		# refine and classify
		pts_tl_out_init_grad_mul = (1 - self.gradient_mul) * pts_tl_out_init.detach() + self.gradient_mul * pts_tl_out_init
		pts_br_out_init_grad_mul = (1 - self.gradient_mul) * pts_br_out_init.detach() + self.gradient_mul * pts_br_out_init
		dcn_tl_offset = pts_tl_out_init_grad_mul - dcn_base_offset
		dcn_br_offset = pts_br_out_init_grad_mul - dcn_base_offset

		hem_feat = torch.cat([hem_score_out, hem_offset_out], dim=1)
		tl_cls_feat = torch.cat([tl_cls_feat, hem_feat], dim=1)
		br_cls_feat = torch.cat([br_cls_feat, hem_feat], dim=1)
		tl_pts_feat = torch.cat([tl_pts_feat, hem_feat], dim=1)
		br_pts_feat = torch.cat([br_pts_feat, hem_feat], dim=1)

		tl_cls_out = self.reppoints_tl_cls_out(self.relu(self.reppoints_tl_cls_conv(tl_cls_feat, dcn_tl_offset)))
		br_cls_out = self.reppoints_br_cls_out(self.relu(self.reppoints_br_cls_conv(br_cls_feat, dcn_br_offset)))
		pts_tl_out_refine = self.reppoints_tl_pts_refine_out(self.relu(self.reppoints_tl_pts_refine_conv(tl_pts_feat, dcn_tl_offset)))
		pts_br_out_refine = self.reppoints_br_pts_refine_out(self.relu(self.reppoints_br_pts_refine_conv(br_pts_feat, dcn_br_offset)))

		return (tl_cls_out, br_cls_out, pts_tl_out_init, pts_br_out_init, pts_tl_out_refine, pts_br_out_refine, hem_score_out, hem_offset_out, sem_scores_out)

	def get_points(self, featmap_sizes, img_metas):
		"""Get points according to feature map sizes.
		Args:
			featmap_sizes (list[tuple]): Multi-level feature map sizes.
			img_metas (list[dict]): Image meta info.
		Returns:
			tuple: points of each image, valid flags of each image
		"""
		num_imgs = len(img_metas)
		num_levels = len(featmap_sizes)

		# since feature map sizes of all images are the same, we only compute
		# points center for one time
		multi_level_points = []
		for i in range(num_levels):
			points = self.point_generators[i].grid_points(featmap_sizes[i], self.point_strides[i])
			multi_level_points.append(points)
		points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]

		# for each image, we compute valid flags of multi level grids
		valid_flag_list = []
		for img_id, img_meta in enumerate(img_metas):
			multi_level_flags = []
			for i in range(num_levels):
				point_stride = self.point_strides[i]
				feat_h, feat_w = featmap_sizes[i]
				h, w = img_meta['pad_shape'][:2]
				valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
				valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
				flags = self.point_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
				multi_level_flags.append(flags)
			valid_flag_list.append(multi_level_flags)

		return points_list, valid_flag_list

	def offset_to_pts(self, center_list, pred_list):
		"""Change from point offset to point coordinate."""
		pts_list = []
		for i_lvl in range(len(self.point_strides)):
			pts_lvl = []
			for i_img in range(len(center_list)):
				pts_center = center_list[i_img][i_lvl][:, :2].repeat(
					1, self.num_points)
				pts_shift = pred_list[i_lvl][i_img]
				yx_pts_shift = pts_shift.permute(1, 2, 0).view(
					-1, 2 * self.num_points)
				y_pts_shift = yx_pts_shift[..., 0::2]
				x_pts_shift = yx_pts_shift[..., 1::2]
				xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
				xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
				pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
				pts_lvl.append(pts)
			pts_lvl = torch.stack(pts_lvl, 0)
			pts_list.append(pts_lvl)
		return pts_list

	def _point_target_single(self,
							 flat_proposals,
							 valid_flags,
							 num_level_proposals,
							 gt_bboxes,
							 gt_bboxes_ignore,
							 gt_labels,
							 label_channels=1,
							 stage='init',
							 unmap_outputs=True):
		inside_flags = valid_flags
		if not inside_flags.any():
			return (None, ) * 6
		# assign gt and sample proposals
		proposals = flat_proposals[inside_flags, :]

		num_level_proposals_inside = self.get_num_level_proposals_inside(num_level_proposals, 
																		 inside_flags)
		if stage == 'init':
			assigner = self.init_assigner
			assigner_type = self.train_cfg.init.assigner.type
			pos_weight = self.train_cfg.init.pos_weight
		else:
			assigner = self.refine_assigner
			assigner_type = self.train_cfg.refine.assigner.type
			pos_weight = self.train_cfg.refine.pos_weight
		if assigner_type != "ATSSAssigner":
			assign_result = assigner.assign(proposals, gt_bboxes, gt_bboxes_ignore, gt_labels)
		else:
			assign_result = assigner.assign(proposals, num_level_proposals_inside, gt_bboxes, 
											gt_bboxes_ignore, gt_labels)
		sampling_result = self.sampler.sample(assign_result, proposals, gt_bboxes)

		num_valid_proposals = proposals.shape[0]
		bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
		bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
		labels = proposals.new_full((num_valid_proposals, ), self.background_label, dtype=torch.long)
		label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

		pos_inds = sampling_result.pos_inds
		neg_inds = sampling_result.neg_inds
		if len(pos_inds) > 0:
			pos_gt_bboxes = sampling_result.pos_gt_bboxes
			bbox_gt[pos_inds, :] = pos_gt_bboxes
			bbox_weights[pos_inds, :] = 1.0
			if gt_labels is None:
				labels[pos_inds] = 1
			else:
				labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
			if pos_weight <= 0:
				label_weights[pos_inds] = 1.0
			else:
				label_weights[pos_inds] = pos_weight
		if len(neg_inds) > 0:
			label_weights[neg_inds] = 1.0

		# map up to original set of proposals
		if unmap_outputs:
			num_total_proposals = flat_proposals.size(0)
			labels = unmap(labels, num_total_proposals, inside_flags)
			label_weights = unmap(label_weights, num_total_proposals, inside_flags)
			bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
			bbox_weights = unmap(bbox_weights, num_total_proposals, inside_flags)

		return labels, label_weights, bbox_gt, bbox_weights, pos_inds, neg_inds

	def get_targets(self,
					proposals_list,
					valid_flag_list,
					gt_bboxes_list,
					img_metas,
					gt_bboxes_ignore_list=None,
					gt_labels_list=None,
					stage='init',
					label_channels=1,
					unmap_outputs=True):
		"""Compute corresponding GT box and classification targets for
		proposals.
		Args:
			proposals_list (list[list]): Multi level points/bboxes of each
				image.
			valid_flag_list (list[list]): Multi level valid flags of each
				image.
			gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
			img_metas (list[dict]): Meta info of each image.
			gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
				ignored.
			gt_bboxes_list (list[Tensor]): Ground truth labels of each box.
			stage (str): `init` or `refine`. Generate target for init stage or
				refine stage
			label_channels (int): Channel of label.
			unmap_outputs (bool): Whether to map outputs back to the original
				set of anchors.
		Returns:
			tuple:
				- labels_list (list[Tensor]): Labels of each level.
				- label_weights_list (list[Tensor]): Label weights of each level.  # noqa: E501
				- bbox_gt_list (list[Tensor]): Ground truth bbox of each level.
				- proposal_list (list[Tensor]): Proposals(points/bboxes) of each level.  # noqa: E501
				- proposal_weights_list (list[Tensor]): Proposal weights of each level.  # noqa: E501
				- num_total_pos (int): Number of positive samples in all images.  # noqa: E501
				- num_total_neg (int): Number of negative samples in all images.  # noqa: E501
		"""
		assert stage in ['init', 'refine']
		num_imgs = len(img_metas)
		assert len(proposals_list) == len(valid_flag_list) == num_imgs

		# points number of multi levels
		num_level_proposals = [points.size(0) for points in proposals_list[0]]
		num_level_proposals_list = [num_level_proposals] * num_imgs

		# concat all level points and flags to a single tensor
		for i in range(num_imgs):
			assert len(proposals_list[i]) == len(valid_flag_list[i])
			proposals_list[i] = torch.cat(proposals_list[i])
			valid_flag_list[i] = torch.cat(valid_flag_list[i])

		# compute targets for each image
		if gt_bboxes_ignore_list is None:
			gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
		if gt_labels_list is None:
			gt_labels_list = [None for _ in range(num_imgs)]
		(all_labels, all_label_weights, all_bbox_gt, all_bbox_weights,
		 pos_inds_list, neg_inds_list) = multi_apply(
			 self._point_target_single,
			 proposals_list,
			 valid_flag_list,
			 num_level_proposals_list,
			 gt_bboxes_list,
			 gt_bboxes_ignore_list,
			 gt_labels_list,
			 stage=stage,
			 label_channels=label_channels,
			 unmap_outputs=unmap_outputs)
		# no valid points
		if any([labels is None for labels in all_labels]):
			return None
		# sampled points of all images
		num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
		num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
		labels_list = images_to_levels(all_labels, num_level_proposals)
		label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
		bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals)
		bbox_weights_list = images_to_levels(all_bbox_weights,
												 num_level_proposals)
		return (labels_list, label_weights_list, bbox_gt_list, bbox_weights_list,
				num_total_pos, num_total_neg)

	def _hm_target_single(self,
						  flat_points,
						  inside_flags,
						  gt_bboxes,
						  gt_labels,
						  unmap_outputs=True):
		# assign gt and sample points
		if not inside_flags.any():
			return (None, ) * 12
		points = flat_points[inside_flags, :]

		assigner = self.hm_assigner
		gt_hm_tl, gt_offset_tl, pos_inds_tl, neg_inds_tl, \
		gt_hm_br, gt_offset_br, pos_inds_br, neg_inds_br,\
		gt_hm_ct, gt_offset_ct, pos_inds_ct, neg_inds_ct = \
			assigner.assign(points, gt_bboxes, gt_labels)

		num_valid_points = points.shape[0]
		hm_tl_weights = points.new_zeros(num_valid_points, dtype=torch.float)
		hm_br_weights = points.new_zeros(num_valid_points, dtype=torch.float)
		hm_ct_weights = points.new_zeros(num_valid_points, dtype=torch.float)
		offset_tl_weights = points.new_zeros([num_valid_points, 2], dtype=torch.float)
		offset_br_weights = points.new_zeros([num_valid_points, 2], dtype=torch.float)
		offset_ct_weights = points.new_zeros([num_valid_points, 2], dtype=torch.float)

		hm_tl_weights[pos_inds_tl] = 1.0
		hm_tl_weights[neg_inds_tl] = 1.0
		offset_tl_weights[pos_inds_tl, :] = 1.0

		hm_br_weights[pos_inds_br] = 1.0
		hm_br_weights[neg_inds_br] = 1.0
		offset_br_weights[pos_inds_br, :] = 1.0

		hm_ct_weights[pos_inds_ct] = 1.0
		hm_ct_weights[neg_inds_ct] = 1.0
		offset_ct_weights[pos_inds_ct, :] = 1.0

		# map up to original set of grids
		if unmap_outputs:
			num_total_points = flat_points.shape[0]
			gt_hm_tl = unmap(gt_hm_tl, num_total_points, inside_flags)
			gt_offset_tl = unmap(gt_offset_tl, num_total_points, inside_flags)
			hm_tl_weights = unmap(hm_tl_weights, num_total_points, inside_flags)
			offset_tl_weights = unmap(offset_tl_weights, num_total_points, inside_flags)

			gt_hm_br = unmap(gt_hm_br, num_total_points, inside_flags)
			gt_offset_br = unmap(gt_offset_br, num_total_points, inside_flags)
			hm_br_weights = unmap(hm_br_weights, num_total_points, inside_flags)
			offset_br_weights = unmap(offset_br_weights, num_total_points, inside_flags)

			gt_hm_ct = unmap(gt_hm_ct, num_total_points, inside_flags)
			gt_offset_ct = unmap(gt_offset_ct, num_total_points, inside_flags)
			hm_ct_weights = unmap(hm_ct_weights, num_total_points, inside_flags)
			offset_ct_weights = unmap(offset_ct_weights, num_total_points, inside_flags)

		return (gt_hm_tl, gt_offset_tl, hm_tl_weights, offset_tl_weights, pos_inds_tl, neg_inds_tl,
				gt_hm_br, gt_offset_br, hm_br_weights, offset_br_weights, pos_inds_br, neg_inds_br,
				gt_hm_ct, gt_offset_ct, hm_ct_weights, offset_ct_weights, pos_inds_ct, neg_inds_ct)

	def get_hm_targets(self,
					   proposals_list,
					   valid_flag_list,
					   gt_bboxes_list,
					   img_metas,
					   gt_labels_list=None,
					   unmap_outputs=True):
		"""Compute refinement and classification targets for points.
		Args:
			points_list (list[list]): Multi level points of each image.
			valid_flag_list (list[list]): Multi level valid flags of each image.
			gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
			img_metas (list[dict]): Meta info of each image.
			cfg (dict): train sample configs.
		Returns:
			tuple
		"""
		num_imgs = len(img_metas)
		assert len(proposals_list) == len(valid_flag_list) == num_imgs

		# points number of multi levels
		num_level_proposals = [points.size(0) for points in proposals_list[0]]

		# concat all level points and flags to a single tensor
		for i in range(len(proposals_list)):
			assert len(proposals_list[i]) == len(valid_flag_list[i])
			proposals_list[i] = torch.cat(proposals_list[i])
			valid_flag_list[i] = torch.cat(valid_flag_list[i])

		if gt_labels_list is None:
			gt_labels_list = [None for _ in range(num_imgs)]
		(all_gt_hm_tl, all_gt_offset_tl, all_hm_tl_weights, all_offset_tl_weights, pos_inds_tl_list, 
		 neg_inds_tl_list, all_gt_hm_br, all_gt_offset_br, all_hm_br_weights, all_offset_br_weights, 
		 pos_inds_br_list, neg_inds_br_list, all_gt_hm_ct, all_gt_offset_ct, all_hm_ct_weights, 
		 all_offset_ct_weights, pos_inds_ct_list, neg_inds_ct_list) = \
										multi_apply(self._hm_target_single,
														  proposals_list,
														  valid_flag_list,
														  gt_bboxes_list,
														  gt_labels_list,
														  unmap_outputs=unmap_outputs)
		# no valid points
		if any([gt_hm_tl is None for gt_hm_tl in all_gt_hm_tl]):
			return None
		# sampled points of all images
		num_total_pos_tl = sum([max(inds.numel(), 1) for inds in pos_inds_tl_list])
		num_total_neg_tl = sum([max(inds.numel(), 1) for inds in neg_inds_tl_list])
		num_total_pos_br = sum([max(inds.numel(), 1) for inds in pos_inds_br_list])
		num_total_neg_br = sum([max(inds.numel(), 1) for inds in neg_inds_br_list])
		num_total_pos_ct = sum([max(inds.numel(), 1) for inds in pos_inds_ct_list])
		num_total_neg_ct = sum([max(inds.numel(), 1) for inds in neg_inds_ct_list])

		gt_hm_tl_list = images_to_levels(all_gt_hm_tl, num_level_proposals)
		gt_offset_tl_list = images_to_levels(all_gt_offset_tl, num_level_proposals)
		hm_tl_weight_list = images_to_levels(all_hm_tl_weights, num_level_proposals)
		offset_tl_weight_list = images_to_levels(all_offset_tl_weights, num_level_proposals)

		gt_hm_br_list = images_to_levels(all_gt_hm_br, num_level_proposals)
		gt_offset_br_list = images_to_levels(all_gt_offset_br, num_level_proposals)
		hm_br_weight_list = images_to_levels(all_hm_br_weights, num_level_proposals)
		offset_br_weight_list = images_to_levels(all_offset_br_weights, num_level_proposals)

		gt_hm_ct_list = images_to_levels(all_gt_hm_ct, num_level_proposals)
		gt_offset_ct_list = images_to_levels(all_gt_offset_ct, num_level_proposals)
		hm_ct_weight_list = images_to_levels(all_hm_ct_weights, num_level_proposals)
		offset_ct_weight_list = images_to_levels(all_offset_ct_weights, num_level_proposals)

		return (gt_hm_tl_list, gt_offset_tl_list, hm_tl_weight_list, offset_tl_weight_list,
				gt_hm_br_list, gt_offset_br_list, hm_br_weight_list, offset_br_weight_list,
				gt_hm_ct_list, gt_offset_ct_list, hm_ct_weight_list, offset_ct_weight_list,
				num_total_pos_tl, num_total_neg_tl, num_total_pos_br, num_total_neg_br,
				num_total_pos_ct, num_total_neg_ct)

	def loss_single(self, tl_cls_score, br_cls_score, tl_pts_pred_init, br_pts_pred_init, 
					tl_pts_pred_refine, br_pts_pred_refine, hm_score, hm_offset,
					tl_labels, br_labels, tl_label_weights, br_label_weights,
					tl_bbox_gt_init, br_bbox_gt_init, tl_bbox_weights_init,
					br_bbox_weights_init, tl_bbox_gt_refine, br_bbox_gt_refine, 
					tl_bbox_weights_refine, br_bbox_weights_refine,
					gt_hm_tl, gt_offset_tl, gt_hm_tl_weight, gt_offset_tl_weight,
					gt_hm_br, gt_offset_br, gt_hm_br_weight, gt_offset_br_weight,
					gt_hm_ct, gt_offset_ct, gt_hm_ct_weight, gt_offset_ct_weight,
					stride,
					tl_num_total_samples_init, br_num_total_samples_init,
					tl_num_total_samples_refine, br_num_total_samples_refine,
					num_total_samples_tl, num_total_samples_br, num_total_samples_ct):
		# classification loss
		tl_labels = tl_labels.reshape(-1)
		br_labels = br_labels.reshape(-1)
		tl_label_weights = tl_label_weights.reshape(-1)
		br_label_weights = br_label_weights.reshape(-1)
		tl_cls_score = tl_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
		br_cls_score = br_cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

		loss_cls = 0

		loss_cls += self.loss_tl_cls(tl_cls_score, tl_labels, tl_label_weights, avg_factor=tl_num_total_samples_refine)

		loss_cls += self.loss_br_cls(br_cls_score, br_labels, br_label_weights, avg_factor=br_num_total_samples_refine)

		loss_cls /= 2.0

		loss_pts_init = 0
		loss_pts_refine = 0

		# points loss
		tl_bbox_gt_init = tl_bbox_gt_init.reshape(-1, 4)
		tl_bbox_weights_init = tl_bbox_weights_init.reshape(-1, 4)
		tl_bbox_pred_init = self.points2bbox(tl_pts_pred_init.reshape(-1, 2 * self.num_points),
											 y_first=False)
		tl_bbox_gt_refine = tl_bbox_gt_refine.reshape(-1, 4)
		tl_bbox_weights_refine = tl_bbox_weights_refine.reshape(-1, 4)
		tl_bbox_pred_refine = self.points2bbox(tl_pts_pred_refine.reshape(-1, 2 * self.num_points), 
											   y_first=False)
		normalize_term = self.point_base_scale * stride
		loss_pts_init += self.loss_tl_bbox_init(
			tl_bbox_pred_init / normalize_term,
			tl_bbox_gt_init / normalize_term,
			tl_bbox_weights_init,
			avg_factor=tl_num_total_samples_init)
		loss_pts_refine += self.loss_tl_bbox_refine(
			tl_bbox_pred_refine / normalize_term,
			tl_bbox_gt_refine / normalize_term,
			tl_bbox_weights_refine,
			avg_factor=tl_num_total_samples_refine)

		br_bbox_gt_init = br_bbox_gt_init.reshape(-1, 4)
		br_bbox_weights_init = br_bbox_weights_init.reshape(-1, 4)
		br_bbox_pred_init = self.points2bbox(br_pts_pred_init.reshape(-1, 2 * self.num_points),
											 y_first=False)
		br_bbox_gt_refine = br_bbox_gt_refine.reshape(-1, 4)
		br_bbox_weights_refine = br_bbox_weights_refine.reshape(-1, 4)
		br_bbox_pred_refine = self.points2bbox(br_pts_pred_refine.reshape(-1, 2 * self.num_points), 
											   y_first=False)
		normalize_term = self.point_base_scale * stride
		loss_pts_init += self.loss_br_bbox_init(
			br_bbox_pred_init / normalize_term,
			br_bbox_gt_init / normalize_term,
			br_bbox_weights_init,
			avg_factor=br_num_total_samples_init)
		loss_pts_refine += self.loss_br_bbox_refine(
			br_bbox_pred_refine / normalize_term,
			br_bbox_gt_refine / normalize_term,
			br_bbox_weights_refine,
			avg_factor=br_num_total_samples_refine)

		loss_pts_init /= 2.0
		loss_pts_refine /= 2.0

		# heatmap cls loss
		hm_score = hm_score.permute(0, 2, 3, 1).reshape(-1, 3)
		hm_score_tl, hm_score_br, hm_score_ct = torch.chunk(hm_score, 3, dim=-1)
		hm_score_tl = hm_score_tl.squeeze(1).sigmoid()
		hm_score_br = hm_score_br.squeeze(1).sigmoid()
		hm_score_ct = hm_score_ct.squeeze(1).sigmoid()

		gt_hm_tl = gt_hm_tl.reshape(-1)
		gt_hm_tl_weight = gt_hm_tl_weight.reshape(-1)
		gt_hm_br = gt_hm_br.reshape(-1)
		gt_hm_br_weight = gt_hm_br_weight.reshape(-1)
		gt_hm_ct = gt_hm_ct.reshape(-1)
		gt_hm_ct_weight = gt_hm_ct_weight.reshape(-1)

		loss_heatmap = 0
		loss_heatmap += self.loss_heatmap(
			hm_score_tl, gt_hm_tl, gt_hm_tl_weight, avg_factor=num_total_samples_tl
		)
		loss_heatmap += self.loss_heatmap(
			hm_score_br, gt_hm_br, gt_hm_br_weight, avg_factor=num_total_samples_br
		)
		loss_heatmap += self.loss_heatmap(
			hm_score_ct, gt_hm_ct, gt_hm_ct_weight, avg_factor=num_total_samples_ct
		)
		loss_heatmap /= 3.0

		# heatmap offset loss
		hm_offset = hm_offset.permute(0, 2, 3, 1).reshape(-1, 6)
		hm_offset_tl, hm_offset_br, hm_offset_ct = torch.chunk(hm_offset, 3, dim=-1)

		gt_offset_tl = gt_offset_tl.reshape(-1, 2)
		gt_offset_tl_weight = gt_offset_tl_weight.reshape(-1, 2)
		gt_offset_br = gt_offset_br.reshape(-1, 2)
		gt_offset_br_weight = gt_offset_br_weight.reshape(-1, 2)
		gt_offset_ct = gt_offset_ct.reshape(-1, 2)
		gt_offset_ct_weight = gt_offset_ct_weight.reshape(-1, 2)

		loss_offset = 0
		loss_offset += self.loss_offset(
			hm_offset_tl, gt_offset_tl, gt_offset_tl_weight,
			avg_factor=num_total_samples_tl
		)
		loss_offset += self.loss_offset(
			hm_offset_br, gt_offset_br, gt_offset_br_weight,
			avg_factor=num_total_samples_br
		)
		loss_offset += self.loss_offset(
			hm_offset_ct, gt_offset_ct, gt_offset_ct_weight,
			avg_factor=num_total_samples_ct
		)
		loss_offset /= 3.0

		return loss_cls, loss_pts_init, loss_pts_refine, loss_heatmap, loss_offset

	def loss(self,
			 tl_cls_scores,
			 br_cls_scores,
			 tl_pts_preds_init,
			 br_pts_preds_init,
			 tl_pts_preds_refine,
			 br_pts_preds_refine,
			 hm_scores,
			 hm_offsets,
			 sem_scores,
			 gt_bboxes,
			 gt_sem_map,
			 gt_sem_weights,
			 gt_labels,
			 img_metas,
			 gt_bboxes_ignore=None):

		tl_gt_bboxes, br_gt_bboxes = self.process_gt(gt_bboxes)

		featmap_sizes = [featmap.size()[-2:] for featmap in tl_cls_scores]
		assert len(featmap_sizes) == len(self.point_generators)
		label_channels = self.cls_out_channels

		# target for initial stage
		candidate_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
		pts_tl_coordinate_preds_init = self.offset_to_pts(candidate_list, tl_pts_preds_init)
		pts_br_coordinate_preds_init = self.offset_to_pts(candidate_list, br_pts_preds_init)
		tl_cls_reg_targets_init = self.get_targets(
			candidate_list.copy(),
			valid_flag_list.copy(),
			tl_gt_bboxes,
			img_metas,
			gt_bboxes_ignore_list=gt_bboxes_ignore,
			gt_labels_list=gt_labels,
			stage='init',
			label_channels=label_channels)
		(*_, tl_bbox_gt_list_init, tl_bbox_weights_list_init,
		 tl_num_total_pos_init, tl_num_total_neg_init) = tl_cls_reg_targets_init

		br_cls_reg_targets_init = self.get_targets(
			candidate_list.copy(),
			valid_flag_list.copy(),
			br_gt_bboxes,
			img_metas,
			gt_bboxes_ignore_list=gt_bboxes_ignore,
			gt_labels_list=gt_labels,
			stage='init',
			label_channels=label_channels)
		(*_, br_bbox_gt_list_init, br_bbox_weights_list_init,
		 br_num_total_pos_init, br_num_total_neg_init) = br_cls_reg_targets_init

		# target for heatmap in initial stage
		proposal_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
		heatmap_targets = self.get_hm_targets(
			proposal_list,
			valid_flag_list,
			gt_bboxes,
			img_metas,
			gt_labels)
		(gt_hm_tl_list, gt_offset_tl_list, gt_hm_tl_weight_list, gt_offset_tl_weight_list,
		 gt_hm_br_list, gt_offset_br_list, gt_hm_br_weight_list, gt_offset_br_weight_list,
		 gt_hm_ct_list, gt_offset_ct_list, gt_hm_ct_weight_list, gt_offset_ct_weight_list,
		 num_total_pos_tl, num_total_neg_tl, num_total_pos_br, num_total_neg_br,
		 num_total_pos_ct, num_total_neg_ct) = heatmap_targets

		# target for refinement stage
		center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
		pts_tl_coordinate_preds_refine = self.offset_to_pts(center_list, tl_pts_preds_refine)
		pts_br_coordinate_preds_refine = self.offset_to_pts(center_list, br_pts_preds_refine)
		tl_bbox_list = []
		br_bbox_list = []
		for i_img, center in enumerate(center_list):
			tl_bbox = []
			br_bbox = []
			for i_lvl in range(len(tl_pts_preds_refine)):
				bbox_preds_init = self.points2bbox(
					tl_pts_preds_init[i_lvl].detach())
				bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
				bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
				tl_bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))

			for i_lvl in range(len(br_pts_preds_refine)):
				bbox_preds_init = self.points2bbox(
					br_pts_preds_init[i_lvl].detach())
				bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
				bbox_center = torch.cat([center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
				br_bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
			tl_bbox_list.append(tl_bbox)
			br_bbox_list.append(br_bbox)

		tl_cls_reg_targets_refine = self.get_targets(
			tl_bbox_list,
			valid_flag_list.copy(),
			tl_gt_bboxes,
			img_metas,
			gt_bboxes_ignore_list=gt_bboxes_ignore,
			gt_labels_list=gt_labels,
			stage='refine',
			label_channels=label_channels)
		(tl_labels_list, tl_label_weights_list,
		 tl_bbox_gt_list_refine, tl_bbox_weights_list_refine,
		 tl_num_total_pos_refine, tl_num_total_neg_refine) = tl_cls_reg_targets_refine

		br_cls_reg_targets_refine = self.get_targets(
			br_bbox_list,
			valid_flag_list.copy(),
			br_gt_bboxes,
			img_metas,
			gt_bboxes_ignore_list=gt_bboxes_ignore,
			gt_labels_list=gt_labels,
			stage='refine',
			label_channels=label_channels)
		(br_labels_list, br_label_weights_list,
		 br_bbox_gt_list_refine, br_bbox_weights_list_refine,
		 br_num_total_pos_refine, br_num_total_neg_refine) = br_cls_reg_targets_refine

		# compute loss
		losses_cls, losses_pts_init, losses_pts_refine, losses_heatmap, losses_offset = multi_apply(
			self.loss_single,
			tl_cls_scores,
			br_cls_scores,
			pts_tl_coordinate_preds_init,
			pts_br_coordinate_preds_init,
			pts_tl_coordinate_preds_refine,
			pts_br_coordinate_preds_refine,
			hm_scores,
			hm_offsets,
			tl_labels_list,
			br_labels_list,
			tl_label_weights_list,
			br_label_weights_list,
			tl_bbox_gt_list_init,
			br_bbox_gt_list_init,
			tl_bbox_weights_list_init,
			br_bbox_weights_list_init,
			tl_bbox_gt_list_refine,
			br_bbox_gt_list_refine,
			tl_bbox_weights_list_refine,
			br_bbox_weights_list_refine,
			gt_hm_tl_list,
			gt_offset_tl_list,
			gt_hm_tl_weight_list,
			gt_offset_tl_weight_list,
			gt_hm_br_list,
			gt_offset_br_list,
			gt_hm_br_weight_list,
			gt_offset_br_weight_list,
			gt_hm_ct_list,
			gt_offset_ct_list,
			gt_hm_ct_weight_list,
			gt_offset_ct_weight_list,
			self.point_strides,
			tl_num_total_samples_init=tl_num_total_pos_init,
			br_num_total_samples_init=br_num_total_pos_init,
			tl_num_total_samples_refine=tl_num_total_pos_refine,
			br_num_total_samples_refine=br_num_total_pos_refine,
			num_total_samples_tl=num_total_pos_tl,
			num_total_samples_br=num_total_pos_br,
			num_total_samples_ct=num_total_pos_ct)

		# sem loss
		concat_sem_scores = []
		concat_gt_sem_map = []
		concat_gt_sem_weights = []

		for i in range(5):
			sem_score = sem_scores[i]
			gt_lvl_sem_map = F.interpolate(gt_sem_map, sem_score.shape[-2:]).reshape(-1)
			gt_lvl_sem_weight = F.interpolate(gt_sem_weights, sem_score.shape[-2:]).reshape(-1)
			sem_score = sem_score.reshape(-1)

			try:
				concat_sem_scores = torch.cat([concat_sem_scores, sem_score])
				concat_gt_sem_map = torch.cat([concat_gt_sem_map, gt_lvl_sem_map])
				concat_gt_sem_weights = torch.cat([concat_gt_sem_weights, gt_lvl_sem_weight])
			except:
				concat_sem_scores = sem_score
				concat_gt_sem_map = gt_lvl_sem_map
				concat_gt_sem_weights = gt_lvl_sem_weight

		loss_sem = self.loss_sem(concat_sem_scores, concat_gt_sem_map, concat_gt_sem_weights,
								 avg_factor=(concat_gt_sem_map > 0).sum())

		loss_dict_all = {'loss_cls': losses_cls,
						 'loss_pts_init': losses_pts_init,
						 'loss_pts_refine': losses_pts_refine,
						 'loss_heatmap': losses_heatmap,
						 'loss_offset': losses_offset,
						 'loss_sem': loss_sem,
						 }
		return loss_dict_all

	def get_bboxes(self,
				   tl_cls_scores,
				   br_cls_scores,
				   tl_pts_preds_init,
				   br_pts_preds_init,
				   tl_pts_preds_refine,
				   br_pts_preds_refine,
				   hm_scores,
				   hm_offsets,
				   sem_scores,
				   img_metas,
				   cfg=None,
				   rescale=False,
				   nms=True):
		assert len(tl_cls_scores) == len(tl_pts_preds_refine)
		tl_bbox_preds_refine = [self.points2bbox(tl_pts_pred_refine) for tl_pts_pred_refine in 
								tl_pts_preds_refine]
		br_bbox_preds_refine = [self.points2bbox(br_pts_pred_refine) for br_pts_pred_refine in 
								br_pts_preds_refine]
		num_levels = len(tl_cls_scores)
		mlvl_points = [
			self.point_generators[i].grid_points(tl_cls_scores[i].size()[-2:],
												 self.point_strides[i])
			for i in range(num_levels)
		]
		result_list = []
		for img_id in range(len(img_metas)):
			tl_cls_score_list = [
				tl_cls_scores[i][img_id].detach() for i in range(num_levels)
			]
			br_cls_score_list = [
				br_cls_scores[i][img_id].detach() for i in range(num_levels)
			]
			tl_bbox_pred_list = [
				tl_bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
			]
			br_bbox_pred_list = [
				br_bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
			]
			hm_scores_list = [
				hm_scores[i][img_id].detach() for i in range(num_levels)
			]
			hm_offsets_list = [
				hm_offsets[i][img_id].detach() for i in range(num_levels)
			]
			img_shape = img_metas[img_id]['img_shape']
			scale_factor = img_metas[img_id]['scale_factor']
			proposals = self._get_bboxes_single(tl_cls_score_list, br_cls_score_list,
												tl_bbox_pred_list, br_bbox_pred_list,
												hm_scores_list, hm_offsets_list,
												mlvl_points, img_shape,
												scale_factor, cfg, rescale,
												nms)
			result_list.append(proposals)
		return result_list

	def _get_bboxes_single(self,
						   tl_cls_scores,
						   br_cls_scores,
						   tl_bbox_preds,
						   br_bbox_preds,
						   hm_scores,
						   hm_offsets,
						   mlvl_points,
						   img_shape,
						   scale_factor,
						   cfg,
						   rescale=False,
						   nms=True):
		def select(score_map, x, y, ks=2, i=0):
			H, W = score_map.shape[-2], score_map.shape[-1]
			score_map = score_map.sigmoid()
			score_map_original = score_map.clone()

			score_map, indices = F.max_pool2d_with_indices(score_map.unsqueeze(0), kernel_size=ks, stride=1, padding=(ks - 1) // 2)

			indices = indices.squeeze(0).squeeze(0)

			if ks % 2 == 0:
				round_func = torch.floor
			else:
				round_func = torch.round

			x_round = round_func((x / self.point_strides[i]).clamp(min=0, max=score_map.shape[-1] - 1))
			y_round = round_func((y / self.point_strides[i]).clamp(min=0, max=score_map.shape[-2] - 1))

			select_indices = indices[y_round.to(torch.long), x_round.to(torch.long)]
			new_x = select_indices % W
			new_y = torch.div(select_indices, W, rounding_mode='floor')

			score_map_squeeze = score_map_original.squeeze(0)
			score = score_map_squeeze[new_y, new_x]

			new_x, new_y = new_x.to(torch.float), new_y.to(torch.float)

			return new_x, new_y, score

		cfg = self.test_cfg if cfg is None else cfg
		assert len(tl_cls_scores) == len(tl_bbox_preds) == len(mlvl_points)
		mlvl_tl_bboxes = []
		mlvl_br_bboxes = []
		mlvl_tl_scores = []
		mlvl_br_scores = []
		dis_thr = cfg.get('distance_threshold', -1)
		for i_lvl, (tl_cls_score, br_cls_score, tl_bbox_pred, br_bbox_pred, points) in enumerate(zip(tl_cls_scores, br_cls_scores, tl_bbox_preds, br_bbox_preds, mlvl_points)):
			assert tl_cls_score.size()[-2:] == tl_bbox_pred.size()[-2:]
			tl_scores = tl_cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
			br_scores = br_cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
			tl_bbox_pred = tl_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
			br_bbox_pred = br_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
			nms_pre = cfg.get('nms_pre', -1)
			if nms_pre > 0 and tl_scores.shape[0] > nms_pre:
				tl_max_scores, _ = tl_scores.max(dim=1)
				_, tl_topk_inds = tl_max_scores.topk(nms_pre)
				tl_points = points[tl_topk_inds, :]
				tl_bbox_pred = tl_bbox_pred[tl_topk_inds, :]
				tl_scores = tl_scores[tl_topk_inds, :]
			else:
				tl_points = points
			if nms_pre > 0 and br_scores.shape[0] > nms_pre:
				br_max_scores, _ = br_scores.max(dim=1)
				_, br_topk_inds = br_max_scores.topk(nms_pre)
				br_points = points[br_topk_inds, :]
				br_bbox_pred = br_bbox_pred[br_topk_inds, :]
				br_scores = br_scores[br_topk_inds, :]
			else:
				br_points = points

			tl_bbox_pos_center = torch.cat([tl_points[:, :2], tl_points[:, :2]], dim=1)
			br_bbox_pos_center = torch.cat([br_points[:, :2], br_points[:, :2]], dim=1)
			tl_bboxes = tl_bbox_pred * self.point_strides[i_lvl] + tl_bbox_pos_center
			br_bboxes = br_bbox_pred * self.point_strides[i_lvl] + br_bbox_pos_center
			x1 = tl_bboxes[:, 0].clamp(min=0, max=img_shape[1])
			y1 = tl_bboxes[:, 1].clamp(min=0, max=img_shape[0])
			x2 = tl_bboxes[:, 2].clamp(min=0, max=img_shape[1])
			y2 = tl_bboxes[:, 3].clamp(min=0, max=img_shape[0])

			x3 = br_bboxes[:, 0].clamp(min=0, max=img_shape[1])
			y3 = br_bboxes[:, 1].clamp(min=0, max=img_shape[0])
			x4 = br_bboxes[:, 2].clamp(min=0, max=img_shape[1])
			y4 = br_bboxes[:, 3].clamp(min=0, max=img_shape[0])

			if i_lvl > 0:
				i = 0 if i_lvl in (1, 2) else 1

				x1_new, y1_new, score1_new = select(hm_scores[i][0, ...], x1, y1, 2, i)
				x2_new, y2_new, score2_new = select(hm_scores[i][2, ...], x2, y2, 2, i)
				x3_new, y3_new, score3_new = select(hm_scores[i][2, ...], x3, y3, 2, i)
				x4_new, y4_new, score4_new = select(hm_scores[i][1, ...], x4, y4, 2, i)

				hm_offset = hm_offsets[i].permute(1, 2, 0)
				point_stride = self.point_strides[i]

				x1 = ((x1_new + hm_offset[y1_new.to(torch.long), x1_new.to(torch.long), 0]) *
					   point_stride).clamp(min=0, max=img_shape[1])
				y1 = ((y1_new + hm_offset[y1_new.to(torch.long), x1_new.to(torch.long), 1]) * 
					   point_stride).clamp(min=0, max=img_shape[0])
				x2 = ((x2_new + hm_offset[y2_new.to(torch.long), x2_new.to(torch.long), 4]) * 
					   point_stride).clamp(min=0, max=img_shape[1])
				y2 = ((y2_new + hm_offset[y2_new.to(torch.long), x2_new.to(torch.long), 5]) * 
					   point_stride).clamp(min=0, max=img_shape[0])
				x3 = ((x3_new + hm_offset[y3_new.to(torch.long), x3_new.to(torch.long), 4]) *
					   point_stride).clamp(min=0, max=img_shape[1])
				y3 = ((y3_new + hm_offset[y3_new.to(torch.long), x3_new.to(torch.long), 5]) * 
					   point_stride).clamp(min=0, max=img_shape[0])
				x4 = ((x4_new + hm_offset[y4_new.to(torch.long), x4_new.to(torch.long), 2]) * 
					   point_stride).clamp(min=0, max=img_shape[1])
				y4 = ((y4_new + hm_offset[y4_new.to(torch.long), x4_new.to(torch.long), 3]) * 
					   point_stride).clamp(min=0, max=img_shape[0])

			tl_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
			br_bboxes = torch.stack([x3, y3, x4, y4], dim=-1)
			mlvl_tl_bboxes.append(tl_bboxes)
			mlvl_br_bboxes.append(br_bboxes)
			mlvl_tl_scores.append(tl_scores)
			mlvl_br_scores.append(br_scores)
		mlvl_tl_bboxes = torch.cat(mlvl_tl_bboxes)
		mlvl_br_bboxes = torch.cat(mlvl_br_bboxes)
		if rescale:
			mlvl_tl_bboxes /= mlvl_tl_bboxes.new_tensor(scale_factor)
			mlvl_br_bboxes /= mlvl_br_bboxes.new_tensor(scale_factor)
		mlvl_tl_scores = torch.cat(mlvl_tl_scores)
		mlvl_br_scores = torch.cat(mlvl_br_scores)
		tl_padding = mlvl_tl_scores.new_zeros(mlvl_tl_scores.shape[0], 1)
		br_padding = mlvl_br_scores.new_zeros(mlvl_br_scores.shape[0], 1)
		mlvl_tl_scores = torch.cat([mlvl_tl_scores, tl_padding], dim=1)
		mlvl_br_scores = torch.cat([mlvl_br_scores, br_padding], dim=1)
		
		det_tl_bboxes, det_tl_labels = multiclass_nms(mlvl_tl_bboxes.float(), mlvl_tl_scores.float(), cfg.score_thr, cfg.nms, cfg.max_per_img)
		det_br_bboxes, det_br_labels = multiclass_nms(mlvl_br_bboxes.float(), mlvl_br_scores.float(), cfg.score_thr, cfg.nms, cfg.max_per_img)
													  
		det_bboxes, det_scores, det_labels = self.decode(det_tl_bboxes, det_tl_labels, det_br_bboxes, det_br_labels, distance_threshold = dis_thr)
		
		if det_bboxes.numel() == 0:
			bboxes = det_bboxes.new_zeros((0, 5))
			labels = det_bboxes.new_zeros((0,), dtype=torch.long)
		else:
			dets, keep = batched_nms(det_bboxes.float(), det_scores.float(), det_labels, cfg.nms)
			if cfg.max_per_img > 0:
				bboxes = dets[:cfg.max_per_img]
				keep   = keep[:cfg.max_per_img]
				labels = det_labels[keep]
		return bboxes, labels

	def get_num_level_proposals_inside(self, num_level_proposals, inside_flags):
		split_inside_flags = torch.split(inside_flags, num_level_proposals)
		num_level_proposals_inside = [int(flags.sum()) for flags in split_inside_flags]
		return num_level_proposals_inside

	def process_gt(self, gt_bboxes_list):
		tl_gt_list = []
		br_gt_list = []
		for gt_bboxes in gt_bboxes_list:
			ctx = (gt_bboxes[:, 2] + gt_bboxes[:, 0])/2.0
			cty = (gt_bboxes[:, 3] + gt_bboxes[:, 1])/2.0

			tlx = gt_bboxes[:, 0]
			tly = gt_bboxes[:, 1]
			brx = gt_bboxes[:, 2]
			bry = gt_bboxes[:, 3]

			tl_gt_list.append(torch.stack([tlx, tly, ctx, cty], dim = 1))
			br_gt_list.append(torch.stack([ctx, cty, brx, bry], dim = 1))
	
		return tl_gt_list, br_gt_list
	
	def decode(self, det_tl_bboxes, det_tl_labels, det_br_bboxes, det_br_labels, distance_threshold = 0.5):
		tl_xs   = det_tl_bboxes[:, 0]
		tl_ys   = det_tl_bboxes[:, 1]
		tl_ctxs = det_tl_bboxes[:, 2]
		tl_ctys = det_tl_bboxes[:, 3]

		br_ctxs = det_br_bboxes[:, 0]
		br_ctys = det_br_bboxes[:, 1]
		br_xs   = det_br_bboxes[:, 2]
		br_ys   = det_br_bboxes[:, 3]

		tl_scores = det_tl_bboxes[:, -1]
		br_scores = det_br_bboxes[:, -1]

		tl_scores = tl_scores.view(tl_scores.size(0), 1).repeat(1, br_scores.size(0))
		br_scores = br_scores.view(1, br_scores.size(0)).repeat(tl_scores.size(0), 1)
		scores = (tl_scores + br_scores) / 2

		tl_clses = det_tl_labels.view(det_tl_labels.size(0), 1).repeat(1, det_br_labels.size(0))
		br_clses = det_br_labels.view(1, det_br_labels.size(0)).repeat(det_tl_labels.size(0), 1)
		cls_inds = (tl_clses != br_clses)

		tl_xs = tl_xs.view(tl_xs.size(0), 1).repeat(1, br_xs.size(0))
		tl_ys = tl_ys.view(tl_ys.size(0), 1).repeat(1, br_ys.size(0))
		br_xs = br_xs.view(1, br_xs.size(0)).repeat(tl_xs.size(0), 1)
		br_ys = br_ys.view(1, br_ys.size(0)).repeat(tl_ys.size(0), 1)

		tl_ctxs = tl_ctxs.view(tl_ctxs.size(0), 1).repeat(1, br_ctxs.size(0))
		tl_ctys = tl_ctys.view(tl_ctys.size(0), 1).repeat(1, br_ctys.size(0))
		br_ctxs = br_ctxs.view(1, br_ctxs.size(0)).repeat(tl_ctxs.size(0), 1)
		br_ctys = br_ctys.view(1, br_ctys.size(0)).repeat(tl_ctys.size(0), 1)

		tl_zeros = tl_xs.new_zeros(*tl_xs.size())
		br_zeros = br_xs.new_zeros(*br_xs.size())
		tl_xs = torch.where(tl_xs > 0.0, tl_xs, tl_zeros)
		tl_ys = torch.where(tl_ys > 0.0, tl_ys, tl_zeros)
		br_xs = torch.where(br_xs > 0.0, br_xs, br_zeros)
		br_ys = torch.where(br_ys > 0.0, br_ys, br_zeros)

		bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=2)
		area_bboxes = ((br_xs-tl_xs)*(br_ys-tl_ys)).abs()

		tl_ctxs *= tl_ctxs.gt(0.0).type_as(tl_ctxs)
		tl_ctys *= tl_ctys.gt(0.0).type_as(tl_ctys)
		br_ctxs *= br_ctxs.gt(0.0).type_as(br_ctxs)
		br_ctys *= br_ctys.gt(0.0).type_as(br_ctys)

		ct_bboxes = torch.stack((tl_ctxs, tl_ctys, br_ctxs, br_ctys), dim=2)

		area_ct_bboxes = ((br_ctxs - tl_ctxs)*(br_ctys - tl_ctys)).abs()

		rcentral = torch.zeros_like(ct_bboxes)
		mu = torch.ones_like(area_bboxes) / 5
		mu[area_bboxes > 3500] = 1 / 4

		bboxes_center_x = (bboxes[..., 0] + bboxes[..., 2]) / 2
		bboxes_center_y = (bboxes[..., 1] + bboxes[..., 3]) / 2
		rcentral[..., 0] = bboxes_center_x - mu * (bboxes[..., 2] - bboxes[..., 0]) / 2
		rcentral[..., 1] = bboxes_center_y - mu * (bboxes[..., 3] - bboxes[..., 1]) / 2
		rcentral[..., 2] = bboxes_center_x + mu * (bboxes[..., 2] - bboxes[..., 0]) / 2
		rcentral[..., 3] = bboxes_center_y + mu * (bboxes[..., 3] - bboxes[..., 1]) / 2
		area_rcentral = ((rcentral[..., 2] - rcentral[..., 0]) * (rcentral[..., 3] - rcentral[..., 1])).abs()
		dists = area_ct_bboxes / area_rcentral

		tl_ctx_inds = (ct_bboxes[..., 0] <= rcentral[..., 0]) | (ct_bboxes[..., 0] >= rcentral[..., 2])
		tl_cty_inds = (ct_bboxes[..., 1] <= rcentral[..., 1]) | (ct_bboxes[..., 1] >= rcentral[..., 3])
		br_ctx_inds = (ct_bboxes[..., 2] <= rcentral[..., 0]) | (ct_bboxes[..., 2] >= rcentral[..., 2])
		br_cty_inds = (ct_bboxes[..., 3] <= rcentral[..., 1]) | (ct_bboxes[..., 3] >= rcentral[..., 3])

		# reject boxes based on distances
		dist_inds = dists > distance_threshold

		# reject boxes based on widths and heights
		width_inds = (br_xs <= tl_xs)
		height_inds = (br_ys <= tl_ys)

		negative_scores = -1 * torch.ones_like(scores)
		scores = torch.where(cls_inds, negative_scores, scores)
		scores = torch.where(width_inds, negative_scores, scores)
		scores = torch.where(height_inds, negative_scores, scores)
		scores = torch.where(dist_inds, negative_scores, scores)

		scores[tl_ctx_inds] = -1
		scores[tl_cty_inds] = -1
		scores[br_ctx_inds] = -1
		scores[br_cty_inds] = -1

		pos_inds = scores > 0
		scores = scores[pos_inds]
		labels = tl_clses[pos_inds]
		bboxes = bboxes[pos_inds, :]
		return bboxes, scores, labels
