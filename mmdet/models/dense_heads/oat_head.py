import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, distance2bbox, bbox2distance, bbox_overlaps, multiclass_nms, reduce_mean

from ..builder import HEADS, build_loss
from .gfl_head import GFLHead

@HEADS.register_module()
class OATHead(GFLHead):
	def __init__(self,
				 num_points = 9,
				 loss_bbox_refine = dict(type='GIoULoss', loss_weight=2.0),
				 init_cfg=dict(
					 type='Normal',
					 layer='Conv2d',
					 std=0.01,
					 override=dict(
						 type='Normal',
						 name='oat_cls',
						 std=0.01,
						 bias_prob=0.01)),
				 **kwargs):
		self.num_points = num_points

		self.dcn_kernel = int(np.sqrt(num_points))
		self.dcn_pad = int((self.dcn_kernel - 1) / 2)
		assert self.dcn_kernel * self.dcn_kernel == num_points,  'The points number should be a square number.'
		assert self.dcn_kernel % 2 == 1, 'The points number should be an odd square number.'
		dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
		dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
		dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
		dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
		self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
		super(OATHead, self).__init__(init_cfg=init_cfg, **kwargs)
		self.loss_bbox_refine = build_loss(loss_bbox_refine)

	def _init_layers(self):
		self.relu = nn.ReLU(inplace=True)
		self.cls_convs = nn.ModuleList()
		self.reg_convs = nn.ModuleList()
		for i in range(self.stacked_convs):
			chn = self.in_channels if i == 0 else self.feat_channels
			self.cls_convs.append(
				ConvModule(
					chn,
					self.feat_channels,
					3,
					stride=1,
					padding=1,
					conv_cfg=self.conv_cfg,
					norm_cfg=self.norm_cfg))
			self.reg_convs.append(
				ConvModule(
					chn,
					self.feat_channels,
					3,
					stride=1,
					padding=1,
					conv_cfg=self.conv_cfg,
					norm_cfg=self.norm_cfg))
		assert self.num_anchors == 1, 'anchor free version'
		self.rfa_reg_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, stride = 1, padding = 1)
		self.rfa_reg = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 3, stride = 1, padding = 1)
		self.rfa_offset = nn.Conv2d(self.feat_channels, 2 * self.num_points - 4, 3, stride = 1, padding = 1)
		self.cfa_cls_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, stride = 1, padding = 1)
		self.cfa_distanglement = nn.Conv2d(self.feat_channels, 2 * self.num_points, 3, stride = 1, padding = 1)

		self.oat_reg_dconv = DeformConv2d(self.feat_channels, self.feat_channels, self.dcn_kernel, 1, padding=self.dcn_pad)
		self.oat_cls_dconv = DeformConv2d(self.feat_channels, self.feat_channels, self.dcn_kernel, 1, padding=self.dcn_pad)
		self.oat_reg_refine = nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 3, stride = 1, padding = 1)
		self.oat_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, stride = 1, padding = 1)

		self.scales = nn.ModuleList([Scale(1.0) for _ in self.anchor_generator.strides])
		self.refine_scales = nn.ModuleList([Scale(1.0) for _ in self.anchor_generator.strides])
		if self.use_dgqp:
			conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
			conf_vector += [self.relu]
			conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]
			self.reg_conf = nn.Sequential(*conf_vector)

	def forward(self, feats):
		return multi_apply(self.forward_single, feats, self.scales, self.refine_scales, self.anchor_generator.strides)

	def forward_single(self, x, scale, refine_scale, stride):
		cls_feat = x
		reg_feat = x

		for cls_layer in self.cls_convs:
			cls_feat = cls_layer(cls_feat)

		for reg_layer in self.reg_convs:
			reg_feat = reg_layer(reg_feat)

		reg_feat_init = self.relu(self.rfa_reg_conv(reg_feat))
		bbox_pred = scale(self.relu(self.rfa_reg(reg_feat_init)))
		point_offset = self.rfa_offset(reg_feat_init).sigmoid()

		dcn_offset = self.gen_dcn_offset(bbox_pred, point_offset, stride)
		reg_dcn_offset = dcn_offset - self.dcn_base_offset.type_as(bbox_pred)
		reg_feat = self.oat_reg_dconv(reg_feat, reg_dcn_offset)
		bbox_pred_refine = refine_scale(self.oat_reg_refine(reg_feat))

		cls_feat_init = self.relu(self.cfa_cls_conv(cls_feat + x))
		distanglement_vector = self.relu(self.cfa_distanglement(cls_feat_init))
		cls_dcn_offset = distanglement_vector.exp() * dcn_offset.detach() - self.dcn_base_offset.type_as(bbox_pred)
		cls_feat = self.oat_cls_dconv(cls_feat, cls_dcn_offset)
		cls_score = self.oat_cls(cls_feat)
		if self.use_dgqp:
			N, C, H, W = bbox_pred_refine.size()
			prob = F.softmax(bbox_pred_refine.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
			prob_topk, _ = prob.topk(self.reg_topk, dim=2)
			if self.add_mean:
				prob_topk_mean = prob_topk.mean(dim=2, keepdim=True)
				stat = torch.cat([prob_topk, prob_topk_mean], dim=2)
			else:
				stat = prob_topk
			quality_score = self.reg_conf(stat.type_as(reg_feat).reshape(N, -1, H, W))
			cls_score = cls_score.sigmoid() * quality_score
		return cls_score, bbox_pred, bbox_pred_refine

	def gen_dcn_offset(self, bbox_pred, point_offset, stride):
		N, C, H, W = bbox_pred.shape
		bbox_pred = bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
		bbox_pred = self.integral(bbox_pred).view(N, H, W, 4).permute(0, 3, 1, 2).contiguous()

		l = bbox_pred[:, 0, :, :]
		t = bbox_pred[:, 1, :, :]
		r = bbox_pred[:, 2, :, :]
		b = bbox_pred[:, 3, :, :]
		w = r + l
		h = b + t

		dcn_offset = torch.zeros((N, 2 * self.num_points, H, W), device = bbox_pred.device)
		dcn_offset[:, 0, :, :] = -1 * t
		dcn_offset[:, 1, :, :] = w * point_offset[:, 0, :, :] - l
		dcn_offset[:, 2, :, :] = h * point_offset[:, 1, :, :] - t
		dcn_offset[:, 3, :, :] = -1 * l
		dcn_offset[:, 4, :, :] = -1 * b
		dcn_offset[:, 5, :, :] = w * point_offset[:, 2, :, :] - l
		dcn_offset[:, 6, :, :] = h * point_offset[:, 3, :, :] - t
		dcn_offset[:, 7, :, :] = -1 * r
		for i in range(4, self.num_points):
			dcn_offset[:, 2 * i, :, :] = h * point_offset[:, 2 * i - 4, :, :] - t
			dcn_offset[:, 2 * i + 1, :, :] = w * point_offset[:, 2 * i - 3, :, :] - l
		return dcn_offset

	def loss_single(self, anchors, cls_score, bbox_pred, bbox_pred_refine, labels, label_weights,
					bbox_targets, stride, num_total_samples):
		"""Compute loss of a single scale level.
		Args:
			anchors (Tensor): Box reference for each scale level with shape
				(N, num_total_anchors, 4).
			cls_score (Tensor): Cls and quality joint scores for each scale
				level has shape (N, num_classes, H, W).
			bbox_pred (Tensor): Box distribution logits for each scale
				level with shape (N, 4*(n+1), H, W), n is max value of integral
				set.
			bbox_pred_refine (Tensor): Refined box distribution logits for each scale
				level with shape (N, 4*(n+1), H, W), n is max value of integral
				set.
			labels (Tensor): Labels of each anchors with shape
				(N, num_total_anchors).
			label_weights (Tensor): Label weights of each anchor with shape
				(N, num_total_anchors)
			bbox_targets (Tensor): BBox regression targets of each anchor wight
				shape (N, num_total_anchors, 4).
			stride (tuple): Stride in this scale level.
			num_total_samples (int): Number of positive samples that is
				reduced over all GPUs.
		Returns:
			dict[str, Tensor]: A dictionary of loss components.
		"""
		assert stride[0] == stride[1], 'h stride is not equal to w stride!'
		anchors = anchors.reshape(-1, 4)
		cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
		bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
		bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
		bbox_targets = bbox_targets.reshape(-1, 4)
		labels = labels.reshape(-1)
		label_weights = label_weights.reshape(-1)

		# FG cat_id: [0, num_classes -1], BG cat_id: num_classes
		bg_class_ind = self.num_classes
		pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)
		score = label_weights.new_zeros(labels.shape)

		if len(pos_inds) > 0:
			pos_bbox_targets = bbox_targets[pos_inds]
			pos_decode_bbox_targets = pos_bbox_targets / stride[0]
			pos_bbox_pred = bbox_pred[pos_inds]
			pos_bbox_pred_refine = bbox_pred_refine[pos_inds]
			pos_anchors = anchors[pos_inds]
			pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

			weight_targets = cls_score.detach()
			if not self.use_dgqp:
				weight_targets = weight_targets.sigmoid()
			weight_targets = weight_targets.max(dim=1)[0][pos_inds]
			pos_bbox_pred_corners = self.integral(pos_bbox_pred)
			pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred_corners)
			loss_bbox = self.loss_bbox(pos_decode_bbox_pred, pos_decode_bbox_targets, weight=weight_targets, avg_factor=1.0)

			pos_bbox_pred_refine_corners = self.integral(pos_bbox_pred_refine)
			pos_decode_bbox_pred_refine = distance2bbox(pos_anchor_centers, pos_bbox_pred_refine_corners)
			loss_bbox_refine = self.loss_bbox_refine(pos_decode_bbox_pred_refine, pos_decode_bbox_targets, weight=weight_targets, avg_factor= 1.0)

			score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred_refine.detach(), pos_decode_bbox_targets, is_aligned=True)
			pred_corners = pos_bbox_pred_refine.reshape(-1, self.reg_max + 1)
			target_corners = bbox2distance(pos_anchor_centers, pos_decode_bbox_targets, self.reg_max).reshape(-1)

			# dfl loss
			loss_dfl = self.loss_dfl(
				pred_corners,
				target_corners,
				weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
				avg_factor=4.0)
		else:
			loss_bbox = bbox_pred.sum() * 0
			loss_bbox_refine = bbox_pred_refine.sum() * 0
			loss_dfl = bbox_pred.sum() * 0
			weight_targets = bbox_pred.new_tensor(0)

		# cls (qfl) loss
		loss_cls = self.loss_cls(
			cls_score, (labels, score),
			weight=label_weights,
			avg_factor=num_total_samples)

		return loss_cls, loss_bbox, loss_bbox_refine, loss_dfl, weight_targets.sum()

	@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
	def loss(self,
			 cls_scores,
			 bbox_preds,
			 bbox_preds_refine,
			 gt_bboxes,
			 gt_labels,
			 img_metas,
			 gt_bboxes_ignore=None):
		"""Compute losses of the head.
		Args:
			cls_scores (list[Tensor]): Cls and quality scores for each scale
				level has shape (N, num_classes, H, W).
			bbox_preds (list[Tensor]): Box distribution logits for each scale
				level with shape (N, 4*(n+1), H, W), n is max value of integral
				set.
			bbox_preds_refine (list[Tensor]): Refined box distribution logits for each scale
				level with shape (N, 4*(n+1), H, W), n is max value of integral
				set.
			gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
				shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
			gt_labels (list[Tensor]): class indices corresponding to each box
			img_metas (list[dict]): Meta information of each image, e.g.,
				image size, scaling factor, etc.
			gt_bboxes_ignore (list[Tensor] | None): specify which bounding
				boxes can be ignored when computing the loss.
		Returns:
			dict[str, Tensor]: A dictionary of loss components.
		"""

		featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
		assert len(featmap_sizes) == self.anchor_generator.num_levels

		device = cls_scores[0].device
		anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
		label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

		cls_reg_targets = self.get_targets(
			anchor_list,
			valid_flag_list,
			gt_bboxes,
			img_metas,
			gt_bboxes_ignore_list=gt_bboxes_ignore,
			gt_labels_list=gt_labels,
			label_channels=label_channels)
		if cls_reg_targets is None:
			return None

		(anchor_list, labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

		num_total_samples = reduce_mean(torch.tensor(num_total_pos, dtype=torch.float, device=device)).item()
		if self.avg_samples_to_int:
			num_total_samples = int(num_total_samples)
		num_total_samples = max(num_total_samples, 1.0)

		losses_cls, losses_bbox, losses_bbox_refine, losses_dfl,\
			avg_factor = multi_apply(
				self.loss_single,
				anchor_list,
				cls_scores,
				bbox_preds,
				bbox_preds_refine,
				labels_list,
				label_weights_list,
				bbox_targets_list,
				self.anchor_generator.strides,
				num_total_samples=num_total_samples)

		avg_factor = sum(avg_factor)
		avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
		losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
		losses_bbox_refine = list(map(lambda x: x / avg_factor, losses_bbox_refine))
		losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
		return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_bbox_refine=losses_bbox_refine, loss_dfl=losses_dfl)

	@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine'))
	def get_bboxes(self,
				   cls_scores,
				   bbox_preds,
				   bbox_preds_refine,
				   img_metas,
				   cfg=None,
				   rescale=False,
				   with_nms=True):
		assert len(cls_scores) == len(bbox_preds)
		num_levels = len(cls_scores)

		device = cls_scores[0].device
		featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
		mlvl_anchors = self.anchor_generator.grid_priors(
			featmap_sizes, device=device)

		mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
		mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
		mlvl_bbox_preds_refine = [bbox_preds_refine[i].detach() for i in range(num_levels)]

		if torch.onnx.is_in_onnx_export():
			assert len(img_metas) == 1, 'Only support one input image while in exporting to ONNX'
			img_shapes = img_metas[0]['img_shape_for_onnx']
		else:
			img_shapes = [img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])]
		scale_factors = [img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])]

		if with_nms:
			result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds, mlvl_bbox_preds_refine,
										   mlvl_anchors, img_shapes, scale_factors, cfg, rescale)
		else:
			result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds, mlvl_bbox_preds_refine,
										   mlvl_anchors, img_shapes, scale_factors, cfg, rescale, with_nms)
		return result_list

	def _get_bboxes(self,
					cls_scores,
					bbox_preds,
					bbox_preds_refine,
					mlvl_anchors,
					img_shapes,
					scale_factors,
					cfg,
					rescale=False,
					with_nms=True):
		"""Transform outputs for a single batch item into labeled boxes.
		Args:
			cls_scores (list[Tensor]): Box scores for a single scale level
				has shape (N, num_classes, H, W).
			bbox_preds (list[Tensor]): Box distribution logits for a single
				scale level with shape (N, 4*(n+1), H, W), n is max value of
				integral set.
			bbox_preds_refine (list[Tensor]): Refined box distribution logits for a single
				scale level with shape (N, 4*(n+1), H, W), n is max value of
				integral set.
			mlvl_anchors (list[Tensor]): Box reference for a single scale level
				with shape (num_total_anchors, 4).
			img_shapes (list[tuple[int]]): Shape of the input image,
				list[(height, width, 3)].
			scale_factors (list[ndarray]): Scale factor of the image arange as
				(w_scale, h_scale, w_scale, h_scale).
			cfg (mmcv.Config | None): Test / postprocessing configuration,
				if None, test_cfg would be used.
			rescale (bool): If True, return boxes in original image space.
				Default: False.
			with_nms (bool): If True, do nms before return boxes.
				Default: True.
		Returns:
			list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
				The first item is an (n, 5) tensor, where 5 represent
				(tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
				The shape of the second tensor in the tuple is (n,), and
				each element represents the class label of the corresponding
				box.
		"""
		cfg = self.test_cfg if cfg is None else cfg
		assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
		batch_size = cls_scores[0].shape[0]

		mlvl_bboxes = []
		mlvl_scores = []
		for cls_score, bbox_pred_refine, stride, anchors in zip(cls_scores, bbox_preds_refine, self.anchor_generator.strides, mlvl_anchors):
			assert cls_score.size()[-2:] == bbox_pred_refine.size()[-2:]
			assert stride[0] == stride[1]
			scores_shape = (batch_size, -1, self.cls_out_channels)
			scores = cls_score.permute(0, 2, 3, 1).reshape(scores_shape)
			if not self.use_dgqp:
				scores = scores.sigmoid()

			bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1)

			bbox_pred_refine = self.integral(bbox_pred_refine) * stride[0]
			bbox_pred_refine = bbox_pred_refine.reshape(batch_size, -1, 4)

			nms_pre = cfg.get('nms_pre', -1)
			if nms_pre > 0 and scores.shape[1] > nms_pre:
				max_scores, _ = scores.max(-1)
				_, topk_inds = max_scores.topk(nms_pre)
				batch_inds = torch.arange(batch_size).view(-1, 1).expand_as(topk_inds).long()
				anchors = anchors[topk_inds, :]
				bbox_pred_refine = bbox_pred_refine[batch_inds, topk_inds, :]
				scores = scores[batch_inds, topk_inds, :]
			else:
				anchors = anchors.expand_as(bbox_pred_refine)

			bboxes = distance2bbox(self.anchor_center(anchors), bbox_pred_refine, max_shape=img_shapes)
			mlvl_bboxes.append(bboxes)
			mlvl_scores.append(scores)

		batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
		if rescale:
			batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(scale_factors).unsqueeze(1)

		batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
		# Add a dummy background class to the backend when using sigmoid
		# remind that we set FG labels to [0, num_class-1] since mmdet v2.0
		# BG cat_id: num_class
		if self.use_sigmoid_cls:
			padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
			batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

		if with_nms:
			det_results = []
			for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes, batch_mlvl_scores):
				det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
				det_results.append(tuple([det_bbox, det_label]))
		else:
			det_results = [tuple(mlvl_bs) for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)]
		return det_results
