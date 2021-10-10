# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import force_fp32

from mmdet.core import bbox2distance, distance2bbox, multi_apply, multiclass_nms, reduce_mean
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.models import HEADS
from mmdet.models.dense_heads import GFLHead
from mmdet.models.dense_heads.gfl_head import Integral

EPS = 1e-12
try:
	import sklearn.mixture as skm
except ImportError:
	skm = None


def levels_to_images(mlvl_tensor):
	"""Concat multi-level feature maps by image.

	[feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
	Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
	(N, H*W , C), then split the element to N elements with shape (H*W, C), and
	concat elements in same image of all level along first dimension.

	Args:
		mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
			corresponding level. Each element is of shape (N, C, H, W)

	Returns:
		list[torch.Tensor]: A list that contains N tensors and each tensor is
			of shape (num_elements, C)
	"""
	batch_size = mlvl_tensor[0].size(0)
	batch_list = [[] for _ in range(batch_size)]
	channels = mlvl_tensor[0].size(1)
	for t in mlvl_tensor:
		t = t.permute(0, 2, 3, 1)
		t = t.view(batch_size, -1, channels).contiguous()
		for img in range(batch_size):
			batch_list[img].append(t[img])
	return [torch.cat(item, 0) for item in batch_list]


@HEADS.register_module()
class PAAGFLHead(GFLHead):
	"""Head of PAAAssignment: Probabilistic Anchor Assignment with IoU
	Prediction for Object Detection.

	Code is modified from the `official github repo
	<https://github.com/kkhoot/PAA/blob/master/paa_core
	/modeling/rpn/paa/loss.py>`_.

	More details can be found in the `paper
	<https://arxiv.org/abs/2007.08103>`_ .

	Args:
		topk (int): Select topk samples with smallest loss in
			each level.
		score_voting (bool): Whether to use score voting in post-process.
		covariance_type : String describing the type of covariance parameters
			to be used in :class:`sklearn.mixture.GaussianMixture`.
			It must be one of:

			- 'full': each component has its own general covariance matrix
			- 'tied': all components share the same general covariance matrix
			- 'diag': each component has its own diagonal covariance matrix
			- 'spherical': each component has its own single variance
			Default: 'diag'. From 'full' to 'spherical', the gmm fitting
			process is faster yet the performance could be influenced. For most
			cases, 'diag' should be a good choice.
	"""

	def __init__(self,
				 *args,
				 topk=9,
				 score_voting=True,
				 covariance_type='diag',
				 **kwargs):
		# topk used in paa reassign process
		self.topk = topk
		self.with_score_voting = score_voting
		self.covariance_type = covariance_type
		super(PAAHead, self).__init__(*args, **kwargs)

	@force_fp32(apply_to=('cls_scores', 'bbox_preds', 'iou_preds'))
	def loss(self,
			 cls_scores,
			 bbox_preds,
			 gt_bboxes,
			 gt_labels,
			 img_metas,
			 gt_bboxes_ignore=None):
		"""Compute losses of the head.

		Args:
			cls_scores (list[Tensor]): Box scores for each scale level
				Has shape (N, num_anchors * num_classes, H, W)
			bbox_preds (list[Tensor]): Box energies / deltas for each scale
				level with shape (N, num_anchors * 4, H, W)
			gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
				shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
			gt_labels (list[Tensor]): class indices corresponding to each box
			img_metas (list[dict]): Meta information of each image, e.g.,
				image size, scaling factor, etc.
			gt_bboxes_ignore (list[Tensor] | None): Specify which bounding
				boxes can be ignored when are computing the loss.

		Returns:
			dict[str, Tensor]: A dictionary of loss gmm_assignment.
		"""

		featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
		assert len(featmap_sizes) == self.anchor_generator.num_levels

		device = cls_scores[0].device
		anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
		stride_list = [[torch.full((len(anchor), ), stride, device = device) for anchor, stride in zip(anchor_list[0], self.anchor_generator.strides)] for _ in range(len(anchor_list))]
		label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
		cls_reg_targets = self.get_targets(
			anchor_list,
			valid_flag_list,
			gt_bboxes,
			img_metas,
			gt_bboxes_ignore_list=gt_bboxes_ignore,
			gt_labels_list=gt_labels,
			label_channels=label_channels,
		)
		(labels, labels_weight, bboxes_target, bboxes_weight, pos_inds, pos_gt_index) = cls_reg_targets
		cls_scores = levels_to_images(cls_scores)
		cls_scores = [item.reshape(-1, self.cls_out_channels) for item in cls_scores]
		bbox_preds = levels_to_images(bbox_preds)
		bbox_preds = [item.reshape(-1, 4 * (self.reg_max + 1)) for item in bbox_preds]
		pos_losses_list, = multi_apply(self.get_pos_loss, anchor_list, stride_list, cls_scores, bbox_preds, labels, labels_weight, bboxes_target, bboxes_weight, pos_inds)

		with torch.no_grad():
			reassign_labels, reassign_label_weight, reassign_bbox_weights, num_pos = multi_apply(
					self.paa_reassign,
					pos_losses_list,
					labels,
					labels_weight,
					bboxes_weight,
					pos_inds,
					pos_gt_index,
					anchor_list)
			num_pos = sum(num_pos)
		# convert all tensor list to a flatten tensor
		cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))
		bbox_preds = torch.cat(bbox_preds, 0).view(-1, bbox_preds[0].size(-1))
		labels = torch.cat(reassign_labels, 0).view(-1)
		flatten_anchors = torch.cat([torch.cat(item, 0) for item in anchor_list])
		flatten_strides = torch.cat([torch.cat(item, 0) for item in stride_list])
		labels_weight = torch.cat(reassign_label_weight, 0).view(-1)
		bboxes_target = torch.cat(bboxes_target, 0).view(-1, bboxes_target[0].size(-1))

		pos_inds_flatten = ((labels >= 0) & (labels < self.num_classes)).nonzero().reshape(-1)
		scores = label_weights.new_zeros(labels.shape)

		if num_pos:
			pos_bbox_targets = bboxes_target[pos_inds_flatten]
			pos_bbox_pred = bbox_preds[pos_inds_flatten]
			pos_anchors = flatten_anchors[pos_inds_flatten]
			pos_strides = flatten_strides[pos_inds_flatten]
			pos_anchor_centers = self.anchor_center(pos_anchors) / pos_strides

			weight_targets = cls_scores.detach()
			if not self.use_dgqp:
				weight_targets = weight_targets.sigmoid()
			weight_targets = weight_targets.max(dim=1)[0][pos_inds_flatten]
			pos_bbox_pred_corners = self.integral(pos_bbox_pred)
			pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred_corners)
			pos_decode_bbox_targets = pos_bbox_targets / pos_strides
			scores[pos_inds_flatten] = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)
			pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
			target_corners = bbox2distance(pos_anchor_centers, pos_decode_bbox_targets, self.reg_max).reshape(-1)

			# regression loss
			avg_factor = reduce_mean(weight_targets.sum()).clamp_(min = 1).item()
			losses_bbox = self.loss_bbox(
				pos_decode_bbox_pred,
				pos_decode_bbox_targets,
				weight=weight_targets,
				avg_factor=avg_factor)

			# dfl loss
			losses_dfl = self.loss_dfl(
				pred_corners,
				target_corners,
				weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
				avg_factor=4.0 * avg_factor)
		else:
			losses_bbox = bbox_preds.sum() * 0
			losses_dfl = bbox_preds.sum() * 0
		losses_cls = self.loss_cls(
			cls_scores,
			(labels, scores),
			labels_weight,
			avg_factor=max(num_pos, 1))  # avoid num_pos=0

		return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

	def get_pos_loss(self, anchors, strides cls_score, bbox_pred, label, label_weight,
					 bbox_target, bbox_weight, pos_inds):
		"""Calculate loss of all potential positive samples obtained from first
		match process.

		Args:
			anchors (list[Tensor]): Anchors of each scale.
			cls_score (Tensor): Box scores of single image with shape
				(num_anchors, num_classes)
			bbox_pred (Tensor): Box energies / deltas of single image
				with shape (num_anchors, 4)
			label (Tensor): classification target of each anchor with
				shape (num_anchors,)
			label_weight (Tensor): Classification loss weight of each
				anchor with shape (num_anchors).
			bbox_target (dict): Regression target of each anchor with
				shape (num_anchors, 4).
			bbox_weight (Tensor): Bbox weight of each anchor with shape
				(num_anchors, 4).
			pos_inds (Tensor): Index of all positive samples got from
				first assign process.

		Returns:
			Tensor: Losses of all positive samples in single image.
		"""
		if not len(pos_inds):
			return cls_score.new([]),
		anchors_all_level = torch.cat(anchors, 0)
		strides_all_level = torch.cat(strides, 0)
		pos_scores = cls_score[pos_inds]
		pos_bbox_pred = bbox_pred[pos_inds]
		pos_label = label[pos_inds]
		pos_label_weight = label_weight[pos_inds]
		pos_bbox_target = bbox_target[pos_inds]
		pos_bbox_weight = bbox_weight[pos_inds]
		pos_anchors = anchors_all_level[pos_inds]
		pos_strides = strides_all_level[pos_inds]
		pos_anchor_centers = self.anchor_center(pos_anchors) / pos_strides

		pos_bbox_pred_corners = self.integral(pos_bbox_pred)
		pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred_corners)
		pos_decode_bbox_targets = pos_bbox_targets / pos_strides
		scores = bbox_overlaps(pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True)

		# to keep loss dimension
		loss_cls = self.loss_cls(
			pos_scores,
			(pos_label, scores),
			pos_label_weight,
			avg_factor=self.loss_cls.loss_weight,
			reduction_override='none')

		loss_bbox = self.loss_bbox(
			pos_bbox_pred,
			pos_bbox_target,
			pos_bbox_weight,
			avg_factor=self.loss_cls.loss_weight,
			reduction_override='none')

		loss_cls = loss_cls.sum(-1)
		pos_loss = loss_bbox + loss_cls
		return pos_loss,

	def paa_reassign(self, pos_losses, label, label_weight, bbox_weight,
					 pos_inds, pos_gt_inds, anchors):
		"""Fit loss to GMM distribution and separate positive, ignore, negative
		samples again with GMM model.

		Args:
			pos_losses (Tensor): Losses of all positive samples in
				single image.
			label (Tensor): classification target of each anchor with
				shape (num_anchors,)
			label_weight (Tensor): Classification loss weight of each
				anchor with shape (num_anchors).
			bbox_weight (Tensor): Bbox weight of each anchor with shape
				(num_anchors, 4).
			pos_inds (Tensor): Index of all positive samples got from
				first assign process.
			pos_gt_inds (Tensor): Gt_index of all positive samples got
				from first assign process.
			anchors (list[Tensor]): Anchors of each scale.

		Returns:
			tuple: Usually returns a tuple containing learning targets.

				- label (Tensor): classification target of each anchor after
				  paa assign, with shape (num_anchors,)
				- label_weight (Tensor): Classification loss weight of each
				  anchor after paa assign, with shape (num_anchors).
				- bbox_weight (Tensor): Bbox weight of each anchor with shape
				  (num_anchors, 4).
				- num_pos (int): The number of positive samples after paa
				  assign.
		"""
		if not len(pos_inds):
			return label, label_weight, bbox_weight, 0
		label = label.clone()
		label_weight = label_weight.clone()
		bbox_weight = bbox_weight.clone()
		num_gt = pos_gt_inds.max() + 1
		num_level = len(anchors)
		num_anchors_each_level = [item.size(0) for item in anchors]
		num_anchors_each_level.insert(0, 0)
		inds_level_interval = np.cumsum(num_anchors_each_level)
		pos_level_mask = []
		for i in range(num_level):
			mask = (pos_inds >= inds_level_interval[i]) & (
				pos_inds < inds_level_interval[i + 1])
			pos_level_mask.append(mask)
		pos_inds_after_paa = [label.new_tensor([])]
		ignore_inds_after_paa = [label.new_tensor([])]
		for gt_ind in range(num_gt):
			pos_inds_gmm = []
			pos_loss_gmm = []
			gt_mask = pos_gt_inds == gt_ind
			for level in range(num_level):
				level_mask = pos_level_mask[level]
				level_gt_mask = level_mask & gt_mask
				value, topk_inds = pos_losses[level_gt_mask].topk(
					min(level_gt_mask.sum(), self.topk), largest=False)
				pos_inds_gmm.append(pos_inds[level_gt_mask][topk_inds])
				pos_loss_gmm.append(value)
			pos_inds_gmm = torch.cat(pos_inds_gmm)
			pos_loss_gmm = torch.cat(pos_loss_gmm)
			# fix gmm need at least two sample
			if len(pos_inds_gmm) < 2:
				continue
			device = pos_inds_gmm.device
			pos_loss_gmm, sort_inds = pos_loss_gmm.sort()
			pos_inds_gmm = pos_inds_gmm[sort_inds]
			pos_loss_gmm = pos_loss_gmm.view(-1, 1).cpu().numpy()
			min_loss, max_loss = pos_loss_gmm.min(), pos_loss_gmm.max()
			means_init = np.array([min_loss, max_loss]).reshape(2, 1)
			weights_init = np.array([0.5, 0.5])
			precisions_init = np.array([1.0, 1.0]).reshape(2, 1, 1)  # full
			if self.covariance_type == 'spherical':
				precisions_init = precisions_init.reshape(2)
			elif self.covariance_type == 'diag':
				precisions_init = precisions_init.reshape(2, 1)
			elif self.covariance_type == 'tied':
				precisions_init = np.array([[1.0]])
			if skm is None:
				raise ImportError('Please run "pip install sklearn" '
								  'to install sklearn first.')
			gmm = skm.GaussianMixture(
				2,
				weights_init=weights_init,
				means_init=means_init,
				precisions_init=precisions_init,
				covariance_type=self.covariance_type)
			gmm.fit(pos_loss_gmm)
			gmm_assignment = gmm.predict(pos_loss_gmm)
			scores = gmm.score_samples(pos_loss_gmm)
			gmm_assignment = torch.from_numpy(gmm_assignment).to(device)
			scores = torch.from_numpy(scores).to(device)

			pos_inds_temp, ignore_inds_temp = self.gmm_separation_scheme(gmm_assignment, scores, pos_inds_gmm)
			pos_inds_after_paa.append(pos_inds_temp)
			ignore_inds_after_paa.append(ignore_inds_temp)

		pos_inds_after_paa = torch.cat(pos_inds_after_paa)
		ignore_inds_after_paa = torch.cat(ignore_inds_after_paa)
		reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_paa).all(1)
		reassign_ids = pos_inds[reassign_mask]
		label[reassign_ids] = self.num_classes
		label_weight[ignore_inds_after_paa] = 0
		bbox_weight[reassign_ids] = 0
		num_pos = len(pos_inds_after_paa)
		return label, label_weight, bbox_weight, num_pos

	def gmm_separation_scheme(self, gmm_assignment, scores, pos_inds_gmm):
		"""A general separation scheme for gmm model.

		It separates a GMM distribution of candidate samples into three
		parts, 0 1 and uncertain areas, and you can implement other
		separation schemes by rewriting this function.

		Args:
			gmm_assignment (Tensor): The prediction of GMM which is of shape
				(num_samples,). The 0/1 value indicates the distribution
				that each sample comes from.
			scores (Tensor): The probability of sample coming from the
				fit GMM distribution. The tensor is of shape (num_samples,).
			pos_inds_gmm (Tensor): All the indexes of samples which are used
				to fit GMM model. The tensor is of shape (num_samples,)

		Returns:
			tuple[Tensor]: The indices of positive and ignored samples.

				- pos_inds_temp (Tensor): Indices of positive samples.
				- ignore_inds_temp (Tensor): Indices of ignore samples.
		"""
		# The implementation is (c) in Fig.3 in origin paper instead of (b).
		# You can refer to issues such as
		# https://github.com/kkhoot/PAA/issues/8 and
		# https://github.com/kkhoot/PAA/issues/9.
		fgs = gmm_assignment == 0
		pos_inds_temp = fgs.new_tensor([], dtype=torch.long)
		ignore_inds_temp = fgs.new_tensor([], dtype=torch.long)
		if fgs.nonzero().numel():
			_, pos_thr_ind = scores[fgs].topk(1)
			pos_inds_temp = pos_inds_gmm[fgs][:pos_thr_ind + 1]
			ignore_inds_temp = pos_inds_gmm.new_tensor([])
		return pos_inds_temp, ignore_inds_temp

	def get_targets(
		self,
		anchor_list,
		valid_flag_list,
		gt_bboxes_list,
		img_metas,
		gt_bboxes_ignore_list=None,
		gt_labels_list=None,
		label_channels=1,
		unmap_outputs=True,
	):
		"""Get targets for PAA head.

		This method is almost the same as `AnchorHead.get_targets()`. We direct
		return the results from _get_targets_single instead map it to levels
		by images_to_levels function.

		Args:
			anchor_list (list[list[Tensor]]): Multi level anchors of each
				image. The outer list indicates images, and the inner list
				corresponds to feature levels of the image. Each element of
				the inner list is a tensor of shape (num_anchors, 4).
			valid_flag_list (list[list[Tensor]]): Multi level valid flags of
				each image. The outer list indicates images, and the inner list
				corresponds to feature levels of the image. Each element of
				the inner list is a tensor of shape (num_anchors, )
			gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
			img_metas (list[dict]): Meta info of each image.
			gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
				ignored.
			gt_labels_list (list[Tensor]): Ground truth labels of each box.
			label_channels (int): Channel of label.
			unmap_outputs (bool): Whether to map outputs back to the original
				set of anchors.

		Returns:
			tuple: Usually returns a tuple containing learning targets.

				- labels (list[Tensor]): Labels of all anchors, each with
					shape (num_anchors,).
				- label_weights (list[Tensor]): Label weights of all anchor.
					each with shape (num_anchors,).
				- bbox_targets (list[Tensor]): BBox targets of all anchors.
					each with shape (num_anchors, 4).
				- bbox_weights (list[Tensor]): BBox weights of all anchors.
					each with shape (num_anchors, 4).
				- pos_inds (list[Tensor]): Contains all index of positive
					sample in all anchor.
				- gt_inds (list[Tensor]): Contains all gt_index of positive
					sample in all anchor.
		"""

		num_imgs = len(img_metas)
		assert len(anchor_list) == len(valid_flag_list) == num_imgs
		concat_anchor_list = []
		concat_valid_flag_list = []
		for i in range(num_imgs):
			assert len(anchor_list[i]) == len(valid_flag_list[i])
			concat_anchor_list.append(torch.cat(anchor_list[i]))
			concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

		# compute targets for each image
		if gt_bboxes_ignore_list is None:
			gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
		if gt_labels_list is None:
			gt_labels_list = [None for _ in range(num_imgs)]
		results = multi_apply(
			self._get_targets_single,
			concat_anchor_list,
			concat_valid_flag_list,
			gt_bboxes_list,
			gt_bboxes_ignore_list,
			gt_labels_list,
			img_metas,
			label_channels=label_channels,
			unmap_outputs=unmap_outputs)

		(labels, label_weights, bbox_targets, bbox_weights, valid_pos_inds,
		 valid_neg_inds, sampling_result) = results

		# Due to valid flag of anchors, we have to calculate the real pos_inds
		# in origin anchor set.
		pos_inds = []
		for i, single_labels in enumerate(labels):
			pos_mask = (0 <= single_labels) & (single_labels < self.num_classes)
			pos_inds.append(pos_mask.nonzero().view(-1))

		gt_inds = [item.pos_assigned_gt_inds for item in sampling_result]
		return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, gt_inds)

	def _get_targets_single(self,
							flat_anchors,
							valid_flags,
							gt_bboxes,
							gt_bboxes_ignore,
							gt_labels,
							img_meta,
							label_channels=1,
							unmap_outputs=True):
		"""Compute regression and classification targets for anchors in a
		single image.

		This method is same as `AnchorHead._get_targets_single()`.
		"""
		assert unmap_outputs, 'We must map outputs back to the original' \
			'set of anchors in PAAhead'
		return super(ATSSHead, self)._get_targets_single(
			flat_anchors,
			valid_flags,
			gt_bboxes,
			gt_bboxes_ignore,
			gt_labels,
			img_meta,
			label_channels=1,
			unmap_outputs=True)
