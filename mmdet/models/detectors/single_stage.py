# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
	"""Base class for single-stage detectors.

	Single-stage detectors directly and densely predict bounding boxes on the
	output features of the backbone+neck.
	"""

	def __init__(self,
				 backbone,
				 neck=None,
				 bbox_head=None,
				 train_cfg=None,
				 test_cfg=None,
				 pretrained=None,
				 init_cfg=None):
		super(SingleStageDetector, self).__init__(init_cfg)
		if pretrained:
			warnings.warn('DeprecationWarning: pretrained is deprecated, '
						  'please use "init_cfg" instead')
			backbone.pretrained = pretrained
		self.backbone = build_backbone(backbone)
		self.use_cbnet = hasattr(self.backbone, 'cb_num_modules')
		if self.use_cbnet:
			self.forward_train = self.forward_train_cbnet

		if neck is not None:
			self.neck = build_neck(neck)
		bbox_head.update(train_cfg=train_cfg)
		bbox_head.update(test_cfg=test_cfg)
		self.bbox_head = build_head(bbox_head)
		self.train_cfg = train_cfg
		self.test_cfg = test_cfg

	def extract_feat(self, img):
		"""Directly extract features from the backbone+neck."""
		x = self.backbone(img)
		if self.with_neck:
			x = self.neck(x)
		return x

	def forward_dummy(self, img):
		"""Used for computing network flops.

		See `mmdetection/tools/analysis_tools/get_flops.py`
		"""
		x = self.extract_feat(img)
		outs = self.bbox_head(x)
		return outs

	def forward_train(self,
					  img,
					  img_metas,
					  gt_bboxes,
					  gt_labels,
					  gt_bboxes_ignore=None):
		"""
		Args:
			img (Tensor): Input images of shape (N, C, H, W).
				Typically these should be mean centered and std scaled.
			img_metas (list[dict]): A List of image info dict where each dict
				has: 'img_shape', 'scale_factor', 'flip', and may also contain
				'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
				For details on the values of these keys see
				:class:`mmdet.datasets.pipelines.Collect`.
			gt_bboxes (list[Tensor]): Each item are the truth boxes for each
				image in [tl_x, tl_y, br_x, br_y] format.
			gt_labels (list[Tensor]): Class indices corresponding to each box
			gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
				boxes can be ignored when computing the loss.

		Returns:
			dict[str, Tensor]: A dictionary of loss components.
		"""
		super(SingleStageDetector, self).forward_train(img, img_metas)
		x = self.extract_feat(img)
		losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
		return losses

	def simple_test(self, img, img_metas, rescale=False):
		"""Test function without test-time augmentation.

		Args:
			img (torch.Tensor): Images with shape (N, C, H, W).
			img_metas (list[dict]): List of image information.
			rescale (bool, optional): Whether to rescale the results.
				Defaults to False.

		Returns:
			list[list[np.ndarray]]: BBox results of each image and classes.
				The outer list corresponds to each image. The inner list
				corresponds to each class.
		"""
		feat = self.extract_feat(img)
		results_list = self.bbox_head.simple_test(
			feat, img_metas, rescale=rescale)
		bbox_results = [
			bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
			for det_bboxes, det_labels in results_list
		]
		return bbox_results

	def aug_test(self, imgs, img_metas, rescale=False):
		"""Test function with test time augmentation.

		Args:
			imgs (list[Tensor]): the outer list indicates test-time
				augmentations and inner Tensor should have a shape NxCxHxW,
				which contains all images in the batch.
			img_metas (list[list[dict]]): the outer list indicates test-time
				augs (multiscale, flip, etc.) and the inner list indicates
				images in a batch. each dict has image information.
			rescale (bool, optional): Whether to rescale the results.
				Defaults to False.

		Returns:
			list[list[np.ndarray]]: BBox results of each image and classes.
				The outer list corresponds to each image. The inner list
				corresponds to each class.
		"""
		assert hasattr(self.bbox_head, 'aug_test'), \
			f'{self.bbox_head.__class__.__name__}' \
			' does not support test-time augmentation'

		feats = self.extract_feats(imgs)
		results_list = self.bbox_head.aug_test(
			feats, img_metas, rescale=rescale)
		bbox_results = [
			bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
			for det_bboxes, det_labels in results_list
		]
		return bbox_results

	def onnx_export(self, img, img_metas):
		"""Test function without test time augmentation.

		Args:
			img (torch.Tensor): input images.
			img_metas (list[dict]): List of image information.

		Returns:
			tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
				and class labels of shape [N, num_det].
		"""
		x = self.extract_feat(img)
		outs = self.bbox_head(x)
		# get origin input shape to support onnx dynamic shape

		# get shape as tensor
		img_shape = torch._shape_as_tensor(img)[2:]
		img_metas[0]['img_shape_for_onnx'] = img_shape
		# get pad input shape to support onnx dynamic shape for exporting
		# `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
		# for inference
		img_metas[0]['pad_shape_for_onnx'] = img_shape
		# TODO:move all onnx related code in bbox_head to onnx_export function
		det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

		return det_bboxes, det_labels

	@staticmethod
	def _update_loss_for_cbnet(losses, idx, weight):
		"""update loss for CBNetV2 by replacing keys and weighting values."""
		new_losses = dict()
		for k, v in losses.items():
			if weight == 1:
				new_k = k
			else:
				new_k = f'aux{idx}_{k}'
			if 'loss' in k:
				if isinstance(v, (list, tuple)):
					new_losses[new_k] = [each_v * weight for each_v in v]
				else:
					new_losses[new_k] = v * weight
			else:
				new_losses[new_k] = v
		return new_losses

	def forward_train_cbnet(self,
							img,
							img_metas,
							gt_bboxes,
							gt_labels,
							gt_bboxes_ignore=None,
							gt_masks=None,
							proposals=None,
							**kwargs):
		"""Forward function for training CBNetV2.
		Args:
			img (Tensor): of shape (N, C, H, W) encoding input images.
				Typically these should be mean centered and std scaled.
			img_metas (list[dict]): list of image info dict where each dict
				has: 'img_shape', 'scale_factor', 'flip', and may also contain
				'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
				For details on the values of these keys see
				`mmdet/datasets/pipelines/formatting.py:Collect`.
			gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
				shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
			gt_labels (list[Tensor]): class indices corresponding to each box
			gt_bboxes_ignore (None | list[Tensor]): specify which bounding
				boxes can be ignored when computing the loss.
			gt_masks (None | Tensor) : true segmentation masks for each box
				used if the architecture supports a segmentation task.
			proposals : override rpn proposals with custom proposals. Use when
				`with_rpn` is False.
		Returns:
			dict[str, Tensor]: a dictionary of loss components
		"""
		super(SingleStageDetector, self).forward_train(img, img_metas)
		xs = self.extract_feat(img)

		if not isinstance(xs[0], (list, tuple)):
			xs = [xs]
		cb_loss_weights = self.train_cfg.get('cb_loss_weights')
		if cb_loss_weights is None:
			if len(xs) > 1:
				# refer CBNetV2 paper
				cb_loss_weights = [0.5] + [1] * (len(xs) - 1)
			else:
				cb_loss_weights = [1]
		assert len(cb_loss_weights) == len(xs)

		losses = dict()

		for i, x in enumerate(xs):
			loss = self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
			if len(xs) > 1:
				loss = self._update_loss_for_cbnet(loss, idx=i, weight=cb_loss_weights[i])
			losses.update(loss)

		return losses
