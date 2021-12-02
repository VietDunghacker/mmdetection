import types
import warnings

import torch
from torch import nn
from torch.nn import functional as F

from mmdet.core import multi_apply
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.dense_heads.fcos_head import FCOSHead
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.models.dense_heads.gfl_head import GFLHead
from mmdet.models.dense_heads.vfnet_head import VFNetHead
from mmdet.models.dense_heads.paa_head import PAAHead
from mmdet.models.dense_heads.reppoints_v2_head import RepPointsV2Head

def anchorfree_forward_features(self, feats):
	return multi_apply(self.forward_feature_single, feats)


def anchorfree_forward_feature_single(self, x):
	if isinstance(x, list):
		cls_feat = x[0]
		reg_feat = x[1]
	else:
		cls_feat = x
		reg_feat = x
	for cls_layer in self.cls_convs:
		cls_feat = cls_layer(cls_feat)
	for reg_layer in self.reg_convs:
		reg_feat = reg_layer(reg_feat)
	return cls_feat, reg_feat


def atss_forward_predictions(self, cls_feats, reg_feats):
	return multi_apply(self.forward_prediction_single, cls_feats, reg_feats, self.scales)


def atss_forward_prediction_single(self, cls_feat, reg_feat, scale):
	cls_score = self.atss_cls(cls_feat)
	bbox_pred = scale(self.atss_reg(reg_feat)).float()
	centerness = self.atss_centerness(reg_feat)
	return cls_score, bbox_pred, centerness

def fcos_forward_predictions(self, cls_feats, reg_feats):
	return multi_apply(self.forward_prediction_single, cls_feats, reg_feats,
					   self.scales, self.strides)

def fcos_forward_prediction_single(self, cls_feat, reg_feat, scale, stride):
	cls_score = self.conv_cls(cls_feat)
	bbox_pred = self.conv_reg(reg_feat)
	if self.centerness_on_reg:
		centerness = self.conv_centerness(reg_feat)
	else:
		centerness = self.conv_centerness(cls_feat)
	# scale the bbox_pred of different level
	# float to avoid overflow when enabling FP16
	bbox_pred = scale(bbox_pred).float()
	if self.norm_on_bbox:
		bbox_pred = F.relu(bbox_pred)
		if not self.training:
			bbox_pred *= stride
	else:
		bbox_pred = bbox_pred.exp()
	return cls_score, bbox_pred, centerness


def retina_forward_predictions(self, cls_feats, reg_feats):
	return multi_apply(self.forward_prediction_single, cls_feats, reg_feats)


def retina_forward_prediction_single(self, cls_feat, reg_feat):
	cls_score = self.retina_cls(cls_feat)
	bbox_pred = self.retina_reg(reg_feat)
	return cls_score, bbox_pred

def gfl_forward_predictions(self, cls_feats, reg_feats):
	return multi_apply(self.forward_prediction_single, cls_feats, reg_feats, self.scales)

def gfl_forward_prediction_single(self, cls_feat, reg_feat, scale):
	cls_score = self.gfl_cls(cls_feat)
	bbox_pred = scale(self.gfl_reg(reg_feat)).float()

	if self.use_dgqp:
		N, C, H, W = bbox_pred.size()
		prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W), dim=2)
		prob_topk, _ = prob.topk(self.reg_topk, dim=2)
		if self.add_mean:
			prob_topk_mean = prob_topk.mean(dim=2, keepdim=True)
			stat = torch.cat([prob_topk, prob_topk_mean], dim=2)
		else:
			stat = prob_topk
		quality_score = self.reg_conf(stat.type_as(reg_feat).reshape(N, -1, H, W))
		cls_score = cls_score.sigmoid() * quality_score
	return cls_score, bbox_pred

def vfnet_forward_predictions(self, cls_feats, reg_feats):
	return multi_apply(self.forward_prediction_single, cls_feats, reg_feats, self.scales, self.scales_refine, self.strides, self.reg_denoms)

def vfnet_forward_prediction_single(self, cls_feat, reg_feat, scale, scale_refine, stride, reg_denom):
	reg_feat_init = self.vfnet_reg_conv(reg_feat)
	if self.bbox_norm_type == 'reg_denom':
		bbox_pred = scale(self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
	elif self.bbox_norm_type == 'stride':
		bbox_pred = scale(self.vfnet_reg(reg_feat_init)).float().exp() * stride
	else:
		raise NotImplementedError

	# compute star deformable convolution offsets
	# converting dcn_offset to reg_feat.dtype thus VFNet can be
	# trained with FP16
	dcn_offset = self.star_dcn_offset(bbox_pred, self.gradient_mul, stride).to(reg_feat.dtype)

	# refine the bbox_pred
	reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset))
	bbox_pred_refine = scale_refine(self.vfnet_reg_refine(reg_feat)).float().exp()
	bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

	# predict the iou-aware cls score
	cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset))
	cls_score = self.vfnet_cls(cls_feat)

	return cls_score, bbox_pred, bbox_pred_refine

def reppointsv2_forward_predictions(self, cls_feats, pts_feats):
	return multi_apply(self.forward_prediction_single, cls_feats, pts_feats)

def reppointsv2_forward_prediction_single(self, cls_feat, pts_feat):
	dcn_base_offset = self.dcn_base_offset.type_as(cls_feat)
	# If we use center_init, the initial reppoints is from center points.
	# If we use bounding bbox representation, the initial reppoints is
	#   from regular grid placed on a pre-defined bbox.
	if self.use_grid_points or not self.center_init:
		scale = self.point_base_scale / 2
		points_init = dcn_base_offset / dcn_base_offset.max() * scale
		bbox_init = x.new_tensor([-scale, -scale, scale, scale]).view(1, 4, 1, 1)
	else:
		points_init = 0

	shared_feat = pts_feat
	for shared_conv in self.shared_convs:
		shared_feat = shared_conv(shared_feat)

	sem_feat = shared_feat
	hem_feat = shared_feat

	sem_scores_out = self.reppoints_sem_out(sem_feat)
	sem_feat = self.reppoints_sem_embedding(sem_feat)

	cls_feat = cls_feat + sem_feat
	pts_feat = pts_feat + sem_feat
	hem_feat = hem_feat + sem_feat

	# generate heatmap and offset
	hem_tl_feat = self.hem_tl(hem_feat)
	hem_br_feat = self.hem_br(hem_feat)

	hem_tl_score_out = self.reppoints_hem_tl_score_out(hem_tl_feat)
	hem_tl_offset_out = self.reppoints_hem_tl_offset_out(hem_tl_feat)
	hem_br_score_out = self.reppoints_hem_br_score_out(hem_br_feat)
	hem_br_offset_out = self.reppoints_hem_br_offset_out(hem_br_feat)

	hem_score_out = torch.cat([hem_tl_score_out, hem_br_score_out], dim=1)
	hem_offset_out = torch.cat([hem_tl_offset_out, hem_br_offset_out], dim=1)

	# initialize reppoints
	pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))
	if self.use_grid_points:
		pts_out_init, bbox_out_init = self.gen_grid_from_reg(pts_out_init, bbox_init.detach())
	else:
		pts_out_init = pts_out_init + points_init
	# refine and classify reppoints
	pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
	dcn_offset = pts_out_init_grad_mul - dcn_base_offset

	hem_feat = torch.cat([hem_score_out, hem_offset_out], dim=1)
	cls_feat = torch.cat([cls_feat, hem_feat], dim=1)
	pts_feat = torch.cat([pts_feat, hem_feat], dim=1)

	cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
	pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
	if self.use_grid_points:
		pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(pts_out_refine, bbox_out_init.detach())
	else:
		pts_out_refine = pts_out_refine + pts_out_init.detach()
	return cls_out, pts_out_init, pts_out_refine, hem_score_out, hem_offset_out, sem_scores_out

def assign_methods_for_bvr(module: nn.Module):
	"""Modify the structure of bbox_head by assigning methods.

	Args:
		module: bbox_head
	"""
	# check whether the bbox_head already has required methods
	if hasattr(module, 'forward_features') and hasattr(module, 'forward_predictions'):
		return

	# check whether BVR supports the bbox_head
	supported_heads = ('ATSSHead', 'FCOSHead', 'RetinaHead', 'GFLHead', 'VFNetHead', 'PAAHead', 'RepPointsV2Head')
	module_name = module.__class__.__name__
	assert module_name in supported_heads, 'not supported bbox_head'
	assert hasattr(module, 'cls_convs'), 'not found cls_convs'
	assert hasattr(module, 'reg_convs'), 'not found reg_convs'

	# warning
	warnings.warn(f'Methods for BVR will be assigned to {module_name}.'
				  ' The bbox_head may break if the prediction'
				  ' is maintained by other branches.')

	# assign forward_features, forward_feature_single
	module.forward_features = types.MethodType(anchorfree_forward_features, module)
	module.forward_feature_single = types.MethodType(anchorfree_forward_feature_single, module)

	# assign forward_predictions, forward_prediction_single
	if isinstance(module, VFNetHead):
		module.forward_predictions = types.MethodType(vfnet_forward_predictions, module)
		module.forward_prediction_single = types.MethodType(vfnet_forward_prediction_single, module)	
	elif isinstance(module, PAAHead):
		module.forward_predictions = types.MethodType(atss_forward_predictions, module)
		module.forward_prediction_single = types.MethodType(atss_forward_prediction_single, module)
	elif isinstance(module, ATSSHead):
		module.forward_predictions = types.MethodType(atss_forward_predictions, module)
		module.forward_prediction_single = types.MethodType(atss_forward_prediction_single, module)
	elif isinstance(module, FCOSHead):
		module.forward_predictions = types.MethodType(fcos_forward_predictions, module)
		module.forward_prediction_single = types.MethodType(fcos_forward_prediction_single, module)
	elif isinstance(module, RetinaHead):
		module.forward_predictions = types.MethodType(retina_forward_predictions, module)
		module.forward_prediction_single = types.MethodType(retina_forward_prediction_single, module)
	elif isinstance(module, GFLHead):
		module.forward_predictions = types.MethodType(gfl_forward_predictions, module)
		module.forward_prediction_single = types.MethodType(gfl_forward_prediction_single, module)
	elif isinstance(module, RepPointsV2Head):
		module.forward_predictions = types.MethodType(reppointsv2_forward_predictions, module)
		module.forward_prediction_single = types.MethodType(reppointsv2_forward_prediction_single, module)
	else:
		assert False, 'this line should be unreachable'

	return
