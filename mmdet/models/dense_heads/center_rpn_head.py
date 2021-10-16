import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmcv.cnn import Scale
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, distance2bbox, reduce_mean
from mmdet.models import HEADS, build_loss

from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin

INF = 100000000

@HEADS.register_module()
class CenterRPNHead(BaseDenseHead, BBoxTestMixin):
	"""Objects as Points Head. CenterHead use center_point to indicate object's
	position. Paper link <https://arxiv.org/abs/1904.07850>
	Args:
		in_channel (int): Number of channel in the input feature map.
		feat_channel (int): Number of channel in the intermediate feature map.
		num_classes (int): Number of categories excluding the background
			category.
		loss_center_heatmap (dict | None): Config of center heatmap loss.
			Default: GaussianFocalLoss.
		loss_wh (dict | None): Config of wh loss. Default: L1Loss.
		loss_offset (dict | None): Config of offset loss. Default: L1Loss.
		train_cfg (dict | None): Training config. Useless in CenterNet,
			but we keep this variable for SingleStageDetector. Default: None.
		test_cfg (dict | None): Testing config of CenterNet. Default: None.
		init_cfg (dict or list[dict], optional): Initialization config dict.
			Default: None
	"""

	def __init__(self,
				 in_channel,
				 num_classes,
				 num_cls_convs,
				 num_box_convs,
				 num_share_convs,
				 use_deformable=False,
				 hm_min_overlap=0.8,
				 min_raduius=4,
				 strides=[8, 16, 32, 64, 128],
				 regress_ranges=[[-1, 80], [64, 160], [128, 320], [256, 640], [512, INF]],
				 not_norm_reg=False,
				 with_agn_hm=False,
				 only_proposal=False,
				 center_nms=False,
				 more_pos=False,
				 more_pos_thresh=0.001,
				 more_pos_topk=9,
				 loss_center_heatmap=dict(
					 type='CustomGaussianFocalLoss',
					 alpha=0.25,
					 ignore_high_fp=0.85,
					 loss_weight=0.5),
				 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
				 conv_cfg=None,
				 norm_cfg=dict(type='BN', requires_grad = True),
				 act_cfg=dict(type='ReLU'),
				 train_cfg=None,
				 test_cfg=None,
				 init_cfg=None):
		super(CenterRPNHead, self).__init__(init_cfg)
		self.out_kernel = 3
		self.conv_cfg = conv_cfg
		self.norm_cfg = norm_cfg
		self.act_cfg = act_cfg
		self.in_channel = in_channel
		self.num_classes = num_classes

		self.min_radius = 4
		self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)

		self.strides = strides
		self.regress_ranges = regress_ranges

		self.with_agn_hm = with_agn_hm
		self.only_proposal = only_proposal
		self.not_norm_reg = True

		self.loss_center_heatmap = build_loss(loss_center_heatmap)
		self.loss_bbox = build_loss(loss_bbox)

		self.train_cfg = train_cfg
		self.test_cfg = test_cfg
		if self.train_cfg:
			self.nms_thresh_train = self.train_cfg.nms_thresh_train
			self.nms_thresh_test = self.train_cfg.nms_thresh_test
			self.pre_nms_topk_train = self.train_cfg.pre_nms_topk_train
			self.post_nms_topk_train = self.train_cfg.post_nms_topk_train
			self.pre_nms_topk_test = self.train_cfg.pre_nms_topk_test
			self.post_nms_topk_test = self.train_cfg.post_nms_topk_test

		self.fp16_enabled = False
		self.center_nms = center_nms

		self.more_pos_topk = more_pos_topk
		self.more_pos_thresh = 0.0001
		if self.center_nms:
			self.not_nms = True

		self.num_cls_convs = num_cls_convs if not with_only_proposal else 0
		self.num_box_convs = num_box_convs
		self.num_share_convs = num_share_convs
		self.use_deformable = use_deformable
		self.in_channels = in_channels

		self._init_layers()

	def _init_layers(self):
		self.cls_convs = ModuleList()
		self.box_convs = ModuleList()
		self.share_convs = ModuleList()

		for i in range(self.num_cls_convs):
			if self.use_deformable:
				conv_cfg = dict(type='DCNv2', deform_groups=4)
			else:
				conv_cfg = self.conv_cfg
			self.cls_convs.append(
				ConvModule(
					channel,
					channel,
					3,
					stride=1,
					padding=1,
					bias = True,
					conv_cfg=conv_cfg,
					norm_cfg=self.norm_cfg,
					act_cfg=self.act_cfg))

		for i in range(self.num_box_convs):
			if self.use_deformable:
				conv_cfg = dict(type='DCNv2', deform_groups=4)
			else:
				conv_cfg = self.conv_cfg
			self.box_convs.append(
				ConvModule(
					channel,
					channel,
					3,
					stride=1,
					padding=1,
					bias = True,
					conv_cfg=conv_cfg,
					norm_cfg=self.norm_cfg,
					act_cfg=self.act_cfg))

		for i in range(self.num_share_convs):
			if self.use_deformable:
				conv_cfg = dict(type='DCNv2', deform_groups=4)
			else:
				conv_cfg = self.conv_cfg
			self.share_convs.append(
				ConvModule(
					channel,
					channel,
					3,
					stride=1,
					padding=1,
					bias = True,
					conv_cfg=conv_cfg,
					norm_cfg=self.norm_cfg,
					act_cfg=self.act_cfg))

		self.scales = nn.ModuleList([Scale(scale=1.0) for _ in range(len(self.strides))])
		self.heatmap_head = nn.Conv2d(self.in_channels, 1, kernel_size=self.out_kernel, stride=1, padding=self.out_kernel // 2)
		self.rpn_reg = nn.Conv2d(self.in_channels, 4, kernel_size=self.out_kernel, stride=1, padding=self.out_kernel // 2)

	def init_weights(self):
		"""Initialize weights of the head."""
		for modules in [self.cls_convs, self.bbox_convs, self.share_convs]:
			for layer in modules.modules():
				if isinstance(layer, nn.Conv2d):
					nn.init.normal_(layer.weight, std=0.01)
					nn.init.constant_(layer.bias, 0)

		nn.init.normal_(self.rpn_reg.weight, std=0.01)
		nn.init.constant_(self.rpn_reg.bias, 8.)

		prior_prob = 0.01
		bias_value = -math.log((1 - prior_prob) / prior_prob)
		nn.init.constant_(self.heatmap_head.bias, bias_value)
		nn.init.normal_(self.heatmap_head.weight, std=0.01)

	def forward(self, feats):
		"""Forward features. Notice CenterNet head does not use FPN.
		Args:
			feats (tuple[Tensor]): Features from the upstream network, each is
				a 4D-tensor.
		Returns:
			rpn_cls_score (List[Tensor]): cls predict for
				all levels, the channels number is num_classes.
			rpn_bbox_reg (List[Tensor]): bbox_reg predicts for all levels,
				the channels number is 4.
			center_heatmap_pred (List[Tensor]): agn_hms predicts for all levels,
				the channels number is 1.
		"""
		return multi_apply(self.forward_single, feats, self.scales)

	def forward_single(self, feat, scale):
		"""Forward feature of a single level.
		Args:
			feat (Tensor): Feature of a single level.
		Returns:
			rpn_cls_score (Tensor): cls predicts, the channels number is class number: 80
			rpn_bbox_reg (Tensor): reg predicts, the channels number is 4
			center_heatmap_pred (Tensor): center predict heatmaps, the channels number is 1
		"""

		feat = self.share_convs(feat)
		cls_feat = self.cls_convs(feat)
		reg_feat = self.bbox_convs(feat)

		if not self.only_proposal:
			rpn_cls_score = self.cls_logits(cls_feat)
		else:
			rpn_cls_score = None
		if self.with_agn_hm:
			center_heatmap_pred = self.heatmap_head(reg_feat)
		else:
			center_heatmap_pred = None
		rpn_bbox_reg = F.relu(scale(self.bbox_pred(reg_feat)), inplace=True)
		return rpn_cls_score, rpn_bbox_reg, center_heatmap_pred

	def loss(self,
			 cls_scores,
			 bbox_preds,
			 center_heatmap_preds,
			 gt_bboxes,
			 gt_labels,
			 img_metas,
			 gt_bboxes_ignore=None):
		"""Compute losses of the dense head.
		Args:
			bbox_preds (list[Tensor]): reg predicts for all levels with
			   shape (B, 4, H, W).
			center_heatmap_preds (list[Tensor]): center predict heatmaps for
			   all levels with shape (B, 1, H, W).
			gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
				shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
			gt_labels (list[Tensor]): class indices corresponding to each box.
			img_metas (list[dict]): Meta information of each image, e.g.,
				image size, scaling factor, etc.
			gt_bboxes_ignore (None | list[Tensor]): specify which bounding
				boxes can be ignored when computing the loss. Default: None
		Returns:
			dict[str, Tensor]: which has components below:
				- loss_centernet_loc (Tensor): loss of center heatmap.
				- loss_centernet_agn_pos (Tensor): loss of
				- loss_centernet_agn_neg (Tensor): loss of.
		"""
		grids = self.compute_grids(center_heatmap_preds)
		featmap_sizes = grids[0].new_tensor([(x.shape[2], x.shape[3]) for x in bbox_preds])
		pos_inds, labels, bbox_targets, heatmap_targets = self.get_targets(grids, featmap_sizes, gt_bboxes, gt_labels)
		flatten_bbox_pred = torch.cat([x.permute(0, 2, 3, 1).reshape(-1, 4) for x in bbox_preds], 0)
		flatten_center_heatmap_pred = torch.cat([x.permute(0, 2, 3, 1).reshape(-1) for x in center_heatmap_preds], 0) if self.with_agn_hm else None
		flatten_points = torch.cat([points.repeat(len(img_metas), 1) for points in grids])
		if self.more_pos:
			# add more pixels as positive if \
			#   1. they are within the center3x3 region of an object
			#   2. their regression losses are small (<self.more_pos_thresh)
			pos_inds, labels = self._add_more_pos(reg_pred, gt_bboxes, gt_labels, featmap_sizes)

		assert (torch.isfinite(flatten_bbox_pred).all().item())
		num_pos_local = torch.tensor(len(pos_inds), dtype=torch.float, device=flatten_bbox_pred[0].device)
		num_pos_avg = max(reduce_mean(num_pos_local), 1.0)

		losses = {}
		reg_inds = torch.nonzero(bbox_targets.max(dim=1)[0] >= 0).squeeze(1)
		flatten_bbox_pred = flatten_bbox_pred[reg_inds]
		reg_targets_pos = bbox_targets[reg_inds]
		flatten_points_pos = flatten_points[reg_inds]  # added by mmz
		reg_weight_map = heatmap_targets.max(dim=1)[0]
		reg_weight_map = reg_weight_map[reg_inds]
		reg_weight_map = reg_weight_map * 0 + 1 if self.not_norm_reg else reg_weight_map
		reg_norm = max(reduce_mean(reg_weight_map.sum()).item(), 1.0)

		# added by mmz
		pos_decoded_bbox_preds = distance2bbox(flatten_points_pos, flatten_bbox_pred)
		pos_decoded_target_preds = distance2bbox(flatten_points_pos, reg_targets_pos)
		bbox_loss = self.loss_bbox(
			pos_decoded_bbox_preds,
			pos_decoded_target_preds,
			weight=reg_weight_map,
			avg_factor=reg_norm)
		if self.with_agn_hm:
			agn_heatmap_loss = self.loss_center_heatmap(
				flatten_center_heatmap_pred.sigmoid(),
				heatmap_targets.max(dim=1)[0],
				pos_inds,
				avg_factor=num_pos_avg
			)

		losses['loss_rpn_bbox'] = bbox_loss
		losses['loss_rpn_heatmap'] = agn_heatmap_loss
		return losses

	def compute_grids(self, agn_hm_pred_per_level):
		grids = []
		for level, agn_hm_pred in enumerate(agn_hm_pred_per_level):
			h, w = agn_hm_pred.size()[-2:]
			shifts_x = torch.arange(
				0, w * self.strides[level],
				step=self.strides[level],
				dtype=torch.float32, device=agn_hm_pred.device)
			shifts_y = torch.arange(
				0, h * self.strides[level],
				step=self.strides[level],
				dtype=torch.float32, device=agn_hm_pred.device)
			shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
			shift_x = shift_x.reshape(-1)
			shift_y = shift_y.reshape(-1)
			grids_per_level = torch.stack((shift_x, shift_y), dim=1) + self.strides[level] // 2
			grids.append(grids_per_level)
		return grids

	def get_targets(self, grids, featmap_sizes, gt_bboxes, gt_labels):
		'''
		Input:
			grids: list of tensors [(hl x wl, 2)]_l
			featmap_sizes: list of tuples L x 2:
			gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
				shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
		Retuen:
			pos_inds: N
			reg_targets: M x 4
			flattened_hms: M x C or M x 1
			N: number of objects in all images
			M: number of pixels from all FPN levels
		'''

		# get positive pixel index
		if self.more_pos:
			pos_inds, labels = self._get_label_inds(gt_bboxes, gt_labels, featmap_sizes)
		else:
			pos_inds, labels = None, None

		heatmap_channels = self.num_classes
		L = len(grids)
		num_loc_list = [len(loc) for loc in grids]
		strides = torch.cat([featmap_sizes.new_ones(num_loc_list[l_]) * self.strides[l_] for l_ in range(L)]).float()  # M
		reg_size_ranges = torch.cat([featmap_sizes.new_tensor(self.regress_ranges[l_]).float().view(1, 2).expand(num_loc_list[l_], 2) for l_ in range(L)])  # M x 2
		grids = torch.cat(grids, dim=0)  # M x 2
		M = grids.shape[0]

		reg_targets = []
		flattened_hms = []
		for i in range(len(gt_bboxes)):  # images
			boxes = gt_bboxes[i]  # N x 4
			area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

			N = boxes.shape[0]
			if N == 0:
				reg_targets.append(grids.new_zeros((M, 4)) - self.INF)
				flattened_hms.append(
					grids.new_zeros((M, 1 if self.only_proposal else heatmap_channels)))
				continue

			l = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N)  # M x N
			t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N)  # M x N
			r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1)  # M x N
			b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1)  # M x N
			reg_target = torch.stack([l, t, r, b], dim=2)  # M x N x 4

			centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2)  # N x 2
			centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # MxNx2
			strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
			centers_discret = ((centers_expanded / strides_expanded).int() *
							   strides_expanded).float() + strides_expanded / 2

			is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) -
						 centers_discret) ** 2).sum(dim=2) == 0)  # M x N
			is_in_boxes = reg_target.min(dim=2)[0] > 0  # M x N
			is_center3x3 = self.get_center3x3(grids, centers, strides) & is_in_boxes  # M x N
			is_cared_in_the_level = self.assign_reg_fpn(reg_target, reg_size_ranges)  # M x N
			reg_mask = is_center3x3 & is_cared_in_the_level  # M x N

			dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - centers_expanded) ** 2).sum(dim=2)  # M x N
			dist2[is_peak] = 0
			radius2 = self.delta ** 2 * 2 * area  # N
			radius2 = torch.clamp(radius2, min=self.min_radius ** 2)
			weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N)  # M x N
			reg_target = self._get_reg_targets(reg_target, weighted_dist2.clone(), reg_mask, area)  # M x 4

			if self.only_proposal:
				flattened_hm = self._create_agn_heatmaps_from_dist(weighted_dist2.clone()) # M x 1
			else:
				flattened_hm = self._create_heatmaps_from_dist(weighted_dist2.clone(), gt_classes, channels=heatmap_channels) # M x C

			reg_targets.append(reg_target)
			flattened_hms.append(flattened_hm)

		# transpose im first training_targets to level first ones
		# reg_targets = self._transpose(reg_targets, num_loc_list)

		for im_i in range(len(reg_targets)):
			reg_targets[im_i] = torch.split(reg_targets[im_i], num_loc_list, dim=0)

		targets_level_first = []
		for targets_per_level in zip(*reg_targets):
			targets_level_first.append(
				torch.cat(targets_per_level, dim=0))
		reg_targets = targets_level_first

		# flattened_hms = self._transpose(flattened_hms, num_loc_list)
		for im_i in range(len(flattened_hms)):
			flattened_hms[im_i] = torch.split(
				flattened_hms[im_i], num_loc_list, dim=0)

		hms_level_first = []
		for hms_per_level in zip(*flattened_hms):
			hms_level_first.append(
				torch.cat(hms_per_level, dim=0))
		flattened_hms = hms_level_first

		for i in range(len(reg_targets)):
			reg_targets[i] = reg_targets[i] / float(self.strides[i])
		reg_targets = torch.cat([x for x in reg_targets], dim=0)  # MB x 4
		flattened_hms = torch.cat([x for x in flattened_hms], dim=0)  # MB x C

		return pos_inds, labels, reg_targets, flattened_hms

	def _get_label_inds(self, gt_bboxes, gt_labels, featmap_sizes):
		'''
		Inputs:
			gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
				shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
			featmap_sizes: L x 2 [(h_l, w_l)]_L
		Returns:
			pos_inds: N'
		'''
		pos_inds = []
		labels = []
		L = len(self.strides)
		B = len(gt_bboxes)
		featmap_sizes = featmap_sizes.long()
		loc_per_level = (featmap_sizes[:, 0] * featmap_sizes[:, 1]).long()
		level_bases = []
		s = 0
		for l in range(L):
			level_bases.append(s)
			s = s + B * loc_per_level[l]
		level_bases = featmap_sizes.new_tensor(level_bases).long()
		strides_default = featmap_sizes.new_tensor(self.strides).float()
		for im_i in range(B):
			bboxes = gt_bboxes[im_i]  # n x 4
			n = bboxes.shape[0]
			centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2)  # n x 2
			centers = centers.view(n, 1, 2).expand(n, L, 2)
			strides = strides_default.view(1, L, 1).expand(n, L, 2)
			centers_inds = (centers / strides).long()  # n x L x 2
			Ws = featmap_sizes[:, 1].view(1, L).expand(n, L)
			pos_ind = level_bases.view(1, L).expand(n, L) + \
				im_i * loc_per_level.view(1, L).expand(n, L) + \
				centers_inds[:, :, 1] * Ws + \
				centers_inds[:, :, 0]  # n x L
			is_cared_in_the_level = self.assign_fpn_level(bboxes)
			pos_ind = pos_ind[is_cared_in_the_level].view(-1)

			pos_inds.append(pos_ind)  # n'
			labels.append(gt_labels[im_i])
		pos_inds = torch.cat(pos_inds, dim=0).long()
		labels = torch.cat(labels, dim=0)
		return pos_inds, labels  # N

	def assign_fpn_level(self, boxes):
		'''
		Inputs:
			boxes: n x 4
			size_ranges: L x 2
		Return:
			is_cared_in_the_level: n x L
		'''
		size_ranges = boxes.new_tensor(
			self.regress_ranges).view(len(self.regress_ranges), 2)  # Lx2
		crit = ((boxes[:, 2:] - boxes[:, :2]) ** 2).sum(dim=1) ** 0.5 / 2  # n
		n, L = crit.shape[0], size_ranges.shape[0]
		crit = crit.view(n, 1).expand(n, L)
		size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
		is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
			(crit <= size_ranges_expand[:, :, 1])
		return is_cared_in_the_level

	def get_center3x3(self, locations, centers, strides):
		'''
		Inputs:
			locations: M x 2
			centers: N x 2
			strides: M
		'''
		M, N = locations.shape[0], centers.shape[0]
		locations_expanded = locations.view(M, 1, 2).expand(M, N, 2)
		centers_expanded = centers.view(1, N, 2).expand(M, N, 2)  # M x N x 2
		strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)  # M x N
		centers_discret = ((centers_expanded / strides_expanded).int() *
						   strides_expanded).float() + strides_expanded / 2
		dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
		dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
		return (dist_x <= strides_expanded[:, :, 0]) & \
			(dist_y <= strides_expanded[:, :, 0])

	def _get_reg_targets(self, reg_targets, dist, mask, area):
		'''
		  reg_targets (M x N x 4): long tensor
		  dist (M x N)
		  is_*: M x N
		'''
		dist[mask == 0] = INF * 1.0
		min_dist, min_inds = dist.min(dim=1)  # M
		reg_targets_per_im = reg_targets[range(len(reg_targets)), min_inds]  # M x N x 4 --> M x 4
		reg_targets_per_im[min_dist == INF] = - INF
		return reg_targets_per_im

	def assign_reg_fpn(self, reg_targets_per_im, size_ranges):
		'''
		TODO (Xingyi): merge it with assign_fpn_level
		Inputs:
			reg_targets_per_im: M x N x 4
			size_ranges: M x 2
		'''
		crit = ((reg_targets_per_im[:, :, :2] + reg_targets_per_im[:, :, 2:])**2).sum(dim=2) ** 0.5 / 2
		is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & (crit <= size_ranges[:, [1]])
		return is_cared_in_the_level

	def _create_agn_heatmaps_from_dist(self, dist):
		'''
		TODO (Xingyi): merge it with _create_heatmaps_from_dist
		dist: M x N
		return:
		  heatmaps: M x 1
		'''
		heatmaps = dist.new_zeros((dist.shape[0], 1))
		heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
		heatmaps[heatmaps < 1e-4] = 0
		return heatmaps

	def _create_heatmaps_from_dist(self, dist, labels, channels):
		'''
		dist: M x N
		labels: N
		return:
		  heatmaps: M x C
		'''
		heatmaps = dist.new_zeros((dist.shape[0], channels))
		for c in range(channels):
			inds = (labels == c) # N
			if inds.int().sum() == 0:
				continue
			heatmaps[:, c] = torch.exp(-dist[:, inds].min(dim=1)[0])
			zeros = heatmaps[:, c] < 1e-4
			heatmaps[zeros, c] = 0
		return heatmaps

	def get_bboxes(self, cls_scores, bbox_preds, center_heatmap_preds, img_metas, cfg=None):

		grids = self.compute_grids(center_heatmap_preds)
		cls_scores = [x.sigmoid() if x is not None else None for x in cls_scores]
		center_heatmap_preds = [x.sigmoid() for x in center_heatmap_preds]

		if self.only_proposal:
			boxlists = multi_apply(self.predict_single_level, grids, center_heatmap_preds, bbox_preds, self.strides, [None for _ in center_heatmap_preds])
		else:
			boxlists = multi_apply(self.predict_single_level, grids, cls_scores, bbox_preds, self.strides, center_heatmap_preds)
		boxlists = [torch.cat(boxlist) for boxlist in boxlists]
		final_boxlists = []
		for b in range(len(boxlists)):
			final_boxlists.append(self.nms_and_topK(boxlists[b], with_nms=not self.not_nms))
		return final_boxlists

	def predict_single_level(self, grids, heatmap, reg_pred, stride, agn_hm, is_proposal=False):
		N, C, H, W = heatmap.shape
		# put in the same format as grids
		if self.center_nms:
			heatmap_nms = nn.functional.max_pool2d(heatmap, (3, 3), stride=1, padding=1)
			heatmap = heatmap * (heatmap_nms == heatmap).float()
		heatmap = heatmap.permute(0, 2, 3, 1)  # N x H x W x C
		heatmap = heatmap.reshape(N, -1, C)  # N x HW x C

		reg_pred_ = reg_pred * stride
		box_regression = reg_pred_.view(N, 4, H, W).permute(0, 2, 3, 1)
		box_regression = box_regression.reshape(N, -1, 4)

		candidate_inds = heatmap > self.more_pos_thresh  # 0.05
		pre_nms_top_n = candidate_inds.view(N, -1).sum(1)  # N
		pre_nms_topk = self.pre_nms_topk_train \
			if self.training else self.pre_nms_topk_test
		pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk)  # N

		if agn_hm is not None:
			agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
			agn_hm = agn_hm.reshape(N, -1)
			heatmap = heatmap * agn_hm[:, :, None]

		results = []
		for i in range(N):
			per_box_cls = heatmap[i]  # HW x C
			per_candidate_inds = candidate_inds[i]  # n
			per_box_cls = per_box_cls[per_candidate_inds]  # n

			per_candidate_nonzeros = per_candidate_inds.nonzero(as_tuple=False)
			per_box_loc = per_candidate_nonzeros[:, 0]  # n
			per_class = per_candidate_nonzeros[:, 1]  # n

			per_box_regression = box_regression[i]  # HW x 4
			per_box_regression = per_box_regression[per_box_loc]  # n x 4
			per_grids = grids[per_box_loc]  # n x 2

			per_pre_nms_top_n = pre_nms_top_n[i]  # 1

			if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
				per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)
				per_class = per_class[top_k_indices]
				per_box_regression = per_box_regression[top_k_indices]
				per_grids = per_grids[top_k_indices]

			detections = distance2bbox(per_grids, per_box_regression, max_shape=None)
			detections[:, 2] = torch.max(detections[:, 2].clone(), detections[:, 0].clone() + 0.01)
			detections[:, 3] = torch.max(detections[:, 3].clone(), detections[:, 1].clone() + 0.01)
			scores = torch.sqrt(per_box_cls) if self.with_agn_hm else per_box_cls  # n
			boxlist = torch.cat([detections, torch.unsqueeze(scores, 1)], dim=1)
			results.append(boxlist)
		return results

	def nms_and_topK(self, boxlist, max_proposals=-1, with_nms=True):
		result = boxlist
		if with_nms:
			_, keep = batched_nms(boxlist[:, :4], boxlist[:, 4].contiguous(), boxlist[:, -1], self.test_cfg.nms)
			if max_proposals > 0:
				keep = keep[:max_proposals]
			result = result[keep]

		num_dets = len(result)
		post_nms_topk = self.post_nms_topk_train if self.training else self.post_nms_topk_test
		if num_dets > post_nms_topk:
			cls_scores = result[:, 4]
			image_thresh, _ = torch.kthvalue(cls_scores.cpu(), num_dets - post_nms_topk + 1)
			keep = cls_scores >= image_thresh.item()
			keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
			result = result[keep]
		return result

	def _add_more_pos(self, reg_pred, gt_bboxes, gt_labels, featmap_sizes):
		labels, level_masks, c33_inds, c33_masks, c33_regs = self._get_c33_inds(gt_bboxes, gt_labels, featmap_sizes)
		N, L, K = labels.shape[0], len(self.strides), 9
		c33_inds[c33_masks == 0] = 0
		reg_pred_c33 = reg_pred[c33_inds].detach() # N x L x K
		invalid_reg = c33_masks == 0
		c33_regs_expand = c33_regs.view(N * L * K, 4).clamp(min=0)
		if N > 0:
			with torch.no_grad():
				c33_reg_loss = self.loss_bbox(
					reg_pred_c33.view(N * L * K, 4), 
					c33_regs_expand, None,
					reduction='none').view(N, L, K).detach() # N x L x K
		else:
			c33_reg_loss = reg_pred_c33.new_zeros((N, L, K)).detach()
		c33_reg_loss[invalid_reg] = INF # N x L x K
		c33_reg_loss.view(N * L, K)[level_masks.view(N * L), 4] = 0 # real center
		c33_reg_loss = c33_reg_loss.view(N, L * K)
		if N == 0:
			loss_thresh = c33_reg_loss.new_ones((N)).float()
		else:
			loss_thresh = torch.kthvalue(c33_reg_loss, self.more_pos_topk, dim=1)[0] # N
		loss_thresh[loss_thresh > self.more_pos_thresh] = self.more_pos_thresh # N
		new_pos = c33_reg_loss.view(N, L, K) < loss_thresh.view(N, 1, 1).expand(N, L, K)
		pos_inds = c33_inds[new_pos].view(-1) # P
		labels = labels.view(N, 1, 1).expand(N, L, K)[new_pos].view(-1)
		return pos_inds, labels
		
	
	def _get_c33_inds(self, gt_bboxes, gt_labels, featmap_sizes):
		labels = []
		level_masks = []
		c33_inds = []
		c33_masks = []
		c33_regs = []
		L = len(self.strides)
		B = len(gt_labels)
		featmap_sizes = featmap_sizes.long()
		loc_per_level = (featmap_sizes[:, 0] * featmap_sizes[:, 1]).long() # L
		level_bases = []
		s = 0
		for l in range(L):
			level_bases.append(s)
			s = s + B * loc_per_level[l]
		level_bases = featmap_sizes.new_tensor(level_bases).long() # L
		strides_default = featmap_sizes.new_tensor(self.strides).float() # L
		K = 9
		dx = featmap_sizes.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).long()
		dy = featmap_sizes.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).long()
		for im_i in range(B):
			bboxes = gt_bboxes[im_i] # n x 4
			n = bboxes.shape[0]
			if n == 0:
				continue
			centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
			centers = centers.view(n, 1, 2).expand(n, L, 2)

			strides = strides_default.view(1, L, 1).expand(n, L, 2) # 
			centers_inds = (centers / strides).long() # n x L x 2
			center_grids = centers_inds * strides + strides // 2# n x L x 2
			l = center_grids[:, :, 0] - bboxes[:, 0].view(n, 1).expand(n, L)
			t = center_grids[:, :, 1] - bboxes[:, 1].view(n, 1).expand(n, L)
			r = bboxes[:, 2].view(n, 1).expand(n, L) - center_grids[:, :, 0]
			b = bboxes[:, 3].view(n, 1).expand(n, L) - center_grids[:, :, 1] # n x L
			reg = torch.stack([l, t, r, b], dim=2) # n x L x 4
			reg = reg / strides_default.view(1, L, 1).expand(n, L, 4).float()
			
			Ws = featmap_sizes[:, 1].view(1, L).expand(n, L)
			Hs = featmap_sizes[:, 0].view(1, L).expand(n, L)
			expand_Ws = Ws.view(n, L, 1).expand(n, L, K)
			expand_Hs = Hs.view(n, L, 1).expand(n, L, K)
			label = gt_labels[im_i].view(n).clone()
			mask = reg.min(dim=2)[0] >= 0 # n x L
			mask = mask & self.assign_fpn_level(bboxes)
			labels.append(label) # n
			level_masks.append(mask) # n x L

			Dy = dy.view(1, 1, K).expand(n, L, K)
			Dx = dx.view(1, 1, K).expand(n, L, K)
			c33_ind = level_bases.view(1, L, 1).expand(n, L, K) + \
					   im_i * loc_per_level.view(1, L, 1).expand(n, L, K) + \
					   (centers_inds[:, :, 1:2].expand(n, L, K) + Dy) * expand_Ws + \
					   (centers_inds[:, :, 0:1].expand(n, L, K) + Dx) # n x L x K
			
			c33_mask = \
				((centers_inds[:, :, 1:2].expand(n, L, K) + dy) < expand_Hs) & \
				((centers_inds[:, :, 1:2].expand(n, L, K) + dy) >= 0) & \
				((centers_inds[:, :, 0:1].expand(n, L, K) + dx) < expand_Ws) & \
				((centers_inds[:, :, 0:1].expand(n, L, K) + dx) >= 0)
			# TODO (Xingyi): think about better way to implement this
			# Currently it hard codes the 3x3 region
			c33_reg = reg.view(n, L, 1, 4).expand(n, L, K, 4).clone()
			c33_reg[:, :, [0, 3, 6], 0] -= 1
			c33_reg[:, :, [0, 3, 6], 2] += 1
			c33_reg[:, :, [2, 5, 8], 0] += 1
			c33_reg[:, :, [2, 5, 8], 2] -= 1
			c33_reg[:, :, [0, 1, 2], 1] -= 1
			c33_reg[:, :, [0, 1, 2], 3] += 1
			c33_reg[:, :, [6, 7, 8], 1] += 1
			c33_reg[:, :, [6, 7, 8], 3] -= 1
			c33_mask = c33_mask & (c33_reg.min(dim=3)[0] >= 0) # n x L x K
			c33_inds.append(c33_ind)
			c33_masks.append(c33_mask)
			c33_regs.append(c33_reg)
		
		if len(level_masks) > 0:
			labels = torch.cat(labels, dim=0)
			level_masks = torch.cat(level_masks, dim=0)
			c33_inds = torch.cat(c33_inds, dim=0).long()
			c33_regs = torch.cat(c33_regs, dim=0)
			c33_masks = torch.cat(c33_masks, dim=0)
		else:
			labels = featmap_sizes.new_zeros((0)).long()
			level_masks = featmap_sizes.new_zeros((0, L)).bool()
			c33_inds = featmap_sizes.new_zeros((0, L, K)).long()
			c33_regs = featmap_sizes.new_zeros((0, L, K, 4)).float()
			c33_masks = featmap_sizes.new_zeros((0, L, K)).bool()
		return labels, level_masks, c33_inds, c33_masks, c33_regs # N x L, N x L x K
