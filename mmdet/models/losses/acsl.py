import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from ..builder import LOSSES

@LOSSES.register_module()
class ACSL(nn.Module):
	def __init__(self, use_sigmoid = False, score_thr=0.7, num_classes = 1203, json_file='./data/lvis/lvis_v0.5_train.json', loss_weight=1.0):
		super(ACSL, self).__init__()
		assert use_sigmoid == False, 'Does not support using sigmoid'
		self.use_sigmoid = use_sigmoid

		self.score_thr = score_thr
		assert self.score_thr > 0 and self.score_thr < 1
		self.loss_weight = loss_weight

		assert len(json_file) != 0
		self.num_classes = num_classes
		self.freq_group = self.get_freq_info(json_file)

	def get_freq_info(self, json_file):
		cats = json.load(open(json_file, 'r'))

		freq_dict = {'rare': [], 'common': [], 'freq': []}

		for cat in cats:
			assert cat['id'] >= 1
			if cat['frequency'] == 'r':
				freq_dict['rare'].append(cat['id'] - 1)
			elif cat['frequency'] == 'c':
				freq_dict['common'].append(cat['id'] - 1)
			elif cat['frequency'] == 'f':
				freq_dict['freq'].append(cat['id'] - 1)
			else:
				print('Something wrong with the json file.')

		return freq_dict

	def forward(self, cls_logits, labels, weight=None, avg_factor=None, reduction_override=None, use_sigmoid = True, **kwargs):
		if use_sigmoid:
			func = F.binary_cross_entropy_with_logits
		else:
			func = F.binary_cross_entropy

		device = cls_logits.device
		n_i, n_c = cls_logits.size()
		# expand the labels to all their parent nodes
		target = cls_logits.new_zeros(n_i, n_c)
		# weight mask, decide which class should be ignored
		#weight_mask = cls_logits.new_zeros(n_i, n_c)

		unique_label = torch.unique(labels)

		with torch.no_grad():
			sigmoid_cls_logits = torch.sigmoid(cls_logits) if use_sigmoid else cls_logits
		# for each sample, if its score on unrealated class hight than score_thr, their gradient should not be ignored
		# this is also applied to negative samples
		high_score_inds = torch.nonzero(sigmoid_cls_logits >= self.score_thr)
		weight_mask = torch.sparse_coo_tensor(high_score_inds.t(), cls_logits.new_ones(high_score_inds.shape[0]), size=(n_i, n_c), device=device).to_dense()

		for cls in unique_label:
			cls = cls.item()
			cls_inds = torch.nonzero(labels == cls).squeeze(1)
			if cls == self.num_classes:
				# construct target vector for background samples
				target[cls_inds, self.num_classes] = 1
				# for bg, set the weight of all classes to 1
				weight_mask[cls_inds] = 0

				cls_inds_cpu = cls_inds.cpu()

				# Solve the rare categories, random choost 1/3 bg samples to suppress rare categories
				rare_cats = self.freq_group['rare']
				rare_cats = torch.tensor(rare_cats, device=cls_logits.device)
				choose_bg_num = int(len(cls_inds) * 0.01)
				choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

				tmp_weight_mask = weight_mask[choose_bg_inds]
				tmp_weight_mask[:, rare_cats] = 1

				weight_mask[choose_bg_inds] = tmp_weight_mask

				# Solve the common categories, random choost 2/3 bg samples to suppress rare categories
				common_cats = self.freq_group['common']
				common_cats = torch.tensor(common_cats, device=cls_logits.device)
				choose_bg_num = int(len(cls_inds) * 0.1)
				choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

				tmp_weight_mask = weight_mask[choose_bg_inds]
				tmp_weight_mask[:, common_cats] = 1

				weight_mask[choose_bg_inds] = tmp_weight_mask
				
				# Solve the frequent categories, random choost all bg samples to suppress rare categories
				freq_cats = self.freq_group['freq']
				freq_cats = torch.tensor(freq_cats, device=cls_logits.device)
				choose_bg_num = int(len(cls_inds) * 1.0)
				choose_bg_inds = torch.tensor(np.random.choice(cls_inds_cpu, size=(choose_bg_num), replace=False), device=device)

				tmp_weight_mask = weight_mask[choose_bg_inds]
				tmp_weight_mask[:, freq_cats] = 1

				weight_mask[choose_bg_inds] = tmp_weight_mask

				# Set the weight for bg to 1
				weight_mask[cls_inds, self.num_classes] = 1
				
			else:
				# construct target vector for foreground samples
				cur_labels = [cls]
				cur_labels = torch.tensor(cur_labels, device=cls_logits.device)
				tmp_label_vec = cls_logits.new_zeros(n_c)
				tmp_label_vec[cur_labels] = 1
				tmp_label_vec = tmp_label_vec.expand(cls_inds.numel(), n_c)
				target[cls_inds] = tmp_label_vec
				# construct weight mask for fg samples
				tmp_weight_mask_vec = weight_mask[cls_inds]
				# set the weight for ground truth category
				tmp_weight_mask_vec[:, cur_labels] = 1

				weight_mask[cls_inds] = tmp_weight_mask_vec

		cls_loss = func(cls_logits, target.float(), reduction='none')

		return torch.sum(weight_mask * cls_loss) / n_i
