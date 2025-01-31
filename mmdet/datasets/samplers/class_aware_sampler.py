import copy
import itertools
import numpy as np
import random
import torch

from mmcv.utils import print_log

from collections import defaultdict
from terminaltables import AsciiTable
from torch.utils.data import Sampler

class ClassAwareSampler(Sampler):
	def __init__(self, dataset, samples_per_gpu):
		"""
		Args:
			size (int): the total number of data of the underlying dataset to sample from
			seed (int): the initial seed of the shuffle. Must be the same
				across all workers. If None, will use a random seed shared
				among workers (require synchronization among all workers).
		"""
		self.dataset = dataset
		self._size = samples_per_gpu
		assert self._size > 0
		
		category_freq = defaultdict(int)
		self.cw = []
		self.empty_gt = set()
		self.multiple_gt = set()
		from mmdet.datasets.coco import CocoDataset
		from mmdet.datasets.lvis import LVISDataset
		self.coco_style = isinstance(self.dataset, CocoDataset) or isinstance(self.dataset, LVISDataset)
		if hasattr(self.dataset, "dataset") and (isinstance(self.dataset.dataset, CocoDataset) or isinstance(self.dataset.dataset, LVISDataset)):
			self.coco_style = True
		self.coco_style = True

		if self.coco_style:
			for idx in range(len(self.dataset)):
				cat_ids = set(self.dataset.get_cat_ids(idx))
				if len([cat_id for cat_id in cat_ids if cat_id in self.dataset.cat_ids]) == 0:
					self.empty_gt.add(idx)
				for cat_id in cat_ids:
					if cat_id in self.dataset.cat_ids:
						category_freq[self.dataset.cat2label[cat_id]] += 1
		else:
			for idx in range(len(self.dataset)):
				cat_ids = self.dataset.get_cat_ids(idx)
				for cat_id in cat_ids:
					if cat_id in self.dataset.cat_ids:
						category_freq[self.dataset.cat2label[cat_id]] += 1
		self.empty_gt = list(self.empty_gt)
		for i in sorted(category_freq.keys()):
			self.cw.append(1. / category_freq[i])
		assert(len(self.cw) == len(self.dataset.CLASSES))
		self.cw = np.array(self.cw)
		self.orig_cw = copy.deepcopy(self.cw)
		self.cw /= sum(self.cw)

		print("Number of negative samples: {}".format(len(self.empty_gt)))

		num_columns = min(6, len(self.cw)  * 2)
		results_flatten = list(itertools.chain(*[(self.dataset.CLASSES[i], "{:.6f}".format(item)) for i, item in enumerate(self.cw)]))
		headers = ['category', 'weight'] * (num_columns // 2)
		results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
		table_data = [headers]
		table_data += [result for result in results_2d]
		table = AsciiTable(table_data)
		#print_log('\n' + table.table)

		self.weights = self._get_class_balance_factor()

	def __iter__(self):
		yield from itertools.islice(self._infinite_indices(), None)

	def _infinite_indices(self):
		while True:
			ids = torch.multinomial(self.weights, self._size * 100, replacement=False)
			yield from ids
			self.weights = self._get_class_balance_factor()

	def _get_class_balance_factor(self):
		ret = []

		if self.coco_style:
			#ret = [1.0] * len(self.dataset)
			for idx in range(len(self.dataset)):
				cat_ids = self.dataset.get_cat_ids(idx)
				
				if len(cat_ids) == 0:
					t = 1e-6
				else:
					t = max([self.cw[self.dataset.cat2label[cat_id]] for cat_id in cat_ids if cat_id in self.dataset.cat_ids])
				ret.append(t)
		else:
			for idx in range(len(self.dataset)):
				cat_ids = self.dataset.get_cat_ids(idx)
				ret.append(max([self.cw[cat_id] for cat_id in cat_ids if cat_id in self.dataset.cat_ids]) + 1e-6)
		return torch.tensor(ret).float()

	def __len__(self):
		return int(np.ceil(len(self.dataset) / self._size)) * self._size
