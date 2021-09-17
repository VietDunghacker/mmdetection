# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import itertools
import random
import torch
from collections import defaultdict
from functools import partial
from mmcv.utils import print_log
from terminaltables import AsciiTable

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .samplers import DistributedGroupSampler, DistributedSampler, GroupSampler

if platform.system() != 'Windows':
	# https://github.com/pytorch/pytorch/issues/973
	import resource
	rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
	base_soft_limit = rlimit[0]
	hard_limit = rlimit[1]
	soft_limit = min(max(4096, base_soft_limit), hard_limit)
	resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')

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
		from .coco import CocoDataset
		from .lvis import LVISDataset
		if isinstance(dataset, CocoDataset) or isinstance(dataset, LVISDataset):
			for idx in range(len(self.dataset)):
				cat_ids = set(self.dataset.get_cat_ids(idx))
				if len(cat_ids) == 0:
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
		self.cw = np.array(self.cw)
		self.cw /= sum(self.cw)
		self.orig_cw = copy.deepcopy(self.cw)

		num_columns = min(6, len(self.cw * 2))
		results_flatten = list(itertools.chain(*["{:.6f}".format(item) for item in self.cw]))
		headers = ['category', 'weight'] * (num_columns // 2)
		results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
		table_data = [headers]
		table_data += [result for result in results_2d]
		table = AsciiTable(table_data)
		print_log('\n' + table.table)

		self.weights = self._get_class_balance_factor()

	def __iter__(self):
		yield from itertools.islice(self._infinite_indices(), None)

	def _infinite_indices(self):
		while True:
			ids = torch.multinomial(self.weights, self._size * 10, replacement=True)

			yield from ids
			self.weights = self._get_class_balance_factor()

	def _get_class_balance_factor(self):
		ret = []
		from .coco import CocoDataset
		from .lvis import LVISDataset
		if isinstance(self.dataset, CocoDataset) or isinstance(dataset, LVISDataset):
			#ret = [1.0] * len(self.dataset)
			for idx in range(len(self.dataset)):
				cat_ids = set(self.dataset.get_cat_ids(idx))
				ret.append(sum([self.cw[self.dataset.cat2label[cat_id]] for cat_id in cat_ids if cat_id in self.dataset.cat_ids]))
		else:
			for idx in range(len(self.dataset)):
				cat_ids = set(self.dataset.get_cat_ids(idx))
				ret.append(sum([self.cw[cat_id] for cat_id in cat_ids if cat_id in self.dataset.cat_ids]))
		return torch.tensor(ret).float()

	def __len__(self):
		return len(self.dataset)


def _concat_dataset(cfg, default_args=None):
	from .dataset_wrappers import ConcatDataset
	ann_files = cfg['ann_file']
	img_prefixes = cfg.get('img_prefix', None)
	seg_prefixes = cfg.get('seg_prefix', None)
	proposal_files = cfg.get('proposal_file', None)
	separate_eval = cfg.get('separate_eval', True)

	datasets = []
	num_dset = len(ann_files)
	for i in range(num_dset):
		data_cfg = copy.deepcopy(cfg)
		# pop 'separate_eval' since it is not a valid key for common datasets.
		if 'separate_eval' in data_cfg:
			data_cfg.pop('separate_eval')
		data_cfg['ann_file'] = ann_files[i]
		if isinstance(img_prefixes, (list, tuple)):
			data_cfg['img_prefix'] = img_prefixes[i]
		if isinstance(seg_prefixes, (list, tuple)):
			data_cfg['seg_prefix'] = seg_prefixes[i]
		if isinstance(proposal_files, (list, tuple)):
			data_cfg['proposal_file'] = proposal_files[i]
		datasets.append(build_dataset(data_cfg, default_args))

	return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
	from .dataset_wrappers import (ConcatDataset, RepeatDataset,
								   ClassBalancedDataset, MultiImageMixDataset)
	if isinstance(cfg, (list, tuple)):
		dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
	elif cfg['type'] == 'ConcatDataset':
		dataset = ConcatDataset(
			[build_dataset(c, default_args) for c in cfg['datasets']],
			cfg.get('separate_eval', True))
	elif cfg['type'] == 'RepeatDataset':
		dataset = RepeatDataset(
			build_dataset(cfg['dataset'], default_args), cfg['times'])
	elif cfg['type'] == 'ClassBalancedDataset':
		dataset = ClassBalancedDataset(
			build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
	elif cfg['type'] == 'MultiImageMixDataset':
		cp_cfg = copy.deepcopy(cfg)
		cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
		cp_cfg.pop('type')
		dataset = MultiImageMixDataset(**cp_cfg)
	elif isinstance(cfg.get('ann_file'), (list, tuple)):
		dataset = _concat_dataset(cfg, default_args)
	else:
		dataset = build_from_cfg(cfg, DATASETS, default_args)

	return dataset


def build_dataloader(dataset,
					 samples_per_gpu,
					 workers_per_gpu,
					 num_gpus=1,
					 dist=True,
					 shuffle=True,
					 seed=None,
					 **kwargs):
	"""Build PyTorch DataLoader.

	In distributed training, each GPU/process has a dataloader.
	In non-distributed training, there is only one dataloader for all GPUs.

	Args:
		dataset (Dataset): A PyTorch dataset.
		samples_per_gpu (int): Number of training samples on each GPU, i.e.,
			batch size of each GPU.
		workers_per_gpu (int): How many subprocesses to use for data loading
			for each GPU.
		num_gpus (int): Number of GPUs. Only used in non-distributed training.
		dist (bool): Distributed training/test or not. Default: True.
		shuffle (bool): Whether to shuffle the data at every epoch.
			Default: True.
		kwargs: any keyword argument to be used to initialize DataLoader

	Returns:
		DataLoader: A PyTorch dataloader.
	"""
	rank, world_size = get_dist_info()
	if dist:
		# DistributedGroupSampler will definitely shuffle the data to satisfy
		# that images on each GPU are in the same group
		if shuffle:
			sampler = DistributedGroupSampler(
				dataset, samples_per_gpu, world_size, rank, seed=seed)
		else:
			sampler = DistributedSampler(
				dataset, world_size, rank, shuffle=False, seed=seed)
		batch_size = samples_per_gpu
		num_workers = workers_per_gpu
	else:
		sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
		batch_size = num_gpus * samples_per_gpu
		num_workers = num_gpus * workers_per_gpu

	init_fn = partial(
		worker_init_fn, num_workers=num_workers, rank=rank,
		seed=seed) if seed is not None else None

	sampler = ClassAwareSampler(dataset, samples_per_gpu) if shuffle else None

	data_loader = DataLoader(
		dataset,
		batch_size=batch_size,
		sampler=sampler,
		num_workers=num_workers,
		collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
		pin_memory=False,
		worker_init_fn=init_fn,
		**kwargs)

	return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
	# The seed of each worker equals to
	# num_worker * rank + worker_id + user_seed
	worker_seed = num_workers * rank + worker_id + seed
	np.random.seed(worker_seed)
	random.seed(worker_seed)
