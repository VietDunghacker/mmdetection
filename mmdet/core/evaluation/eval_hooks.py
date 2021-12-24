# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os.path as osp
from math import inf
from mmcv.utils import print_log

from terminaltables import AsciiTable

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

class EvalHook(BaseEvalHook):
	rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
	init_value_map = {'greater': -inf, 'less': inf}
	_default_greater_keys = [
		'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
		'mAcc', 'aAcc'
	]
	_default_less_keys = ['loss']

	def __init__(self, train_dataloader, val_dataloader,
				 start=None,
				 interval=1,
				 by_epoch=True,
				 save_best=None,
				 rule=None,
				 test_fn=None,
				 greater_keys=None,
				 less_keys=None,
				 out_dir=None,
				 file_client_args=None,
				 **eval_kwargs):
		if not isinstance(val_dataloader, DataLoader):
			raise TypeError('dataloader must be a pytorch DataLoader, but got'
							f' {type(dataloader)}')

		if interval <= 0:
			raise ValueError(f'interval must be a positive number, '
							 f'but got {interval}')

		assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

		if start is not None and start < 0:
			raise ValueError(f'The evaluation start epoch {start} is smaller '
							 f'than 0')

		self.train_dataloader = train_dataloader
		self.dataloader = val_dataloader
		self.interval = interval
		self.start = start
		self.by_epoch = by_epoch

		assert isinstance(save_best, str) or save_best is None, \
			'""save_best"" should be a str or None ' \
			f'rather than {type(save_best)}'
		self.save_best = save_best
		self.eval_kwargs = eval_kwargs
		self.initial_flag = True

		if test_fn is None:
			from mmcv.engine import single_gpu_test
			self.test_fn = single_gpu_test
		else:
			self.test_fn = test_fn

		if greater_keys is None:
			self.greater_keys = self._default_greater_keys
		else:
			if not isinstance(greater_keys, (list, tuple)):
				greater_keys = (greater_keys, )
			assert is_seq_of(greater_keys, str)
			self.greater_keys = greater_keys

		if less_keys is None:
			self.less_keys = self._default_less_keys
		else:
			if not isinstance(less_keys, (list, tuple)):
				less_keys = (less_keys, )
			assert is_seq_of(less_keys, str)
			self.less_keys = less_keys

		if self.save_best is not None:
			self.best_ckpt_path = None
			self._init_rule(rule, self.save_best)

		self.out_dir = out_dir
		self.file_client_args = file_client_args

	def _do_evaluate(self, runner):
		"""perform evaluation and save ckpt."""
		if not self._should_evaluate(runner):
			return

		from mmdet.apis import single_gpu_test
		results = single_gpu_test(runner.model, self.dataloader, show=False)
		runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
		key_score = self.evaluate(runner, results)
		if self.save_best:
			self._save_ckpt(runner, key_score)

	def evaluate(self, runner, results):
		eval_res = self.dataloader.dataset.evaluate(results, logger=runner.logger, **self.eval_kwargs)
		from mmdet.datasets.builder import ClassAwareSampler
		if isinstance(self.train_dataloader.sampler, ClassAwareSampler) and 'AP_per_class' in eval_res.keys():
			for i, ap in enumerate(eval_res['AP_per_class']):
				new_cw = self.train_dataloader.sampler.orig_cw[i] * (1 - ap) ** 2
				self.train_dataloader.sampler.cw[i] = new_cw
			sum_cw = sum(self.train_dataloader.sampler.cw)
			for i in range(len(self.train_dataloader.dataset.CLASSES)):
				self.train_dataloader.sampler.cw[i] /= sum_cw

			num_columns = min(6, len(eval_res['AP_per_class']) * 2)
			results_flatten = list(itertools.chain(*[(name, "{:.6f}".format(item)) for name, item in zip(self.train_dataloader.dataset.CLASSES, self.train_dataloader.sampler.cw)]))
			headers = ['category', 'weight'] * (num_columns // 2)
			results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
			table_data = [headers]
			table_data += [result for result in results_2d]
			table = AsciiTable(table_data)
			#print_log('\n' + table.table, logger=runner.logger)

		if 'AP_per_class' in eval_res.keys():
			del eval_res['AP_per_class']

		for name, val in eval_res.items():
			runner.log_buffer.output[name] = val
		runner.log_buffer.ready = True
		if self.save_best is not None:
			if self.key_indicator == 'auto':
				# infer from eval_results
				self._init_rule(self.rule, list(eval_res.keys())[0])
			return eval_res[self.key_indicator]
		else:
			return None


class DistEvalHook(BaseDistEvalHook):

	def _do_evaluate(self, runner):
		"""perform evaluation and save ckpt."""
		# Synchronization of BatchNorm's buffer (running_mean
		# and running_var) is not supported in the DDP of pytorch,
		# which may cause the inconsistent performance of models in
		# different ranks, so we broadcast BatchNorm's buffers
		# of rank 0 to other ranks to avoid this.
		if self.broadcast_bn_buffer:
			model = runner.model
			for name, module in model.named_modules():
				if isinstance(module,
							  _BatchNorm) and module.track_running_stats:
					dist.broadcast(module.running_var, 0)
					dist.broadcast(module.running_mean, 0)

		if not self._should_evaluate(runner):
			return

		tmpdir = self.tmpdir
		if tmpdir is None:
			tmpdir = osp.join(runner.work_dir, '.eval_hook')

		from mmdet.apis import multi_gpu_test
		results = multi_gpu_test(
			runner.model,
			self.dataloader,
			tmpdir=tmpdir,
			gpu_collect=self.gpu_collect)
		if runner.rank == 0:
			print('\n')
			runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
			key_score = self.evaluate(runner, results)

			if self.save_best:
				self._save_ckpt(runner, key_score)
