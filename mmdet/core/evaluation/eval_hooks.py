import os.path as osp
import warnings
from math import inf

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.utils import is_seq_of
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from mmdet.utils import get_root_logger
from mmcv.utils import print_log

class EvalHook(BaseEvalHook):
	def __init__(self, train_dataloader, val_dataloader,
				 start=None,
				 interval=1,
				 by_epoch=True,
				 save_best=None,
				 rule=None,
				 test_fn=None,
				 greater_keys=None,
				 less_keys=None,
				 **eval_kwargs):
		if not isinstance(val_dataloader, DataLoader):
			raise TypeError('dataloader must be a pytorch DataLoader, but got'
							f' {type(dataloader)}')
		if not interval > 0:
			raise ValueError(f'interval must be positive, but got {interval}')
		assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'
		if start is not None and start < 0:
			raise ValueError(f'The evaluation start epoch {start} is smaller '
							 f'than 0')
		self.train_dataloader = train_dataloader
		self.dataloader = val_dataloader
		self.interval = interval
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
				self.train_dataloader.sampler.cw[i] = self.train_dataloader.sampler.orig_cw[i] * (1.001 - ap) ** 2
			sum_cw = sum(self.train_dataloader.sampler.cw)
			for i in range(len(self.train_dataloader.dataset.CLASSES)):
				self.train_dataloader.sampler.cw[i] /= sum_cw

			'''txt = "New class weights:\n"
			for i, name in enumerate(self.train_dataloader.dataset.CLASSES):
				txt += "{:40s}: {:.6f}\n".format(name, self.train_dataloader.sampler.cw[i])
			print_log(txt)'''

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
