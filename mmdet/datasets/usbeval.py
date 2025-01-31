import datetime
import numpy as np
import time
from .api_wrappers import COCOeval


def calc_area_range_info(area_range_type):
	"""Calculate area ranges and related information."""
	# use COCO setting as default
	area_ranges = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2],
				   [96**2, 1e5**2]]
	area_labels = ['all', 'small', 'medium', 'large']
	relative_area = False

	if area_range_type == 'COCO':
		pass
	elif area_range_type == 'relative_scale_ap':
		relative_area = True
		area_ranges = [[0**2, 1**2]]
		area_labels = ['all']
		inv_scale_thrs = np.power(2, np.arange(0, 5))[::-1]
		for inv_min, inv_max in zip(inv_scale_thrs[:-1], inv_scale_thrs[1:]):
			if inv_max == 16:
				area_ranges.append([0**2, 1 / inv_max**2])
				area_labels.append(f'0_1/{inv_max}')
			else:
				area_ranges.append([1 / inv_min**2, 1 / inv_max**2])
				area_labels.append(f'1/{inv_min}_1/{inv_max}')
	elif area_range_type == 'absolute_scale_ap':
		scale_thrs = np.power(2, np.arange(6, 11))
		scale_thrs[0] = 0
		scale_thrs[-1] = 1e5
		for min_scale, max_scale in zip(scale_thrs[:-1], scale_thrs[1:]):
			area_ranges.append([min_scale**2, max_scale**2])
			area_labels.append(f'{min_scale:.0f}_{max_scale:.0f}')
	elif area_range_type == 'absolute_scale_ap_linear':
		scale_thrs = np.arange(0, 1024 + 32 + 1, 32)
		scale_thrs[-1] = 1e5
		for min_scale, max_scale in zip(scale_thrs[:-1], scale_thrs[1:]):
			area_ranges.append([min_scale**2, max_scale**2])
			area_labels.append(f'{min_scale:.0f}_{max_scale:.0f}')
	elif area_range_type == 'TinyPerson':
		area_ranges = [[1**2, 1e5**2], [1**2, 20**2], [1**2, 8**2],
					   [8**2, 12**2], [12**2, 20**2], [20**2, 32**2],
					   [32**2, 1e5**2]]
		area_labels = [
			'all', 'tiny', 'tiny1', 'tiny2', 'tiny3', 'small', 'reasonable'
		]
	else:
		raise NotImplementedError

	assert len(area_ranges) == len(area_labels)
	area_range_map = dict(zip(area_labels, area_ranges))

	return area_ranges, area_labels, relative_area


class USBeval(COCOeval):

	def __init__(self,
				 cocoGt=None,
				 cocoDt=None,
				 iouType='segm',
				 area_range_type='COCO'):
		"""Initialize CocoEval using coco APIs for gt and dt.
		:param cocoGt: coco object with ground truth annotations
		:param cocoDt: coco object with detection results
		:return: None
		"""
		super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)
		area_ranges, area_labels, relative_area = calc_area_range_info(area_range_type)
		self.params.areaRng = area_ranges
		self.params.areaRngLbl = area_labels
		self.relative_area = relative_area

	def evaluateImg(self, imgId, catId, aRng, maxDet):
		'''
		perform evaluation for single category and image
		:return: dict (single image results)
		'''
		img_info = self.cocoGt.loadImgs([imgId])[0]
		img_area = img_info['width'] * img_info['height']

		p = self.params
		if p.useCats:
			gt = self._gts[imgId, catId]
			dt = self._dts[imgId, catId]
		else:
			gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
			dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
		if len(gt) == 0 and len(dt) == 0:
			return None

		for g in gt:
			if self.relative_area:
				area = g['area'] / img_area
			else:
				area = g['area']
			if g['ignore'] or (area < aRng[0] or area > aRng[1]):
				g['_ignore'] = 1
			else:
				g['_ignore'] = 0

		# sort dt highest score first, sort gt ignore last
		gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
		gt = [gt[i] for i in gtind]
		dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
		dt = [dt[i] for i in dtind[0:maxDet]]
		iscrowd = [int(o['iscrowd']) for o in gt]
		# load computed ious
		ious = self.ious[imgId, catId][:, gtind] if len(
			self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

		T = len(p.iouThrs)
		G = len(gt)
		D = len(dt)
		gtm = np.zeros((T, G))
		dtm = np.zeros((T, D))
		gtIg = np.array([g['_ignore'] for g in gt])
		dtIg = np.zeros((T, D))
		dtIoU = np.zeros((T, D))
		if not len(ious) == 0:
			for tind, t in enumerate(p.iouThrs):
				for dind, d in enumerate(dt):
					# information about best match so far (m=-1 -> unmatched)
					iou = min([t, 1 - 1e-10])
					m = -1
					for gind, g in enumerate(gt):
						# if this gt already matched, and not a crowd, continue
						if gtm[tind, gind] > 0 and not iscrowd[gind]:
							continue
						# if dt matched to reg gt, and on ignore gt, stop
						if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
							break
						# continue to next gt unless better match made
						if ious[dind, gind] < iou:
							continue
						# if match successful and best so far, store
						iou = ious[dind, gind]
						m = gind
					# if match made store id of match for both dt and gt
					if m == -1:
						continue
					dtIg[tind, dind] = gtIg[m]
					dtm[tind, dind] = gt[m]['id']
					gtm[tind, m] = d['id']
					dtIoU[tind, dind] = iou
		# set unmatched detections outside of area range to ignore
		if self.relative_area:
			a = np.array([
				d['area'] / img_area < aRng[0]
				or d['area'] / img_area > aRng[1] for d in dt
			])
		else:
			a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt])
		a = a.reshape((1, len(dt)))
		dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
		# store results for given image and category
		return {
			'image_id': imgId,
			'category_id': catId,
			'aRng': aRng,
			'maxDet': maxDet,
			'dtIds': [d['id'] for d in dt],
			'gtIds': [g['id'] for g in gt],
			'dtMatches': dtm,
			'gtMatches': gtm,
			'dtScores': [d['score'] for d in dt],
			'gtIgnore': gtIg,
			'dtIgnore': dtIg,
			'dtIoUs': dtIoU,
		}

	def accumulate(self, p=None):
		'''
		Accumulate per image evaluation results and
		store the result in self.eval
		:param p: input params for evaluation
		:return: None
		'''
		print('Accumulating evaluation results...')
		tic = time.time()
		if not self.evalImgs:
			print('Please run evaluate() first')
		# allows input customized parameters
		if p is None:
			p = self.params
		p.catIds = p.catIds if p.useCats == 1 else [-1]
		T = len(p.iouThrs)
		R = len(p.recThrs)
		K = len(p.catIds) if p.useCats else 1
		A = len(p.areaRng)
		M = len(p.maxDets)
		# -1 for the precision of absent categories
		precision = -np.ones((T, R, K, A, M))
		recall = -np.ones((T, K, A, M))
		scores = -np.ones((T, R, K, A, M))
		olrp_loc = -np.ones((K, A, M))
		olrp_fp = -np.ones((K, A, M))
		olrp_fn = -np.ones((K, A, M))
		olrp = -np.ones((K, A, M))
		lrp_opt_thr = -np.ones((K, A, M))

		# create dictionary for future indexing
		_pe = self._paramsEval
		catIds = _pe.catIds if _pe.useCats else [-1]
		setK = set(catIds)
		setA = set(map(tuple, _pe.areaRng))
		setM = set(_pe.maxDets)
		setI = set(_pe.imgIds)
		# get inds to evaluate
		k_list = [n for n, k in enumerate(p.catIds) if k in setK]
		m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
		a_list = [
			n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
			if a in setA
		]
		i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
		I0 = len(_pe.imgIds)
		A0 = len(_pe.areaRng)
		# retrieve E at each category, area range, and max number of detections
		for k, k0 in enumerate(k_list):
			Nk = k0 * A0 * I0
			for a, a0 in enumerate(a_list):
				Na = a0 * I0
				for m, maxDet in enumerate(m_list):
					E = [self.evalImgs[Nk + Na + i] for i in i_list]
					E = [e for e in E if e is not None]
					if len(E) == 0:
						continue
					dtScores = np.concatenate(
						[e['dtScores'][0:maxDet] for e in E])

					# different sorting method generates slightly
					# different results.
					# mergesort is used to be consistent as Matlab
					# implementation.
					inds = np.argsort(-dtScores, kind='mergesort')
					dtScoresSorted = dtScores[inds]

					dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
					dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
					dtIoU = np.concatenate([e['dtIoUs'][:, 0:maxDet] for e in E], axis=1)[:, inds]

					gtIg = np.concatenate([e['gtIgnore'] for e in E])
					npig = np.count_nonzero(gtIg == 0)
					if npig == 0:
						continue
					tps = np.logical_and(dtm, np.logical_not(dtIg))
					fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

					dtIoU = np.multiply(dtIoU, tps)
					tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
					fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
					for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
						tp = np.array(tp)
						fp = np.array(fp)
						nd = len(tp)
						rc = tp / npig
						pr = tp / (fp + tp + np.spacing(1))
						q = np.zeros((R, ))
						ss = np.zeros((R, ))

						if nd:
							recall[t, k, a, m] = rc[-1]
						else:
							recall[t, k, a, m] = 0

						# numpy is slow without cython optimization
						# for accessing elements use python array
						# gets significant speed improvement
						pr = pr.tolist()
						q = q.tolist()

						for i in range(nd - 1, 0, -1):
							if pr[i] > pr[i - 1]:
								pr[i - 1] = pr[i]

						inds = np.searchsorted(rc, p.recThrs, side='left')
						try:
							for ri, pi in enumerate(inds):
								q[ri] = pr[pi]
								ss[ri] = dtScoresSorted[pi]
						except BaseException:
							pass
						precision[t, :, k, a, m] = np.array(q)
						scores[t, :, k, a, m] = np.array(ss)

					# oLRP and Opt.Thr. Computation
					tp_num = np.cumsum(tps[0, :])
					fp_num = np.cumsum(fps[0, :])
					fn_num = npig - tp_num
					# If there is detection
					if tp_num.shape[0] > 0:
						# There is some TPs
						if tp_num[-1] > 0:
							total_loc = tp_num - np.cumsum(dtIoU[0, :])
							lrps = (total_loc / (1 - _pe.iouThrs[0]) + fp_num +
									fn_num) / (tp_num + fp_num + fn_num)
							opt_pos_idx = np.argmin(lrps)
							olrp[k, a, m] = lrps[opt_pos_idx]
							olrp_loc[k, a, m] = total_loc[opt_pos_idx] / \
								tp_num[opt_pos_idx]
							olrp_fp[k, a, m] = fp_num[opt_pos_idx] / \
								(tp_num[opt_pos_idx] + fp_num[opt_pos_idx])
							olrp_fn[k, a, m] = fn_num[opt_pos_idx] / npig
							lrp_opt_thr[k, a, m] = dtScoresSorted[opt_pos_idx]
						# There is No TP
						else:
							olrp_loc[k, a, m] = np.nan
							olrp_fp[k, a, m] = np.nan
							olrp_fn[k, a, m] = 1.
							olrp[k, a, m] = 1.
							lrp_opt_thr[k, a, m] = np.nan
					# No detection
					else:
						olrp_loc[k, a, m] = np.nan
						olrp_fp[k, a, m] = np.nan
						olrp_fn[k, a, m] = 1.
						olrp[k, a, m] = 1.
						lrp_opt_thr[k, a, m] = np.nan
		self.eval = {
			'params': p,
			'counts': [T, R, K, A, M],
			'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
			'precision': precision,
			'recall':   recall,
			'scores': scores,
			'olrp_loc': olrp_loc,
			'olrp_fp': olrp_fp,
			'olrp_fn': olrp_fn,
			'olrp': olrp,
			'lrp_opt_thr': lrp_opt_thr,
		}
		toc = time.time()
		print('DONE (t={:0.2f}s).'.format(toc - tic))


	def _summarize(self, ap=1, iouThr=None, areaRng='all', maxDets=100, lrp_type=None):
		"""Compute and display a specific metric."""
		p = self.params
		iStr = (' {:<18} {} @[IoU={:<9} | area={:>11s} | maxDets={:>4d}] = {:0.4f}')
		titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
		typeStr = '(AP)' if ap == 1 else '(AR)'
		iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)

		aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
		mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
		if ap == 1:
			# dimension of precision: [TxRxKxAxM]
			s = self.eval['precision']
			# IoU
			if iouThr is not None:
				if iouThr == 0.9:
					t = [8]
				else:
					t = np.where(iouThr == p.iouThrs)[0]
				s = s[t]
			s = s[:, :, :, aind, mind]
			if len(s[s > -1]) == 0:
				mean_s = -1
			else:
				mean_s = np.mean(s[s > -1])
		elif ap == 0:
			# dimension of recall: [TxKxAxM]
			s = self.eval['recall']
			if iouThr is not None:
				t = np.where(iouThr == p.iouThrs)[0]
				s = s[t]
			s = s[:, :, aind, mind]
		else:
			# # dimension of LRP: [KxAxM]
			# Person 0, Broccoli 50
			if lrp_type == 'oLRP':
				s = self.eval['olrp'][:, aind, mind]
				titleStr = 'Optimal LRP'
				typeStr = '	'
			if lrp_type == 'oLRP_Localisation':
				s = self.eval['olrp_loc'][:, aind, mind]
				titleStr = 'Optimal LRP Loc'
				typeStr = '	'
			if lrp_type == 'oLRP_false_positive':
				s = self.eval['olrp_fp'][:, aind, mind]
				titleStr = 'Optimal LRP FP'
				typeStr = '	'
			if lrp_type == 'oLRP_false_negative':
				s = self.eval['olrp_fn'][:, aind, mind]
				titleStr = 'Optimal LRP FN'
				typeStr = '	'
			if lrp_type == 'oLRP_thresholds':
				s = self.eval['lrp_opt_thr'][:, aind, mind].squeeze(axis=1)
				titleStr = '# Class-specific LRP-Optimal Thresholds # \n'
				typeStr = '	'
				# Floor by using 3 decimal digits
				print(titleStr, np.round(s - 0.5 * 10**(-3), 3))
				return s
		idx = (~np.isnan(s))
		s = s[idx]
		if len(s[s > -1]) == 0:
			mean_s = -1
		else:
			mean_s = np.mean(s[s > -1])
		print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
		return mean_s

	@staticmethod
	def _shorten_area_label(area_label):
		"""Shorten area label like mmdet."""
		area_label_short = area_label
		shortening_map = {'small': 's', 'medium': 'm', 'large': 'l'}
		for long, short in shortening_map.items():
			area_label_short = area_label_short.replace(long, short)
		return area_label_short

	def _summarizeDets(self):
		"""Compute and display summary metrics for detection."""
		max_dets = self.params.maxDets
		stats = {}
		# summarize AP
		for area_label in self.params.areaRngLbl:
			area_label_short = self._shorten_area_label(area_label)
			if area_label == 'all':
				stats['mAP'] = self._summarize(1)
				for i in range(50, 100, 5):
					stats['mAP_{}'.format(i)] = self._summarize(1, iouThr=i/100, maxDets=max_dets[2])
			else:
				stats[f'mAP_{area_label_short}'] = self._summarize(1, areaRng=area_label, maxDets=max_dets[2])
		# summarize AR
		for area_label in self.params.areaRngLbl:
			area_label_short = self._shorten_area_label(area_label)
			if area_label == 'all':
				for max_det in max_dets:
					stats[f'AR@{max_det}'] = self._summarize(0, maxDets=max_det)
			elif area_label in ['small', 'medium', 'large']:
				key = f'AR_{area_label_short}@{max_dets[2]}'
				stats[key] = self._summarize(0, areaRng=area_label, maxDets=max_dets[2])

		stats['oLRP'] = self._summarize(-1, iouThr=.5, areaRng='all', maxDets=max_dets[2], lrp_type='oLRP')
		stats['oLRP_loc'] = self._summarize(-1, iouThr=.5, areaRng='all', maxDets=max_dets[2], lrp_type='oLRP_Localisation')
		stats['oLRP_fp'] = self._summarize(-1, iouThr=.5, areaRng='all', maxDets=max_dets[2], lrp_type='oLRP_false_positive')
		stats['oLRP_fn'] = self._summarize(-1, iouThr=.5, areaRng='all', maxDets=max_dets[2], lrp_type='oLRP_false_negative')
		stats['oLRP_small'] = self._summarize(-1, iouThr=.5, areaRng='small', maxDets=max_dets[2], lrp_type='oLRP')
		stats['oLRP_medium'] = self._summarize(-1, iouThr=.5, areaRng='medium', maxDets=max_dets[2], lrp_type='oLRP')
		stats['oLRP_large'] = self._summarize(-1, iouThr=.5, areaRng='large', maxDets=max_dets[2],  lrp_type='oLRP')

		return stats

	def summarize(self):
		"""Compute and display summary metrics for evaluation results."""
		if not self.eval:
			raise Exception('Please run accumulate() first')
		iouType = self.params.iouType
		if iouType == 'segm' or iouType == 'bbox':
			summarize = self._summarizeDets
		elif iouType == 'keypoints':
			raise NotImplementedError
		self.stats = summarize()