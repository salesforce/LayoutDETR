'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Ning Yu

 * Modified from StyleGAN3 repo: https://github.com/NVlabs/stylegan3
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
'''

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib

from metrics.metric_layoutnet import LayoutFID, compute_overlap, compute_alignment

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True, run_dir=None):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.run_dir        = run_dir

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(pth):
    return os.path.splitext(pth.split('/')[-1])[0]

def get_feature_detector(pth, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (pth, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        _feature_detector_cache[key] = LayoutFID(pth, device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        #assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) #and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone() #BC
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # BGC->BC # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_pth, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=8, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_pth=detector_pth, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_pth)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector_obj = get_feature_detector(pth=detector_pth, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for samples, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        bbox_real = samples['bboxes'].to(opts.device).to(torch.float32)
        bbox_class = samples['labels'].to(opts.device).to(torch.int64)
        mask = samples['mask'].to(opts.device).to(torch.bool)
        padding_mask = ~mask
        label_idx_replace = 'ads_banner_collection' in detector_pth or 'AMT_uploaded_ads_banners' in detector_pth
        label_idx_replace_2 = 'cgl_dataset' in detector_pth
        features = detector_obj.model.extract_features(bbox_real, bbox_class, padding_mask, label_idx_replace=label_idx_replace, label_idx_replace_2=label_idx_replace_2)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_pth, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=8, data_loader_kwargs=None, max_items=None, batch_gen=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_size)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='generator features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector_obj = get_feature_detector(pth=detector_pth, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for samples, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        bbox_real = samples['bboxes'].to(opts.device).to(torch.float32)
        bbox_class = samples['labels'].to(opts.device).to(torch.int64)
        bbox_text = list(map(list, zip(*(samples['texts'])))) # have to transpose the list of lists of texts
        bbox_patch = samples['patches'].to(opts.device).to(torch.float32)
        mask = samples['mask'].to(opts.device).to(torch.bool)
        padding_mask = ~mask
        background = samples['background'].to(opts.device).to(torch.float32)
        gen_z = torch.randn([bbox_class.shape[0], bbox_class.shape[1], G.z_dim], dtype=torch.float32, device=opts.device)
        bbox_fake = G(z=gen_z, bbox_class=bbox_class, bbox_real=bbox_real, bbox_text=bbox_text, bbox_patch=bbox_patch, padding_mask=padding_mask, background=background, c=next(c_iter), **opts.G_kwargs)
        label_idx_replace = 'ads_banner_collection' in detector_pth or 'AMT_uploaded_ads_banners' in detector_pth
        label_idx_replace_2 = 'cgl_dataset' in detector_pth
        features = detector_obj.model.extract_features(bbox_fake.detach(), bbox_class, padding_mask, label_idx_replace=label_idx_replace, label_idx_replace_2=label_idx_replace_2)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------

def compute_maxIoU_overlap_alignment_wrapper(opts, rel_lo=0, rel_hi=1, batch_size=8, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_size)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats_bbox_real = FeatureStats(max_items=num_items, capture_all=True, **stats_kwargs)
    stats_bbox_fake = FeatureStats(max_items=num_items, capture_all=True, **stats_kwargs)
    stats_bbox_class = FeatureStats(max_items=num_items, capture_all=True, **stats_kwargs)
    stats_mask = FeatureStats(max_items=num_items, capture_all=True, **stats_kwargs)
    stats_overlap = FeatureStats(max_items=num_items, capture_all=True, **stats_kwargs)
    stats_alignment = FeatureStats(max_items=num_items, capture_all=True, **stats_kwargs)
    progress = opts.progress.sub(tag='calculate maximum IoU', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for samples, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        bbox_real = samples['bboxes'].to(opts.device).to(torch.float32)
        bbox_class = samples['labels'].to(opts.device).to(torch.int64)
        bbox_text = list(map(list, zip(*(samples['texts'])))) # have to transpose the list of lists of texts
        bbox_patch = samples['patches'].to(opts.device).to(torch.float32)
        mask = samples['mask'].to(opts.device).to(torch.bool)
        padding_mask = ~mask
        background = samples['background'].to(opts.device).to(torch.float32)
        gen_z = torch.randn([bbox_class.shape[0], bbox_class.shape[1], G.z_dim], dtype=torch.float32, device=opts.device)
        bbox_fake = G(z=gen_z, bbox_class=bbox_class, bbox_real=bbox_real, bbox_text=bbox_text, bbox_patch=bbox_patch, padding_mask=padding_mask, background=background, c=next(c_iter), **opts.G_kwargs)
        stats_bbox_real.append_torch(bbox_real, num_gpus=opts.num_gpus, rank=opts.rank)
        stats_bbox_fake.append_torch(bbox_fake, num_gpus=opts.num_gpus, rank=opts.rank)
        stats_bbox_class.append_torch(bbox_class, num_gpus=opts.num_gpus, rank=opts.rank)
        stats_mask.append_torch(mask, num_gpus=opts.num_gpus, rank=opts.rank)

        overlap = compute_overlap(bbox_fake, mask).unsqueeze(-1)
        stats_overlap.append_torch(overlap, num_gpus=opts.num_gpus, rank=opts.rank)

        alignment = compute_alignment(bbox_fake, mask).unsqueeze(-1)
        stats_alignment.append_torch(alignment, num_gpus=opts.num_gpus, rank=opts.rank)

        progress.update(stats_bbox_real.num_items)

    return stats_bbox_real, stats_bbox_fake, stats_bbox_class, stats_mask, stats_overlap, stats_alignment

#----------------------------------------------------------------------------