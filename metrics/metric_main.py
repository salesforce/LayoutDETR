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
import json
import torch
import dnnlib

from . import metric_utils_layout
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score
from . import equivariance

from . import layout_frechet_inception_distance
from . import overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k
from . import rendering_utils

#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils_layout.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils_layout.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------

def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')

#----------------------------------------------------------------------------
# For LayoutGAN.

@register_metric
def layout_fid50k_train(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    layout_fid = layout_frechet_inception_distance.compute_layout_fid(opts, max_real=None, num_gen=50000)
    return dict(layout_fid50k_train=layout_fid)

@register_metric
def layout_fid50k_val(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    layout_fid = layout_frechet_inception_distance.compute_layout_fid(opts, max_real=None, num_gen=50000)
    return dict(layout_fid50k_val=layout_fid)

@register_metric
def overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    overlap, alignment, layoutwiseIoU, layoutwiseDocSim = overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k.compute_overlap_alignment_laywise_IoU_layerwise_DocSim(opts, max_real=None, num_gen=50000)
    return dict(overlap_50k_train=overlap, alignment_50k_train=alignment, layoutwise_iou50k_train=layoutwiseIoU, layoutwise_docsim50k_train=layoutwiseDocSim)

@register_metric
def overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    overlap, alignment, layoutwiseIoU, layoutwiseDocSim = overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k.compute_overlap_alignment_laywise_IoU_layerwise_DocSim(opts, max_real=None, num_gen=50000)
    return dict(overlap_50k_val=overlap, alignment_50k_val=alignment, layoutwise_iou50k_val=layoutwiseIoU, layoutwise_docsim50k_val=layoutwiseDocSim)

@register_metric
def fid50k_train(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_train=fid)

@register_metric
def fid50k_val(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_val=fid)

@register_metric
def rendering_train(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    rendering_utils.render_train(opts, max_items=None)
    return dict(rendering_train=1)

@register_metric
def rendering_val(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    rendering_utils.render_val(opts, max_items=None)
    return dict(rendering_val=1)

#----------------------------------------------------------------------------
# Recommended metrics.

@register_metric
def fid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)

@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k_full=kid)

@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(opts, max_real=200000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)

@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(opts, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=2)
    return dict(ppl2_wend=ppl)

@register_metric
def eqt50k_int(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqt_int=True)
    return dict(eqt50k_int=psnr)

@register_metric
def eqt50k_frac(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqt_frac=True)
    return dict(eqt50k_frac=psnr)

@register_metric
def eqr50k(opts):
    opts.G_kwargs.update(force_fp32=True)
    psnr = equivariance.compute_equivariance_metrics(opts, num_samples=50000, batch_size=4, compute_eqr=True)
    return dict(eqr50k=psnr)

#----------------------------------------------------------------------------
# Legacy metrics.

@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)

@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000)
    return dict(kid50k=kid)

@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(opts, max_real=50000, num_gen=50000, nhood_size=3, row_batch_size=10000, col_batch_size=10000)
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)

@register_metric
def is50k(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10)
    return dict(is50k_mean=mean, is50k_std=std)

#----------------------------------------------------------------------------
