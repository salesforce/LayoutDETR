'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Ning Yu

 * Modified from StyleGAN3 repo: https://github.com/NVlabs/stylegan3
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

 * Reimplemented from "GANs trained by a two time-scale update rule converge to a local Nash equilibrium": https://github.com/bioinf-jku/TTUR/blob/master/fid.py
'''

import numpy as np
import scipy.linalg
from . import metric_utils_layout

#----------------------------------------------------------------------------

def compute_layout_fid(opts, max_real, num_gen):
    dataset_name = opts.dataset_kwargs['path'].split('/')[-3]
    detector_pth = 'pretrained/layoutnet_%s.pth.tar' % dataset_name
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils_layout.compute_feature_stats_for_dataset(
        opts=opts, detector_pth=detector_pth, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils_layout.compute_feature_stats_for_generator(
        opts=opts, detector_pth=detector_pth, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
