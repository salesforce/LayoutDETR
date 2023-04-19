// Copyright (c) 2023 Salesforce, Inc.
// All rights reserved.
// SPDX-License-Identifier: Apache License 2.0
// For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
// By Ning Yu

// Redistributed from StyleGAN3 repo: https://github.com/NVlabs/stylegan3
// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

#include <cuda_runtime.h>

//------------------------------------------------------------------------
// CUDA kernel parameters.

struct upfirdn2d_kernel_params
{
    const void*     x;
    const float*    f;
    void*           y;

    int2            up;
    int2            down;
    int2            pad0;
    int             flip;
    float           gain;

    int4            inSize;         // [width, height, channel, batch]
    int4            inStride;
    int2            filterSize;     // [width, height]
    int2            filterStride;
    int4            outSize;        // [width, height, channel, batch]
    int4            outStride;
    int             sizeMinor;
    int             sizeMajor;

    int             loopMinor;
    int             loopMajor;
    int             loopX;
    int             launchMinor;
    int             launchMajor;
};

//------------------------------------------------------------------------
// CUDA kernel specialization.

struct upfirdn2d_kernel_spec
{
    void*   kernel;
    int     tileOutW;
    int     tileOutH;
    int     loopMinor;
    int     loopX;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> upfirdn2d_kernel_spec choose_upfirdn2d_kernel(const upfirdn2d_kernel_params& p);

//------------------------------------------------------------------------
