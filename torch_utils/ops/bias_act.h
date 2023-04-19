// Copyright (c) 2023 Salesforce, Inc.
// All rights reserved.
// SPDX-License-Identifier: Apache License 2.0
// For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
// By Ning Yu

// Redistributed from StyleGAN3 repo: https://github.com/NVlabs/stylegan3
// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

struct bias_act_kernel_params
{
    const void* x;      // [sizeX]
    const void* b;      // [sizeB] or NULL
    const void* xref;   // [sizeX] or NULL
    const void* yref;   // [sizeX] or NULL
    const void* dy;     // [sizeX] or NULL
    void*       y;      // [sizeX]

    int         grad;
    int         act;
    float       alpha;
    float       gain;
    float       clamp;

    int         sizeX;
    int         sizeB;
    int         stepB;
    int         loopX;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p);

//------------------------------------------------------------------------
