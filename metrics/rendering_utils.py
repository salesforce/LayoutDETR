import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib
from PIL import Image

from util import save_real_image_with_background

import json
import copy
from gen_single_sample import horizontal_center_aligned, de_overlap
from gen_single_sample_server import visualize_banner
from selenium import webdriver
from selenium.webdriver import Chrome
import random
import uuid
import pdb
import skimage

# emtpy text color or button background color means they are adaptive to the underlying background
json_temp = {
    "modelId": "BANNERS",
    "task": "banner",
    "numResults": 5,
    "resultFormat": ["image"],
    "contentStyle": {
        "elements": [
           {
                "type": "header",
                "text": "",
                "style": {
                    "fontFamily": "",
                    "color": "",
                    "fontFormat": "bold"
                }
            },
            {
                "type": "body",
                "text": "",
                "style": {
                    "fontFamily": "",
                    "color": ""
                }
            },
            {
                "type": "button",
                "text": "",
                "buttonParams": {
                    "backgroundColor": "",
                    "backgroundImage": "",
                    "radius": ""
                },
                "style": {
                    "fontFamily": "",
                    "color": ""
                }
            },
            {
                "type": "disclaimer / footnote",
                "text": "",
                "style": {
                    "fontFamily": "",
                    "color": ""
                }
            },
        ],
    }
}

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

def render_train(opts, batch_size=1, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # initialize Chrome based web driver for html screenshot
    options = webdriver.ChromeOptions()
    options.add_argument('no-sandbox')
    options.add_argument('headless')
    browser = Chrome(executable_path='/usr/bin/chromedriver', options=options)
    # make sure browser window size has enough resolution for the largest background image
    browser.set_window_size(4096, 4096)

    temp = json.dumps(json_temp)
    banner_specs = json.loads(temp)
    # label id to element specs dictionary
    element_specs = {0: copy.deepcopy(banner_specs['contentStyle']['elements'][0]),
                     3: copy.deepcopy(banner_specs['contentStyle']['elements'][1]),
                     5: copy.deepcopy(banner_specs['contentStyle']['elements'][2]),
                     4: copy.deepcopy(banner_specs['contentStyle']['elements'][3])}

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_size)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)

    # Main loop.
    os.makedirs(os.path.join(opts.run_dir, 'train_rendering_fake'), exist_ok=True)
    os.makedirs(os.path.join(opts.run_dir, 'train_rendering_real'), exist_ok=True)
    count = 0
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for samples, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        name = list(samples['name'])
        W_page = list(samples['W_page'].cpu().numpy())
        H_page = list(samples['H_page'].cpu().numpy())
        bbox_real = samples['bboxes'].to(opts.device).to(torch.float32)
        bbox_class = samples['labels'].to(opts.device).to(torch.int64)
        bbox_text = list(map(list, zip(*(samples['texts'])))) # have to transpose the list of lists of texts
        bbox_patch = samples['patches'].to(opts.device).to(torch.float32)
        bbox_patch_orig = samples['patches_orig'].to(opts.device).to(torch.float32)
        mask = samples['mask'].to(opts.device).to(torch.bool)
        padding_mask = ~mask
        background = samples['background'].to(opts.device).to(torch.float32)
        background_orig = samples['background_orig'].to(opts.device).to(torch.float32)
        gen_z = torch.randn([bbox_class.shape[0], bbox_class.shape[1], G.z_dim], dtype=torch.float32, device=opts.device)
        bbox_fake = G(z=gen_z, bbox_class=bbox_class, bbox_real=bbox_real, bbox_text=bbox_text, bbox_patch=bbox_patch, padding_mask=padding_mask, background=background, c=next(c_iter), **opts.G_kwargs)
        bbox_fake = horizontal_center_aligned(bbox_fake, mask)
        bbox_fake = de_overlap(bbox_fake, mask)
        B = bbox_real.size(0)
        for i in range(B):
            mask_i = mask[i].detach().cpu().numpy()
            if mask_i.sum() > 4:
                continue
            bbox_class_i = bbox_class[i].detach().cpu().numpy()
            flag = False
            for idx in [0, 3, 4, 5]:
                if (bbox_class_i[mask_i]==idx).sum() > 1:
                    flag = True
                    break
            for idx in [1, 2, 6, 7]:
                if (bbox_class_i[mask_i]==idx).sum() > 0:
                    flag = True
                    break
            if flag:
                continue
            box_fake_i = bbox_fake[i].detach().cpu().numpy()
            box_real_i = bbox_real[i].detach().cpu().numpy()
            bbox_text_i = bbox_text[i]
            background_orig_i = background_orig[i].detach().cpu().numpy()
            rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
            rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))
            background_orig_i = np.transpose(background_orig_i, [1,2,0])
            background_orig_i = np.clip(background_orig_i * rgb_std + rgb_mean, 0.0, 1.0)
            background_orig_i = skimage.transform.resize(background_orig_i, (H_page[i], W_page[i]), anti_aliasing=True)
            background_orig_i = Image.fromarray((background_orig_i*255.0).astype(np.uint8))

            banner_specs['numResults'] = 1
            banner_specs['contentStyle']['elements'] = []
            for j, m in enumerate(mask_i):
                if m:
                    banner_specs['contentStyle']['elements'].append(element_specs[bbox_class_i[j]])
            this_font = 'Arial'
            this_radius = 0.5
            for j, e in enumerate(banner_specs['contentStyle']['elements']):
                e['text'] = bbox_text_i[j]
                e['style']['fontFamily'] = this_font
                if e['type'] == 'button':
                    e['buttonParams']['radius'] = this_radius
            visualize_banner(box_fake_i, mask_i, banner_specs['contentStyle']['elements'],
                             True, background_orig_i, browser, banner_specs["resultFormat"],
                             os.path.join(os.getcwd(), opts.run_dir, 'train_rendering_fake', name[i].replace('.json', '')))
            visualize_banner(box_real_i, mask_i, banner_specs['contentStyle']['elements'],
                             True, background_orig_i, browser, banner_specs["resultFormat"],
                             os.path.join(os.getcwd(), opts.run_dir, 'train_rendering_real', name[i].replace('.json', '')))
            print('Saving %06d samples: ' % count, name[i].replace('.json', '_vis.png'))
            count += 1

#----------------------------------------------------------------------------

def render_val(opts, batch_size=1, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # initialize Chrome based web driver for html screenshot
    options = webdriver.ChromeOptions()
    options.add_argument('no-sandbox')
    options.add_argument('headless')
    browser = Chrome(executable_path='/usr/bin/chromedriver', options=options)
    # make sure browser window size has enough resolution for the largest background image
    browser.set_window_size(4096, 4096)

    temp = json.dumps(json_temp)
    banner_specs = json.loads(temp)
    # label id to element specs dictionary
    element_specs = {0: copy.deepcopy(banner_specs['contentStyle']['elements'][0]),
                     3: copy.deepcopy(banner_specs['contentStyle']['elements'][1]),
                     5: copy.deepcopy(banner_specs['contentStyle']['elements'][2]),
                     4: copy.deepcopy(banner_specs['contentStyle']['elements'][3])}

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_size)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)

    # Main loop.
    os.makedirs(os.path.join(opts.run_dir, 'val_rendering_fake'), exist_ok=True)
    os.makedirs(os.path.join(opts.run_dir, 'val_rendering_real'), exist_ok=True)
    count = 0
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for samples, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        name = list(samples['name'])
        W_page = list(samples['W_page'].cpu().numpy())
        H_page = list(samples['H_page'].cpu().numpy())
        bbox_real = samples['bboxes'].to(opts.device).to(torch.float32)
        bbox_class = samples['labels'].to(opts.device).to(torch.int64)
        bbox_text = list(map(list, zip(*(samples['texts'])))) # have to transpose the list of lists of texts
        bbox_patch = samples['patches'].to(opts.device).to(torch.float32)
        bbox_patch_orig = samples['patches_orig'].to(opts.device).to(torch.float32)
        mask = samples['mask'].to(opts.device).to(torch.bool)
        padding_mask = ~mask
        background = samples['background'].to(opts.device).to(torch.float32)
        background_orig = samples['background_orig'].to(opts.device).to(torch.float32)
        gen_z = torch.randn([bbox_class.shape[0], bbox_class.shape[1], G.z_dim], dtype=torch.float32, device=opts.device)
        bbox_fake = G(z=gen_z, bbox_class=bbox_class, bbox_real=bbox_real, bbox_text=bbox_text, bbox_patch=bbox_patch, padding_mask=padding_mask, background=background, c=next(c_iter), **opts.G_kwargs)
        bbox_fake = horizontal_center_aligned(bbox_fake, mask)
        bbox_fake = de_overlap(bbox_fake, mask)
        B = bbox_real.size(0)
        for i in range(B):
            mask_i = mask[i].detach().cpu().numpy()
            if mask_i.sum() > 4:
                continue
            bbox_class_i = bbox_class[i].detach().cpu().numpy()
            flag = False
            for idx in [0, 3, 4, 5]:
                if (bbox_class_i[mask_i]==idx).sum() > 1:
                    flag = True
                    break
            for idx in [1, 2, 6, 7]:
                if (bbox_class_i[mask_i]==idx).sum() > 0:
                    flag = True
                    break
            if flag:
                continue
            box_fake_i = bbox_fake[i].detach().cpu().numpy()
            box_real_i = bbox_real[i].detach().cpu().numpy()
            bbox_text_i = bbox_text[i]
            background_orig_i = background_orig[i].detach().cpu().numpy()
            rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
            rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))
            background_orig_i = np.transpose(background_orig_i, [1,2,0])
            background_orig_i = np.clip(background_orig_i * rgb_std + rgb_mean, 0.0, 1.0)
            background_orig_i = skimage.transform.resize(background_orig_i, (H_page[i], W_page[i]), anti_aliasing=True)
            background_orig_i = Image.fromarray((background_orig_i*255.0).astype(np.uint8))

            banner_specs['numResults'] = 1
            banner_specs['contentStyle']['elements'] = []
            for j, m in enumerate(mask_i):
                if m:
                    banner_specs['contentStyle']['elements'].append(element_specs[bbox_class_i[j]])
            this_font = 'Arial'
            this_radius = 0.5
            for j, e in enumerate(banner_specs['contentStyle']['elements']):
                e['text'] = bbox_text_i[j]
                e['style']['fontFamily'] = this_font
                if e['type'] == 'button':
                    e['buttonParams']['radius'] = this_radius
            visualize_banner(box_fake_i, mask_i, banner_specs['contentStyle']['elements'],
                             True, background_orig_i, browser, banner_specs["resultFormat"],
                             os.path.join(os.getcwd(), opts.run_dir, 'val_rendering_fake', name[i].replace('.json', '')))
            visualize_banner(box_real_i, mask_i, banner_specs['contentStyle']['elements'],
                             True, background_orig_i, browser, banner_specs["resultFormat"],
                             os.path.join(os.getcwd(), opts.run_dir, 'val_rendering_real', name[i].replace('.json', '')))
            print('Saving %06d samples: ' % count, name[i].replace('.json', '_vis.png'))
            count += 1