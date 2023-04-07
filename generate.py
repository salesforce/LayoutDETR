import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFilter
import seaborn as sns
import torch

import legacy

import json
import copy
from metrics.metric_layoutnet import compute_overlap, compute_alignment
from util import convert_xywh_to_ltrb
from generate_util import visualize_banner
from selenium import webdriver
from selenium.webdriver import Chrome
import random
import uuid
import pdb


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.
    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def save_bboxes_with_background(boxes, masks, labels, background_orig, path):
    colors = sns.color_palette('husl', n_colors=13)
    colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
    background_orig_temp = background_orig.copy()
    W_page, H_page = background_orig_temp.size
    draw = ImageDraw.Draw(background_orig_temp, 'RGBA')
    boxes = boxes[masks]
    labels = labels[masks]
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)), key=lambda i: area[i], reverse=True)
    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * W_page, x2 * W_page
        y1, y2 = y1 * H_page, y2 * H_page
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
    background_orig_temp.save(path, format='png', compress_level=0, optimize=False)

#----------------------------------------------------------------------------

def jitter(bbox_fake, out_jittering_strength, seed): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    perturb = torch.from_numpy(np.random.RandomState(seed).uniform(low=math.log(1.0-out_jittering_strength), high=math.log(1.0+out_jittering_strength), size=bbox_fake.shape)).to(bbox_fake.device).to(torch.float32)
    bbox_fake *= perturb.exp()
    return bbox_fake

#----------------------------------------------------------------------------

def horizontal_center_aligned(bbox_fake, mask): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    xc_mean = bbox_fake[mask][:,0].mean()
    bbox_fake[:,:,0] = xc_mean
    return bbox_fake

def horizontal_left_aligned(bbox_fake, mask): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    num = bbox_fake[mask].shape[0]
    x1_sum = 0.0
    for i in range(num):
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox_fake[0,i])
        x1_sum += x1
    x1_mean = x1_sum / float(num)
    for i in range(num):
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox_fake[0,i])
        bbox_fake[0,i,0] -= x1 - x1_mean
    return bbox_fake

def de_overlap(bbox_fake, mask): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    num = bbox_fake[mask].shape[0]
    for i in range(num):
        xc1, yc1, w1, h1 = bbox_fake[0,i]
        for j in range(num):
            if i != j:
                xc2, yc2, w2, h2 = bbox_fake[0,j]
                if abs(yc2 - yc1) < h1/2 + h2/2:
                    diff = h1/2 + h2/2 - abs(yc2 - yc1)
                    if yc1 < yc2:
                        bbox_fake[0,i,1] -= diff/2
                        bbox_fake[0,j,1] += diff/2
                    else:
                        bbox_fake[0,i,1] += diff/2
                        bbox_fake[0,j,1] -= diff/2
    for i in range(num):
        xc1, yc1, w1, h1 = bbox_fake[0,i]
        for j in range(num):
            if i != j:
                xc2, yc2, w2, h2 = bbox_fake[0,j]
                if abs(yc2 - yc1) < h1/2 + h2/2:
                    diff = h1/2 + h2/2 - abs(yc2 - yc1)
                    bbox_fake[0,i,3] -= diff/2
                    bbox_fake[0,j,3] -= diff/2
    return bbox_fake

#----------------------------------------------------------------------------

label_list = [
        'header',
        'pre-header',
        'post-header',
        'body text',
        'disclaimer / footnote',
        'button',
        'callout',
        'logo'
        ]
label2index = dict()
for idx, label in enumerate(label_list):
    label2index[label] = idx

#----------------------------------------------------------------------------
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

font_type = ['Helvetica', 'Verdana', 'Times New Roman', 'Georgia', 'Aria', 'Arial']

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--bg', type=str, help='Path of a background image', required=True)
@click.option('--bg-preprocessing', 'bg_preprocessing', help='Postprocess the background image', type=click.Choice(['256', '128', 'blur', 'jpeg', 'rec', '3x_mask', 'edge', 'none']), default='none', show_default=True)
@click.option('--strings', type=str, help="Strings to be printed on the banner. Multiple strings are separated by '|'", required=True)
@click.option('--string-labels', 'string_labels', type=str, help="String labels. Multiple labels are separated by '|'", required=True)
@click.option('--outfile', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--out-jittering-strength', 'out_jittering_strength', help='Randomly jitter the output bounding box parameters with a certain strength', type=click.FloatRange(min=0.0, max=1.0), default=0.0, show_default=True)
@click.option('--out-postprocessing', 'out_postprocessing', help='Postprocess the output layout', type=click.Choice(['horizontal_center_aligned', 'horizontal_left_aligned', 'none']), default='none', show_default=True)
def generate_images(
    network_pkl: str,
    bg: str,
    bg_preprocessing: str,
    strings: str,
    string_labels: str,
    outfile: str,
    out_jittering_strength: float,
    out_postprocessing: str,
):
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

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    print('Loading background image from "%s"...' % bg)
    background_orig = Image.open(bg).convert('RGB')
    W_page, H_page = background_orig.size
    if W_page > H_page and W_page > 4096:
        W_page_new = 4096
        H_page_new = int(float(H_page) / float(W_page) * float(W_page_new))
        background_orig = background_orig.resize((W_page_new, H_page_new), Image.ANTIALIAS)
    elif H_page > W_page and H_page > 4096:
        H_page_new = 4096
        W_page_new = int(float(W_page) / float(H_page) * float(H_page_new))
        background_orig = background_orig.resize((W_page_new, H_page_new), Image.ANTIALIAS)

    if bg_preprocessing == '256':
        background = np.array(background_orig.resize((256, 256), Image.ANTIALIAS))
    elif bg_preprocessing == '128':
        background = np.array(background_orig.resize((128, 128), Image.ANTIALIAS))
    elif bg_preprocessing == 'blur':
        background = background_orig.filter(ImageFilter.GaussianBlur(radius=3))
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    elif bg_preprocessing == 'jpeg':
        idx = bg.rfind('/')
        bg_new = bg[:idx] + '_jpeg' + bg[idx:].replace('.png', '.jpg')
        background = Image.open(bg_new).convert('RGB')
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    elif bg_preprocessing == 'rec':
        idx = bg.rfind('/')
        bg_new = bg[:idx] + '_rec' + bg[idx:]
        background = Image.open(bg_new).convert('RGB')
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    elif bg_preprocessing == 'edge':
        background = background_orig.convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    else:
        background = np.array(background_orig.resize((1024, 1024), Image.ANTIALIAS))
        
    if background.ndim == 2:
        background = np.dstack((background, background, background))
    background = background[:,:,:3]
    rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
    rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))
    background = (background.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
    background = background.transpose(2, 0, 1)
    background = torch.from_numpy(background).to(device).to(torch.float32).unsqueeze(0)
    bbox_text = strings.split('|')
    bbox_label = string_labels.split('|')
    bbox_label = [label2index[label] for label in bbox_label]

    print('Loading layout bboxes')
    mask = torch.from_numpy(np.array([1] * len(bbox_text) + [0] * (9-len(bbox_text)))).to(device).to(torch.bool).unsqueeze(0)
    bbox_patch_dummy = torch.zeros((1, 9, 3, 256, 256)).to(device).to(torch.float32)

    z = torch.from_numpy(np.random.RandomState(0).randn(1, 9, G.z_dim)).to(device).to(torch.float32)
    bbox_text_temp = list(bbox_text)
    bbox_text_temp += [''] * (9-len(bbox_text))
    bbox_text_temp = [bbox_text_temp]
    bbox_label_temp = list(bbox_label)
    bbox_class_temp = torch.from_numpy(np.array(bbox_label_temp + [0] * (9-len(bbox_label_temp)))).to(device).to(torch.int64).unsqueeze(0)
    bbox_fake = G(z=z, bbox_class=bbox_class_temp, bbox_real=None, bbox_text=bbox_text_temp, bbox_patch=bbox_patch_dummy, padding_mask=~mask, background=background, c=None)
    if out_jittering_strength > 0.0:
        bbox_fake = jitter(bbox_fake, out_jittering_strength, seed=0)

    # if out_postprocessing not decided, i.e. random mode, randomly choose among
    # ['horizontal_center_aligned', 'horizontal_left_aligned', 'none']
    if out_postprocessing == 'none':
        rand_val = random.random()
        if 0 <= rand_val < 0.34:
            out_postprocessing == 'horizontal_center_aligned'
        elif 0.34 <= rand_val < 0.67:
            out_postprocessing == 'horizontal_left_aligned'
    if out_postprocessing == 'horizontal_center_aligned':
        bbox_fake = horizontal_center_aligned(bbox_fake, mask)
        bbox_fake = de_overlap(bbox_fake, mask)
        bbox_alignment = True
    elif out_postprocessing == 'horizontal_left_aligned':
        bbox_fake = horizontal_left_aligned(bbox_fake, mask)
        bbox_fake = de_overlap(bbox_fake, mask)
        bbox_alignment = False
    else:
        bbox_alignment = True # still center align text strings

    outfile = os.path.join(os.getcwd(), outfile)
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)

    banner_specs['numResults'] = 1
    ###################################
    # config json format banner specs
    ###################################
    bbox_fake = bbox_fake.detach().cpu().numpy().squeeze()
    mask = mask.detach().cpu().numpy().squeeze()
    bbox_class = bbox_class_temp.detach().cpu().numpy().squeeze()
    text = bbox_text_temp[0]
    banner_specs['contentStyle']['elements'] = []
    for i, m in enumerate(mask):
        if m:
            banner_specs['contentStyle']['elements'].append(element_specs[bbox_class[i]])
    for i, e in enumerate(banner_specs['contentStyle']['elements']):
        e['text'] = text[i]
        e['style']['fontFamily'] = 'Arial'
        if e['type'] == 'button':
            e['buttonParams']['radius'] = 0.5

    visualize_banner(bbox_fake, mask, banner_specs['contentStyle']['elements'],
                    bbox_alignment, background_orig, browser, banner_specs["resultFormat"],
                    outfile)
    save_bboxes_with_background(bbox_fake, mask, bbox_class, background_orig, outfile+'_bboxes.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------