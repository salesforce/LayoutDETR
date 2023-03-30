import os
import re
from typing import List, Optional, Tuple, Union, Dict
import uuid

import click
import dnnlib
import numpy as np
from PIL import Image, ImageDraw
import seaborn as sns
import torch
import legacy

from metrics.metric_layoutnet import compute_overlap, compute_alignment
from util import convert_xywh_to_ltrb
from e2e_pipeline.utils_server import safeMakeDirs
from bs4 import BeautifulSoup
from io import BytesIO
import re
import sys
import argparse
import json
import math
from selenium import webdriver
from selenium.webdriver import Chrome

HTML_TEMP = \
    """
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    .container {
      position: relative;
      color: white;
    }
    .body {
      margin: 0;
      padding: 0;
    }
    </style>
    </head>
    <body class="body">
    <div class="container">
      <img src="" alt="" style="width:auto;position:absolute;top:0px;left:0px;">
    </div>
    </body>
    </html> 
    """

TEXT_CSS_TEMP = 'align-items:center;position:absolute;word-wrap:break-word;overflow-wrap:' \
                'break-word;display:flex;'
#TEXT_CSS_TEMP += 'border-width:1px;border-style:solid;border-color:black;'

LABEL_LIST = [
        'header',
        'pre-header',
        'post-header',
        'body',
        'disclaimer / footnote',
        'button',
        'callout',
        'logo'
        ]

#----------------------------------------------------------------------------

def get_adaptive_font_size1(w_tbox, h_tbox, H_page, text, font2height=0.038422, font_aspect_ratio=0.52,
                            min_font_size=9):
    print('w:{}, h:{}, H_page:{}, text:{}\n'.format(w_tbox, h_tbox, H_page, text))
    font_size = int(H_page*font2height)
    num_word = len(text)
    return str(max(min_font_size, int((w_tbox * h_tbox / num_word / font_aspect_ratio) ** .5)))

#----------------------------------------------------------------------------

def get_adaptive_font_size2(w_tbox, h_tbox, H_page, text, text_type, font_aspect_ratio=0.52,
                            min_font_size=9):
    font2h = {'header': 0.076844, 'body': 0.04322475, 'button': 0.04082337, 'disclaimer / footnote': 0.032}

    font_size = int(H_page*font2h[text_type])
    num_word = len(text)
    num_line = num_word * font_size * font_aspect_ratio / w_tbox
    if num_line < 1 or num_line * font_size < h_tbox:
        return str(font_size), int(num_word * font_size * font_aspect_ratio * 1.25)
    else:  # num_word * font_size * font_aspect_ratio * font_size < w_tbox * h_tbox
        return str(max(min_font_size, int((w_tbox * h_tbox / num_word / font_aspect_ratio) ** .5))), int(num_word * font_size * font_aspect_ratio * 1.25)

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

def jitter(bbox_fake, seed): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    perturb = torch.from_numpy(np.random.RandomState(seed).uniform(low=math.log(0.8), high=math.log(1.2), size=bbox_fake.shape)).to(bbox_fake.device).to(torch.float32)
    bbox_fake *= perturb.exp()
    return bbox_fake

#----------------------------------------------------------------------------

def get_adaptive_font_color(img):
    img = np.array(img)
    clr = []
    for ch in range(3):
        clr.append(np.median(img[:, :, ch]))

    return 'rgba'+str((0, 0, 0, 255)) if sum(clr) > 255 * 3 / 1.5 else 'rgba:'+str((255, 255, 255, 255))

#----------------------------------------------------------------------------

def get_adaptive_font_button_color(img):
    img = np.array(img)
    clr = []
    for ch in range(3):
        clr.append(np.median(img[:, :, ch]))

    # adaptive font color, background color
    if sum(clr) < 255 * 2:
        return 'rgba'+str((0, 0, 0, 255)), 'rgba'+str((255, 255, 255, 255))
    else:
        return 'rgba'+str((255, 255, 255, 255)), 'rgba'+str((0, 0, 0, 255))

#----------------------------------------------------------------------------

def get_adaptive_font_size(w_tbox, h_tbox, H_page, text, font2height=0.038422, font_aspect_ratio=0.52,
                           min_font_size=9):
    font_size = int(H_page*font2height)
    num_word = len(text)
    num_line = num_word * font_size * font_aspect_ratio / w_tbox
    if num_line < 1 or num_line * font_size < h_tbox:
        return str(font_size)
    else:  # num_word * font_size * font_aspect_ratio * font_size < w_tbox * h_tbox
        return str(max(min_font_size, int((w_tbox * h_tbox / num_word / font_aspect_ratio) ** .5)))

# ----------------------------------------------------------------------------
def visualize_banner(boxes, masks, styles, is_center, background_img, browser, output_format, generated_file_path):
    soup = BeautifulSoup(HTML_TEMP, "html.parser")
    # insert img src div
    img = soup.findAll('img')
    img[0]['src'] = os.path.basename(generated_file_path + '.png')
    background_img.save(generated_file_path + '.png')

    W_page, H_page = background_img.size
    w_page, h_page = W_page, H_page  # thumbnail resolution
    boxes = boxes[masks]
    for i in range(boxes.shape[0]):
        text = styles[i]['text']
        if not text:
            continue
        
        x1, y1, x2, y2 = convert_xywh_to_ltrb(boxes[i])
        x1, x2 = max(0, int(x1 * W_page)), min(W_page - 1, int(x2 * W_page))
        y1, y2 = max(0, int(y1 * H_page)), min(H_page - 1, int(y2 * H_page))
        h_tbox, w_tbox = int(y2 - y1 + 1), int(x2 - x1 + 1)
        font_color = styles[i]['style']['color']
        font_family = styles[i]['style']['fontFamily']
        font_family = 'font-family:' + font_family + ';' if 'fontFamily' in styles[i]['style'] and \
                                                            styles[i]['style']['fontFamily'] else 'font-family:Arial;'
        if font_color:
            font_color = 'color:' + font_color + ';'
        else:
            if styles[i]['type'] == 'button':
                font_color = 'color:' + get_adaptive_font_button_color(background_img.crop([x1, y1, x2, y2]))[0] + ';'
            else:
                font_color = 'color:' + get_adaptive_font_color(background_img.crop([x1, y1, x2, y2])) + ';'

        font_size, text_width = get_adaptive_font_size2(w_tbox, h_tbox, H_page, text, styles[i]['type'])

        # button resize and alignment
        if styles[i]['type'] == 'button':
            r_mar = 1.3
            font_size_int = int(font_size)
            mar = font_size_int/2*r_mar
            y_mid = (y1 + y2)/2
            if is_center:
                x_mid = (x1 + x2)/2
                y1 = max(0, y_mid - mar - 1)
                y2 = min(H_page - 1, y_mid + mar)
                x1 = max(0, x_mid - text_width/2 - mar - 1)
                x2 = min(W_page - 1, x_mid + text_width/2 + mar)
            else:
                y1 = max(0, y_mid - mar - 1)
                y2 = min(H_page - 1, y_mid + mar)
                x2 = min(W_page - 1, x1 + text_width + mar * 2)
            h_tbox, w_tbox = int(y2 - y1 + 1), int(x2 - x1 + 1)

        font_size = 'font-size:' + font_size + 'px;'
        tbox_id = 'id="' + styles[i]['type'] + '";'

        if styles[i]['type'] == 'button' or is_center:
            tbox_style = TEXT_CSS_TEMP + 'text-align:center;justify-content:center;'
        else:
            tbox_style = TEXT_CSS_TEMP + 'text-align:left;'

        tbox_style = tbox_style + font_color + font_size + font_family + tbox_id
        tbox_style += 'width:' + str(w_tbox) + 'px;max-width:' + str(w_tbox) + 'px;'
        tbox_style += 'height:' + str(h_tbox) + 'px;max-height:' + str(h_tbox) + 'px;'
        tbox_style += 'top:' + str(y1) + 'px;'
        tbox_style += 'left:' + str(x1) + 'px;'
        if styles[i]['type'].lower() == 'button':
            if styles[i]['buttonParams']['backgroundColor']:
                tbox_style += 'background-color:' + styles[i]['buttonParams']['backgroundColor'] + ';'
            else:
                tbox_style += 'background-color:' + get_adaptive_font_button_color(background_img.crop([x1, y1, x2, y2]))[1] + ';'

            if styles[i]['buttonParams']['radius']:
                tbox_style += 'border-radius:' + str(styles[i]['buttonParams']['radius']).strip() + 'em;'

        tbox_attr = {'style': tbox_style}
        new_div = soup.new_tag("div", **tbox_attr)
        new_div.string = text
        soup.html.body.div.append(new_div)

    soup.prettify()
    generated_image_path_vis = generated_html_path = ''
    if 'image' in output_format:
        generated_image_path_vis = generated_file_path + '_vis.png'
        with open(generated_file_path + '.html', "w") as f:
            f.write(str(soup))
        try:
            browser.get("file:///" + generated_file_path + '.html')
        except Exception as e:
            pass
        png = browser.get_screenshot_as_png()
        screenshot = Image.open(BytesIO(png))
        screenshot = screenshot.crop([0, 0, W_page, H_page])
        if W_page > w_page or H_page > h_page:
            screenshot.thumbnail((w_page, h_page), Image.ANTIALIAS)
        screenshot.save(generated_image_path_vis)

    if 'html' in output_format:
        generated_html_path = generated_file_path + '.html'
        # avoid saving html twice
        if 'image' not in output_format:
            with open(generated_html_path, "w") as f:
                f.write(str(soup))

    return generated_image_path_vis, generated_html_path

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

def load_model(model_dir):
    model_path = os.path.join(model_dir, 'ads_multi.pkl')
    print('Loading networks from "%s"...' % model_path)
    device = torch.device('cuda')
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    return G

def generate_banners(
    G: str,
    bg: str,
    input_styles: List[Dict[str, str]],
    post_process: Dict[str, float],
    seeds: List[int],
    output_format: List[str],
    browser: Chrome,
    output_dir: str,
):
    device = 'cuda'
    print('Loading background image from "%s"...' % bg)
    background_orig = Image.open(bg)
    background = np.array(background_orig.resize((1024, 1024), Image.ANTIALIAS))
    # background = np.array(background_orig.resize((256, 256), Image.ANTIALIAS))
    rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
    rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))
    background = (background.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
    background = background.transpose(2, 0, 1)
    background = torch.from_numpy(background).to(device).to(torch.float32).unsqueeze(0)
    label2index = dict()
    for idx, label in enumerate(LABEL_LIST):
        label2index[label] = idx

    # construct the input text strings and the corresponding styles
    bbox_text = []
    bbox_label = []
    bbox_style = []
    sorted_input_styles = []
    note_style = None
    for style in input_styles:
        try:
            if style['type'] == 'disclaimer / footnote':
                note_style = style
            else:
                sorted_input_styles.append(style)
        except KeyError:
            continue

    if note_style:
        sorted_input_styles.append(note_style)

    for style in sorted_input_styles:
        try:
            if style['type'] == 'header' or style['type'] == 'body' or style['type'] == 'button' or \
                    style['type'] == 'disclaimer / footnote':
                bbox_text.append(style["text"])
                bbox_label.append(label2index[style["type"]])
                bbox_style.append(style)
        except KeyError:
            continue

    print('Loading layout bboxes')
    print(bbox_label)
    bbox_fake_list = []
    mask_list = []
    bbox_style_list = []
    overlap = []
    alignment = []
    is_center_list = []
    mask = torch.from_numpy(np.array([1] * len(bbox_text) + [0] * (9-len(bbox_text)))).to(device).to(torch.bool).unsqueeze(0)
    bbox_patch_dummy = torch.zeros((1, 9, 3, 256, 256)).to(device).to(torch.float32)
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, 9, G.z_dim)).to(device).to(torch.float32)
        order = list(range(len(bbox_text)))
        bbox_text_temp = [bbox_text[i] for i in order]
        bbox_text_temp += [''] * (9-len(bbox_text))
        bbox_text_temp = [bbox_text_temp]
        bbox_label_temp = [bbox_label[i] for i in order]
        bbox_class_temp = torch.from_numpy(np.array(bbox_label_temp + [0] * (9-len(bbox_label_temp)))).to(device).to(torch.int64).unsqueeze(0)
        bbox_fake = G(z=z, bbox_class=bbox_class_temp, bbox_real=None, bbox_text=bbox_text_temp, bbox_patch=bbox_patch_dummy, padding_mask=~mask, background=background, c=None)
        if seed != 1 and 'jitter' in post_process and np.random.rand() < post_process['jitter']:
            bbox_fake = jitter(bbox_fake, seed)
        if 'horizontal_center_aligned' in post_process and np.random.rand() < post_process['horizontal_center_aligned']:
            bbox_fake = horizontal_center_aligned(bbox_fake, mask)
            bbox_fake = de_overlap(bbox_fake, mask)
            is_center_list.append(True)
        else:
            bbox_fake = horizontal_left_aligned(bbox_fake, mask)
            bbox_fake = de_overlap(bbox_fake, mask)
            is_center_list.append(False)
        bbox_fake_list.append(bbox_fake[0])
        mask_list.append(mask[0])

        # record the original bbox_style order
        bbox_style_temp = [bbox_style[i] for i in order]
        bbox_style_temp += [''] * (9-len(bbox_style))
        bbox_style_list.append(bbox_style_temp)

        overlap.append(compute_overlap(bbox_fake, mask).cpu().numpy()[0])
        alignment.append(compute_alignment(bbox_fake, mask).cpu().numpy()[0])
    
    ###################################
    # Save random sample variants according to overlap
    ###################################
    subdir = '%s' % output_dir
    os.makedirs(subdir, exist_ok=True)
    order = np.argsort(np.array(overlap))
    generated_image_paths = []
    generated_html_paths = []
    for j, idx in enumerate(order):
        generated_path = os.path.join(output_dir, f'{str(uuid.uuid4())}')

        generated_image_path, generated_html_path = \
            visualize_banner(bbox_fake_list[idx], mask_list[idx], bbox_style_list[idx], is_center_list[idx], background_orig,
                             browser, output_format, generated_path)

        generated_image_paths.append(generated_image_path)
        generated_html_paths.append(generated_html_path)

    return generated_image_paths, generated_html_paths