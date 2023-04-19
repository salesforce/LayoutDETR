'''
 * Copyright (c) 2023 Salesforce, Inc.
 * All rights reserved.
 * SPDX-License-Identifier: Apache License 2.0
 * For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
 * By Ning Yu

 * Modified from StyleGAN3 repo: https://github.com/NVlabs/stylegan3
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
'''

import json
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

import torch
import torchvision.utils as vutils
import torchvision.transforms as T

import skimage.transform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed:", seed)


def init_experiment(args, prefix):
    if args.seed is None:
        args.seed = random.randint(0, 10000)

    set_seed(args.seed)

    if not args.name:
        args.name = datetime.now().strftime('%Y%m%d%H%M%S%f')

    out_dir = Path('output') / args.dataset / prefix / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'args.json'
    with json_path.open('w') as f:
        json.dump(vars(args), f, indent=2)

    return out_dir


def save_checkpoint(state, is_best, out_dir):
    out_path = Path(out_dir) / 'checkpoint.pth.tar'
    torch.save(state, out_path)

    if is_best:
        best_path = Path(out_dir) / 'model_best.pth.tar'
        shutil.copyfile(out_path, best_path)


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def expand2square(pil_img):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new('RGB', (width, width), color=(0, 0, 0))
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new('RGB', (height, height), color=(0, 0, 0))
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def convert_layout_to_image(boxes, labels, colors, W_page, H_page, size_canvas):
    img = Image.new('RGB', (W_page, H_page), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * W_page, x2 * W_page
        y1, y2 = y1 * H_page, y2 * H_page
        draw.rectangle([x1, y1, x2, y2],
                       outline=color,
                       fill=c_fill)

    if W_page > H_page:
        W_page_new = size_canvas
        H_page_new = int(float(H_page) / float(W_page) * float(size_canvas)) // 2 * 2
    else:
        H_page_new = size_canvas
        W_page_new = int(float(W_page) / float(H_page) * float(size_canvas)) // 2 * 2
    img = img.resize((W_page_new, H_page_new), resample=Image.BILINEAR)
    return expand2square(img)


def save_image(batch_boxes, batch_labels, batch_mask,
               dataset_colors, out_path, W_page, H_page, size_canvas=128,
               nrow=None,
               return_instead_of_save=False):
    # batch_boxes: [B, N, 4]
    # batch_labels: [B, N]
    # batch_mask: [B, N]

    imgs = []
    B = batch_boxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes = batch_boxes[i][mask_i]
        labels = batch_labels[i][mask_i]
        img = convert_layout_to_image(boxes, labels,
                                      dataset_colors,
                                      W_page[i], H_page[i], size_canvas)
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if return_instead_of_save:
        return image
        
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))
    vutils.save_image(image, out_path, normalize=False, nrow=nrow)


def convert_layout_to_real_image(boxes_fake, boxes_real, images, W_page, H_page, size_canvas):
    rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
    rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))

    img = np.ones((H_page, W_page, 3)).astype('float')

    # draw from larger boxes_fake
    area = [b[2] * b[3] for b in boxes_fake]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox_fake, bbox_real, image = boxes_fake[i], boxes_real[i], images[i]

        width, height = int(bbox_real[2] * W_page), int(bbox_real[3] * H_page)
        image = np.transpose(image.cpu().numpy(), [1,2,0])
        cy = image.shape[0] // 2
        cx = image.shape[1] // 2
        im = image[cy-height//2:cy+height-height//2, cx-width//2:cx+width-width//2]
        im = np.clip(im * rgb_std + rgb_mean, 0.0, 1.0)


        x1_fake, y1_fake, x2_fake, y2_fake = convert_xywh_to_ltrb(bbox_fake)
        x1_fake, x2_fake = int(round(x1_fake.cpu().numpy() * float(W_page))), int(round(x2_fake.cpu().numpy() * float(W_page)))
        y1_fake, y2_fake = int(round(y1_fake.cpu().numpy() * float(H_page))), int(round(y2_fake.cpu().numpy() * float(H_page)))
        
        im = skimage.transform.resize(im, (max(y2_fake-y1_fake, 1), max(x2_fake-x1_fake, 1)), anti_aliasing=True)
        if y1_fake < 0:
            im = im[-y1_fake:]
            y_start = 0
        else:
            y_start = y1_fake
        if y2_fake > H_page:
            im = im[:H_page-y2_fake]
            y_end = H_page
        else:
            y_end = y2_fake
        if x1_fake < 0:
            im = im[:, -x1_fake:]
            x_start = 0
        else:
            x_start = x1_fake
        if x2_fake > W_page:
            im = im[:, :W_page-x2_fake]
            x_end = W_page
        else:
            x_end = x2_fake
        img[y_start:y_end, x_start:x_end] = im

    img = (img * 255.0).astype('ubyte')
    img = Image.fromarray(img, 'RGB')

    if W_page > H_page:
        W_page_new = size_canvas
        H_page_new = int(float(H_page) / float(W_page) * float(size_canvas)) // 2 * 2
    else:
        H_page_new = size_canvas
        W_page_new = int(float(W_page) / float(H_page) * float(size_canvas)) // 2 * 2
    img = img.resize((W_page_new, H_page_new), resample=Image.BILINEAR)
    return expand2square(img)


def save_real_image(batch_boxes_fake, batch_boxes_real, batch_images, batch_mask,
               out_path, W_page, H_page, size_canvas=1024,
               nrow=None):
    # batch_boxes_fake: [B, N, 4]
    # batch_boxes_real: [B, N, 4]
    # batch_images: [B, N, 3, H_page, W_page]
    # batch_mask: [B, N]

    imgs = []
    B = batch_boxes_fake.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes_fake = batch_boxes_fake[i][mask_i]
        boxes_real = batch_boxes_real[i][mask_i]
        images = batch_images[i][mask_i]
        img = convert_layout_to_real_image(boxes_fake, boxes_real, images,
                                      W_page[i], H_page[i], size_canvas)
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    vutils.save_image(image, out_path, normalize=False, nrow=nrow)


def convert_layout_to_real_image_with_background(boxes_fake, boxes_real, images, bg, W_page, H_page, size_canvas):
    rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
    rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))

    img = np.transpose(bg.cpu().numpy(), [1,2,0])
    img = np.clip(img * rgb_std + rgb_mean, 0.0, 1.0)
    img = skimage.transform.resize(img, (H_page, W_page), anti_aliasing=True)

    # draw from larger boxes_fake
    area = [b[2] * b[3] for b in boxes_fake]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox_fake, bbox_real, image = boxes_fake[i], boxes_real[i], images[i]

        width, height = int(bbox_real[2] * W_page), int(bbox_real[3] * H_page)
        image = np.transpose(image.cpu().numpy(), [1,2,0])
        cy = image.shape[0] // 2
        cx = image.shape[1] // 2
        im = image[cy-height//2:cy+height-height//2, cx-width//2:cx+width-width//2]
        im = np.clip(im * rgb_std + rgb_mean, 0.0, 1.0)

        x1_fake, y1_fake, x2_fake, y2_fake = convert_xywh_to_ltrb(bbox_fake)
        x1_fake, x2_fake = int(round(x1_fake.cpu().numpy() * float(W_page))), int(round(x2_fake.cpu().numpy() * float(W_page)))
        y1_fake, y2_fake = int(round(y1_fake.cpu().numpy() * float(H_page))), int(round(y2_fake.cpu().numpy() * float(H_page)))
        
        im = skimage.transform.resize(im, (max(y2_fake-y1_fake, 1), max(x2_fake-x1_fake, 1)), anti_aliasing=True)
        if y1_fake < 0:
            im = im[-y1_fake:]
            y_start = 0
        else:
            y_start = y1_fake
        if y2_fake > H_page:
            im = im[:H_page-y2_fake]
            y_end = H_page
        else:
            y_end = y2_fake
        if x1_fake < 0:
            im = im[:, -x1_fake:]
            x_start = 0
        else:
            x_start = x1_fake
        if x2_fake > W_page:
            im = im[:, :W_page-x2_fake]
            x_end = W_page
        else:
            x_end = x2_fake
        img[y_start:y_end, x_start:x_end] = im

    img = (img * 255.0).astype('ubyte')
    img = Image.fromarray(img, 'RGB')

    if W_page > H_page:
        W_page_new = size_canvas
        H_page_new = int(float(H_page) / float(W_page) * float(size_canvas)) // 2 * 2
    else:
        H_page_new = size_canvas
        W_page_new = int(float(W_page) / float(H_page) * float(size_canvas)) // 2 * 2
    img = img.resize((W_page_new, H_page_new), resample=Image.BILINEAR)
    return expand2square(img)


def save_real_image_with_background(batch_boxes_fake, batch_boxes_real, batch_images, batch_mask, background,
               out_path, W_page, H_page, size_canvas=1024,
               nrow=None,
               return_instead_of_save=False):
    # batch_boxes_fake: [B, N, 4]
    # batch_images: [B, N, 3, H_page, W_page]
    # batch_mask: [B, N]
    # background: [B, 3, H_page, W_page]

    imgs = []
    B = batch_boxes_fake.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes_fake = batch_boxes_fake[i][mask_i]
        boxes_real = batch_boxes_real[i][mask_i]
        images = batch_images[i][mask_i]
        bg = background[i]
        img = convert_layout_to_real_image_with_background(boxes_fake, boxes_real, images, bg, W_page[i], H_page[i], size_canvas)
        imgs.append(to_tensor(img))
    images = torch.stack(imgs)

    if return_instead_of_save:
        return images
        
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))
    vutils.save_image(images, out_path, normalize=False, nrow=nrow)
