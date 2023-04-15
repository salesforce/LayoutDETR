import functools
import gzip
import io
import json
import os
import pickle
import re
import sys
import tarfile
import zipfile
import csv
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
from tqdm import tqdm

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

#----------------------------------------------------------------------------

def parse_tuple(s: str) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

#----------------------------------------------------------------------------

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

#----------------------------------------------------------------------------

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION # type: ignore

#----------------------------------------------------------------------------

def append_child(element, elements):
    if 'children' in element.keys():
        for child in element['children']:
            elements.append(child)
            elements = append_child(child, elements)
    return elements

def convert_xywh_to_ltrb(bboxes):
    xc = bboxes[0]
    yc = bboxes[1]
    w = bboxes[2]
    h = bboxes[3]
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2

def lexicographic_sort_idx(bboxes):
    bboxes_temp = np.transpose(np.array(bboxes))
    l, t, _, _ = convert_xywh_to_ltrb(bboxes_temp)
    _zip = zip(*sorted(enumerate(zip(t, l)), key=lambda c: c[1:]))
    idx = list(list(_zip)[0])
    return idx

#----------------------------------------------------------------------------

def open_ads_banner_collection_manual_gt(source_dir, max_samples: Optional[int]):
    input_samples = sorted(Path(source_dir).glob('*.json'))

    #######################################
    # Load page labels
    #######################################

    page_labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            page_labels = json.load(file)['labels']
            if page_labels is not None:
                page_labels = { x[0]: x[1] for x in page_labels }
            else:
                page_labels = {}

    #######################################
    # Load bboxes, their labels, and their image patches
    #######################################

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

    max_idx = maybe_min(len(input_samples), max_samples)

    def iterate_samples():
        for idx, fname in enumerate(input_samples):
            page = PIL.Image.open(str(fname).replace('.json', '.png'))
            W_page = page.size[0]
            H_page = page.size[1]

            # Load page label
            arch_fname = os.path.relpath(str(fname), source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            page_label = page_labels.get(arch_fname)
            
            with fname.open() as f:
                ann = json.load(f)

            def is_valid(element):
                if 'label' not in element or element['label'] not in label_list:
                    return False
                if 'str' not in element or len(element['str']) == 0 or len(element['str']) >= 256:
                    return False
                x1, y1, x2, y2 = element['xyxy_word_fit']
                if x1 < 0 or y1 < 0 or W_page < x2 or H_page < y2:
                    return False
                if x2 <= x1 or y2 <= y1:
                    return False
                width = int(x2) - int(x1)
                height = int(y2) - int(y1)
                if width > 1024 or height > 1024:
                    return False
                if width > height:
                    height_new = int(float(height) / float(width) * 256.0) // 2 * 2
                    if height_new == 0:
                        return False
                else:
                    width_new = int(float(width) / float(height) * 256.0) // 2 * 2
                    if width_new == 0:
                        return False
                return True

            _elements = list(filter(is_valid, ann))
            # my filter to keep only elements that are not overlapped by other elements: there is no bbox in the current bbox
            valid_list = []
            for i, e in enumerate(_elements):
                x1, y1, x2, y2 = e['xyxy_word_fit']
                valid = True
                for j, ee in enumerate(_elements):
                    if i != j:
                        xx1, yy1, xx2, yy2 = ee['xyxy_word_fit']
                        x1_max = max([x1, xx1])
                        y1_max = max([y1, yy1])
                        x2_min = min([x2, xx2])
                        y2_min = min([y2, yy2])
                        if x1_max < x2_min and y1_max < y2_min and float((x2_min - x1_max) * (y2_min - y1_max)) / float((x2 - x1) * (y2 - y1)) >= 0.95:
                            valid = False
                            break
                valid_list.append(valid)
            _elements = [e for i, e in enumerate(_elements) if valid_list[i]]
            filtered = len(ann) != len(_elements)
            elements = _elements
            N = len(elements)
            if N == 0 or 9 < N:
                continue

            bboxes = []
            labels = []
            texts = []
            patches = []
            patches_orig = []
            patch_masks = []

            page = np.array(page)
            if page.ndim == 2:
                page = np.stack((page, page, page), axis=2)
            elif page.shape[2] == 4:
                page = page[:,:,:3]
            for element in elements:
                # bbox
                x1, y1, x2, y2 = element['xyxy_word_fit']
                xc = (x1 + x2) / 2.
                yc = (y1 + y2) / 2.
                width = x2 - x1
                height = y2 - y1
                b = [xc / float(W_page), yc / float(H_page), width / float(W_page), height / float(H_page)]
                bboxes.append(b)
                # label
                labels.append(label2index[element['label']])
                # text
                text = element['str']
                texts.append(text)
                # image
                patches.append(page[int(y1):int(y2), int(x1):int(x2)])
                patch_orig = np.zeros((1024, 1024, 3), dtype=page.dtype)
                h = int(y2)-int(y1)
                w = int(x2)-int(x1)
                patch_orig[512-h//2:512+h-h//2, 512-w//2:512+w-w//2] = page[int(y1):int(y2), int(x1):int(x2)]
                patches_orig.append(patch_orig)
                patch_mask = np.zeros((1024, 1024), dtype=page.dtype)
                patch_mask[512-h//2:512+h-h//2, 512-w//2:512+w-w//2] = 255
                patch_masks.append(patch_mask)
            # background image
            background_orig_path = str(fname).replace('manual_json_png_gt_label', 'manual_LaMa_3x_stringOnly_inpainted_background_images').replace('.json', '_inpainted.png')
            assert os.path.isfile(background_orig_path)
            background_orig = PIL.Image.open(background_orig_path)
            background_orig = background_orig.resize((1024, 1024), resample=PIL.Image.BILINEAR)
            background_orig = np.array(background_orig)
            assert background_orig.ndim == 3 and background_orig.shape[2] == 3

            # Lexicographic sort
            sort_idx = lexicographic_sort_idx(bboxes)
            bboxes = [bboxes[i] for i in sort_idx]
            labels = [labels[i] for i in sort_idx]
            texts = [texts[i] for i in sort_idx]
            patches = [patches[i] for i in sort_idx]
            patches_orig = [patches_orig[i] for i in sort_idx]
            patch_masks = [patch_masks[i] for i in sort_idx]

            attr = {'name': fname.name, 'width': W_page, 'height': H_page, 'num_bbox_labels': len(label_list), 'filtered': filtered, 'has_canvas_element': False}
            yield dict(attr=attr, bboxes=bboxes, labels=labels, texts=texts, patches=patches, patches_orig=patches_orig, patch_masks=patch_masks, background_orig=background_orig, page_label=page_label)
            if idx >= max_idx-1:
                break
    return max_idx, iterate_samples()

#----------------------------------------------------------------------------

def open_dataset(source, max_samples: Optional[int]):
    if 'ads_banner_collection_manual' in source:
        if os.path.isdir(source):
            return open_ads_banner_collection_manual_gt(source, max_samples=max_samples)
        error('Missing input directory')
    else:
        error('Unknown dataset')

#----------------------------------------------------------------------------

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest', help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--max-samples', help='Output only up to `max-samples` samples', type=int, default=None)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_samples: Optional[int],
):

    PIL.Image.init() # type: ignore

    if dest == '':
        ctx.fail('--dest output filename or directory must not be an empty string')

    num_files, input_iter_1 = open_dataset(source, max_samples=max_samples)
    _, input_iter_2 = open_dataset(source, max_samples=max_samples)
    archive_root_dir_train, save_bytes_train, close_dest_train = open_dest(os.path.join(dest, 'train.zip'))
    archive_root_dir_val, save_bytes_val, close_dest_val = open_dest(os.path.join(dest, 'val.zip'))

    # Save the bbox and bbox label information.
    samples = []
    for idx, sample in tqdm(enumerate(input_iter_1), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/page{idx_str}'
        samples.append([archive_fname, dict(attr=sample['attr'], bboxes=sample['bboxes'], labels=sample['labels'], texts=sample['texts'], page_label=sample['page_label'])])
    s = int(len(samples) * .90)
    metadata_train = {'samples': samples[:s]}
    save_bytes_train(os.path.join(archive_root_dir_train, 'non_image.json'), json.dumps(metadata_train))
    metadata_val = {'samples': samples[s:]}
    save_bytes_val(os.path.join(archive_root_dir_val, 'non_image.json'), json.dumps(metadata_val))

    # Save the patches and patch masks as an uncompressed PNG.
    for idx, sample in tqdm(enumerate(input_iter_2), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/page{idx_str}'
        patches = sample['patches']
        for i, patch in enumerate(patches):
            patch = PIL.Image.fromarray(patch, 'RGB')
            image_bits = io.BytesIO()
            patch.save(image_bits, format='png', compress_level=0, optimize=False)
            if idx < s:
                save_bytes_train(os.path.join(archive_root_dir_train, archive_fname + '_%d_patch.png' % i), image_bits.getbuffer())
            else:
                save_bytes_val(os.path.join(archive_root_dir_val, archive_fname + '_%d_patch.png' % i), image_bits.getbuffer())
        patches_orig = sample['patches_orig']
        for i, patch_orig in enumerate(patches_orig):
            patch_orig = PIL.Image.fromarray(patch_orig, 'RGB')
            image_bits = io.BytesIO()
            patch_orig.save(image_bits, format='png', compress_level=0, optimize=False)
            if idx < s:
                save_bytes_train(os.path.join(archive_root_dir_train, archive_fname + '_%d_patch_orig.png' % i), image_bits.getbuffer())
            else:
                save_bytes_val(os.path.join(archive_root_dir_val, archive_fname + '_%d_patch_orig.png' % i), image_bits.getbuffer())
        patch_masks = sample['patch_masks']
        for i, patch_mask in enumerate(patch_masks):
            patch_mask = PIL.Image.fromarray(patch_mask, 'L')
            image_bits = io.BytesIO()
            patch_mask.save(image_bits, format='png', compress_level=0, optimize=False)
            if idx < s:
                save_bytes_train(os.path.join(archive_root_dir_train, archive_fname + '_%d_patch_mask.png' % i), image_bits.getbuffer())
            else:
                save_bytes_val(os.path.join(archive_root_dir_val, archive_fname + '_%d_patch_mask.png' % i), image_bits.getbuffer())
        background_orig = sample['background_orig']
        background_orig = PIL.Image.fromarray(background_orig, 'RGB')
        image_bits = io.BytesIO()
        background_orig.save(image_bits, format='png', compress_level=0, optimize=False)
        if idx < s:
            save_bytes_train(os.path.join(archive_root_dir_train, archive_fname + '_background_orig.png'), image_bits.getbuffer())
        else:
            save_bytes_val(os.path.join(archive_root_dir_val, archive_fname + '_background_orig.png'), image_bits.getbuffer())

    close_dest_train()
    close_dest_val()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
