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

def open_image_folder(source_dir, *, max_images: Optional[int]):
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_image_ext(f) and os.path.isfile(f)]

    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            try:
                img = np.array(PIL.Image.open(fname))
            except:
                continue
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_image_zip(source, *, max_images: Optional[int]):
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]

        # Load labels.
        labels = {}
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                labels = json.load(file)['labels']
                if labels is not None:
                    labels = { x[0]: x[1] for x in labels }
                else:
                    labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = PIL.Image.open(file) # type: ignore
                    img = np.array(img)
                yield dict(img=img, label=labels.get(fname))
                if idx >= max_idx-1:
                    break
    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_lmdb(lmdb_dir: str, *, max_images: Optional[int]):
    import cv2  # pip install opencv-python # pylint: disable=import-error
    import lmdb  # pip install lmdb # pylint: disable=import-error

    with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
        max_idx = maybe_min(txn.stat()['entries'], max_images)

    def iterate_images():
        with lmdb.open(lmdb_dir, readonly=True, lock=False).begin(write=False) as txn:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.array(PIL.Image.open(io.BytesIO(value)))
                    yield dict(img=img, label=None)
                    if idx >= max_idx-1:
                        break
                except:
                    print(sys.exc_info()[1])

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def open_mnist(images_gz: str, *, max_images: Optional[int]):
    labels_gz = images_gz.replace('-images-idx3-ubyte.gz', '-labels-idx1-ubyte.gz')
    assert labels_gz != images_gz
    images = []
    labels = []

    with gzip.open(images_gz, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    with gzip.open(labels_gz, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

#----------------------------------------------------------------------------

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error ('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

#----------------------------------------------------------------------------

def open_enrico(source_dir, *, max_samples: Optional[int]):
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
        'Toolbar',
        'Image',
        'Text',
        'Icon',
        'Text Button',
        'Input',
        'List Item',
        'Advertisement',
        'Pager Indicator',
        'Web View',
        'Background Image',
        'Drawer',
        'Modal',
    ]

    label2index = dict()
    for idx, label in enumerate(label_list):
        label2index[label] = idx

    max_idx = maybe_min(len(input_samples), max_samples)

    def iterate_samples():
        for idx, fname in enumerate(input_samples):
            # Load page label
            arch_fname = os.path.relpath(str(fname), source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            page_label = page_labels.get(arch_fname)
            
            with fname.open() as f:
                ann = json.load(f)
            B = ann['bounds']
            W, H = float(B[2]), float(B[3])
            if B[0] != 0 or B[1] != 0 or H < W:
                continue

            page_path = str(fname).replace('semantic_annotations', 'combined').replace('.json', '.png')
            assert os.path.isfile(page_path)
            page = np.array(PIL.Image.open(page_path))
            H_page = page.shape[0]
            W_page = page.shape[1]
            if page.ndim == 2:
                page = np.stack((page, page, page), axis=2)

            def is_valid(element):
                if element['componentLabel'] not in label_list:
                    return False
                x1, y1, x2, y2 = element['bounds']
                if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                    return False
                if x2 <= x1 or y2 <= y1:
                    return False
                x1 = round(float(x1) / float(W) * float(W_page)) # /1440 *540
                y1 = round(float(y1) / float(H) * float(H_page)) # /2560 *960
                x2 = round(float(x2) / float(W) * float(W_page)) # /1440 *540
                y2 = round(float(y2) / float(H) * float(H_page)) # /2560 *960
                if x1 == x2 or y1 == y2:
                    return False
                width = int(x2) - int(x1)
                height = int(y2) - int(y1)
                if width > height:
                    height_new = int(float(height) / float(width) * 256.0) // 2 * 2
                    if height_new == 0:
                        return False
                else:
                    width_new = int(float(width) / float(height) * 256.0) // 2 * 2
                    if width_new == 0:
                        return False
                return True

            elements = append_child(ann, [])
            _elements = list(filter(is_valid, elements))
            # my filter to keep only elements that are not overlapped by other elements: there is no bbox in the current bbox
            valid_list = []
            for i, e in enumerate(_elements):
                x1, y1, x2, y2 = e['bounds']
                valid = True
                for j, ee in enumerate(_elements):
                    if i != j:
                        xx1, yy1, xx2, yy2 = ee['bounds']
                        if not (xx1 > x2 or xx2 < x1 or yy1 > y2 or yy2 < y1):
                            valid = False
                            break
                valid_list.append(valid)
            _elements = [e for i, e in enumerate(_elements) if valid_list[i]]
            filtered = len(elements) != len(_elements)
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
            for element in elements:
                # bbox
                x1, y1, x2, y2 = element['bounds']
                x1 = round(float(x1) / float(W) * float(W_page)) # /1440 *540
                y1 = round(float(y1) / float(H) * float(H_page)) # /2560 *960
                x2 = round(float(x2) / float(W) * float(W_page)) # /1440 *540
                y2 = round(float(y2) / float(H) * float(H_page)) # /2560 *960
                xc = (x1 + x2) / 2.
                yc = (y1 + y2) / 2.
                width = x2 - x1
                height = y2 - y1
                b = [xc / float(W_page), yc / float(H_page), width / float(W_page), height / float(H_page)]
                bboxes.append(b)
                # label
                l = element['componentLabel']
                labels.append(label2index[l])
                # text
                text = element['text'] if 'text' in element else ''
                texts.append(text)
                # image
                patches.append(page[int(y1):int(y2), int(x1):int(x2)])
                patch_orig = np.zeros((H_page, W_page, 3), dtype=page.dtype)
                hc = H_page // 2
                wc = W_page // 2
                h = int(y2)-int(y1)
                w = int(x2)-int(x1)
                patch_orig[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = page[int(y1):int(y2), int(x1):int(x2)]
                patches_orig.append(patch_orig)
                patch_mask = np.zeros((H_page, W_page), dtype=page.dtype)
                patch_mask[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = 255
                patch_masks.append(patch_mask)
            # background image
            background_orig_path = str(fname).replace('semantic_annotations', 'LaMa_inpainted_background_images').replace('.json', '_inpainted.png')
            assert os.path.isfile(background_orig_path)
            background_orig = PIL.Image.open(background_orig_path)
            background_orig = background_orig.resize((W_page, H_page), resample=PIL.Image.BILINEAR)
            background_orig = np.array(background_orig)
            assert background_orig.ndim == 3 and background_orig.shape[2] == 3 and background_orig.shape[0] == H_page and background_orig.shape[1] == W_page

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

def open_clay(source_dir, *, max_samples: Optional[int]):
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
        'TOOLBAR',
        'IMAGE',
        'TEXT',
        'PICTOGRAM',
        'BUTTON',
        'TEXT_INPUT',
        'LIST_ITEM',
        'ADVERTISEMENT',
        'PAGER_INDICATOR',
        'Web View', # removed from CLAY
        'Background Image', # removed from CLAY
        'DRAWER',
        'Modal' # removed from CLAY
        ]

    label2index = dict()
    for idx, label in enumerate(label_list):
        label2index[label] = idx

    max_idx = maybe_min(len(input_samples), max_samples)

    with open(source_dir.replace('combined', 'label_map.txt')) as f:
        lines = f.readlines()
    label_dict = {}
    for line in lines:
        key_value = line.split(': ')
        key = key_value[0]
        value = key_value[1][:-1]
        label_dict[key] = value

    clay_dict = {}
    with open(source_dir.replace('combined', 'clay_labels.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for count, row in enumerate(spamreader):
            if row[0] != 'screen_id':
                screen_id = row[0]
                node_id = row[1]
                label = row[2]
                if screen_id not in clay_dict:
                    clay_dict[screen_id] = {}
                clay_dict[screen_id][node_id] = label_dict[label]

    H_page = 960
    W_page = 540
    W_json = 1440
    H_json = 2560
    def iterate_samples():
        for idx, fname in enumerate(input_samples):
            screen_id = str(fname)[len(source_dir)+1:-5]
            if screen_id not in clay_dict:
                continue
            page = PIL.Image.open(str(fname).replace('.json', '.jpg'))
            if float(page.size[0]) / float(page.size[1]) != float(W_page) / float(H_page):
                continue

            # Load page label
            arch_fname = os.path.relpath(str(fname), source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            page_label = page_labels.get(arch_fname)
            
            with fname.open() as f:
                ann = json.load(f)
            B = ann['activity']['root']['bounds']
            if B[0] != 0 or B[1] != 0:
                continue

            def is_valid(element):
                if element['pointer'] not in clay_dict[screen_id]:
                    return False
                if clay_dict[screen_id][element['pointer']] not in label_list:
                    return False
                x1, y1, x2, y2 = element['bounds']
                if x1 < 0 or y1 < 0 or W_json < x2 or H_json < y2:
                    return False
                if x2 <= x1 or y2 <= y1:
                    return False
                x1 = round(float(x1) / float(W_json) * float(W_page)) # /1440 *540
                y1 = round(float(y1) / float(H_json) * float(H_page)) # /2560 *960
                x2 = round(float(x2) / float(W_json) * float(W_page)) # /1440 *540
                y2 = round(float(y2) / float(H_json) * float(H_page)) # /2560 *960
                if x1 == x2 or y1 == y2:
                    return False
                width = int(x2) - int(x1)
                height = int(y2) - int(y1)
                if width > height:
                    height_new = int(float(height) / float(width) * 256.0) // 2 * 2
                    if height_new == 0:
                        return False
                else:
                    width_new = int(float(width) / float(height) * 256.0) // 2 * 2
                    if width_new == 0:
                        return False
                return True

            try:
                elements = append_child(ann['activity']['root'], [])
            except:
                continue
            _elements = list(filter(is_valid, elements))
            # my filter to keep only elements that are not overlapped by other elements: there is no bbox in the current bbox
            valid_list = []
            for i, e in enumerate(_elements):
                x1, y1, x2, y2 = e['bounds']
                valid = True
                for j, ee in enumerate(_elements):
                    if i != j:
                        xx1, yy1, xx2, yy2 = ee['bounds']
                        if not (xx1 > x2 or xx2 < x1 or yy1 > y2 or yy2 < y1):
                            valid = False
                            break
                valid_list.append(valid)
            _elements = [e for i, e in enumerate(_elements) if valid_list[i]]
            filtered = len(elements) != len(_elements)
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
            if page.size[0] != W_page or page.size[1] != H_page:
                page = page.resize((W_page, H_page), PIL.Image.ANTIALIAS)
            page = np.array(page)
            if page.ndim == 2:
                page = np.stack((page, page, page), axis=2)
            for element in elements:
                # bbox
                x1, y1, x2, y2 = element['bounds']
                x1 = round(float(x1) / float(W_json) * float(W_page)) # /1440 *540
                y1 = round(float(y1) / float(H_json) * float(H_page)) # /2560 *960
                x2 = round(float(x2) / float(W_json) * float(W_page)) # /1440 *540
                y2 = round(float(y2) / float(H_json) * float(H_page)) # /2560 *960
                xc = (x1 + x2) / 2.
                yc = (y1 + y2) / 2.
                width = x2 - x1
                height = y2 - y1
                b = [xc / float(W_page), yc / float(H_page), width / float(W_page), height / float(H_page)]
                bboxes.append(b)
                # label
                l = clay_dict[screen_id][element['pointer']]
                labels.append(label2index[l])
                # text
                text = element['text'] if 'text' in element else ''
                texts.append(text)
                # image
                patches.append(page[int(y1):int(y2), int(x1):int(x2)])
                patch_orig = np.zeros((H_page, W_page, 3), dtype=page.dtype)
                hc = H_page // 2
                wc = W_page // 2
                h = int(y2)-int(y1)
                w = int(x2)-int(x1)
                patch_orig[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = page[int(y1):int(y2), int(x1):int(x2)]
                patches_orig.append(patch_orig)
                patch_mask = np.zeros((H_page, W_page), dtype=page.dtype)
                patch_mask[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = 255
                patch_masks.append(patch_mask)
            # background image
            background_orig_path = str(fname).replace('combined', 'LaMa_inpainted_background_images').replace('.json', '_inpainted.png')
            assert os.path.isfile(background_orig_path)
            background_orig = PIL.Image.open(background_orig_path)
            background_orig = background_orig.resize((W_page, H_page), resample=PIL.Image.BILINEAR)
            background_orig = np.array(background_orig)
            assert background_orig.ndim == 3 and background_orig.shape[2] == 3 and background_orig.shape[0] == H_page and background_orig.shape[1] == W_page

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

def open_ads_banner_collection(source_dir, *, max_samples: Optional[int]):
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
        'TOOLBAR',
        'IMAGE',
        'TEXT',
        'PICTOGRAM',
        'BUTTON',
        'TEXT_INPUT',
        'LIST_ITEM',
        'ADVERTISEMENT',
        'PAGER_INDICATOR',
        'Web View', # removed from CLAY
        'Background Image', # removed from CLAY
        'DRAWER',
        'Modal' # removed from CLAY
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
                if 'is_cluster' not in element or element['is_cluster']:
                    return False
                if 'str' not in element or len(element['str']) == 0:
                    return False
                if 'word' not in element or len(element['word']) == 0:
                    return False
                x1, y1, x2, y2 = element['xyxy']
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
                x1, y1, x2, y2 = e['xyxy']
                valid = True
                for j, ee in enumerate(_elements):
                    if i != j:
                        xx1, yy1, xx2, yy2 = ee['xyxy']
                        if not (xx1 > x2 or xx2 < x1 or yy1 > y2 or yy2 < y1):
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
                x1, y1, x2, y2 = element['xyxy']
                xc = (x1 + x2) / 2.
                yc = (y1 + y2) / 2.
                width = x2 - x1
                height = y2 - y1
                b = [xc / float(W_page), yc / float(H_page), width / float(W_page), height / float(H_page)]
                bboxes.append(b)
                # label
                labels.append(label2index['TEXT'])
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
            background_orig_path = str(fname).replace('json_png_pseudo_label_blue_bbox_filtered', 'LaMa_3x_stringOnly_inpainted_background_images_edge').replace('.json', '_inpainted.png')
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

def open_ads_banner_collection_manual_gt(source_dir, *, max_samples: Optional[int]):
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

def open_ads_banner_collection_manual_gt_header_consolidated(source_dir, *, max_samples: Optional[int]):
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
        #'pre-header',
        #'post-header',
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
                if 'label' not in element:
                    return False
                label = element['label']
                if label == 'pre-header' or label == 'post-header':
                    label = 'header'
                if label not in label_list:
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
                label = element['label']
                if label == 'pre-header' or label == 'post-header':
                    label = 'header'
                labels.append(label2index[label])
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

def open_ads_banner_collection_manual_gt_header_consolidated_3labels(source_dir, *, max_samples: Optional[int]):
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
        #'pre-header',
        #'post-header',
        'body text',
        #'disclaimer / footnote',
        'button',
        #'callout',
        #'logo'
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
                if 'label' not in element:
                    return False
                label = element['label']
                if label == 'pre-header' or label == 'post-header':
                    label = 'header'
                if label not in label_list:
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
                label = element['label']
                if label == 'pre-header' or label == 'post-header':
                    label = 'header'
                labels.append(label2index[label])
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

def open_floors_product_samples(source_dir, *, max_samples: Optional[int]):
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
            background_orig_path = str(fname).replace('json_png_manual_label', 'LaMa_stringOnly_inpainted_background_images').replace('.json', '_inpainted.png')
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

def open_AMT_uploaded_ads_banners(source_dir, *, max_samples: Optional[int]):
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
        'body',
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
            page = PIL.Image.open(str(fname).replace('ocr_bbox_refine', 'images_unique_png').replace('out_', '').replace('.json', '.png'))
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
                x1, y1, x2, y2 = element['xyxy']
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
                x1, y1, x2, y2 = e['xyxy']
                valid = True
                for j, ee in enumerate(_elements):
                    if i != j:
                        xx1, yy1, xx2, yy2 = ee['xyxy']
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
                x1, y1, x2, y2 = element['xyxy']
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
            background_orig_path = str(fname).replace('ocr_bbox_refine', 'manual_LaMa_3x_stringOnly_inpainted_background_images').replace('out_', '').replace('.json', '_inpainted.png')
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

def open_ads_banner_collection_AMT_uploaded_ads_banners(source_dir, *, max_samples: Optional[int]):
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

def open_cgl_dataset(source_dir, *, max_samples: Optional[int]):
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
        'Logo',
        'Text',
        'Underlay',
        'Embellishment',
        'Highlighted text'
        ]

    label2index = dict()
    for idx, label in enumerate(label_list):
        label2index[label] = idx

    max_idx = maybe_min(len(input_samples), max_samples)

    W_standard = 513
    H_standard = 750
    def iterate_samples():
        for idx, fname in enumerate(input_samples):
            page = PIL.Image.open(str(fname).replace('layout_imgs_6w_ocr', 'layout_imgs_6w_png').replace('.json', '.png'))
            W_page = page.size[0]
            H_page = page.size[1]
            if float(W_page) / float(H_page) != float(W_standard) / float(H_standard):
                continue

            # Load page label
            arch_fname = os.path.relpath(str(fname), source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            page_label = page_labels.get(arch_fname)
            
            with fname.open() as f:
                ann = json.load(f)

            def is_valid(element):
                if 'label' not in element or element['label'] not in label_list:
                    return False
                if 'str' not in element or len(element['str']) == 0 or len(element['str']) >= 64:
                    return False
                x1, y1, x2, y2 = element['xyxy']
                if x1 < 0 or y1 < 0 or W_page < x2 or H_page < y2:
                    return False
                if x2 <= x1 or y2 <= y1:
                    return False
                width = int(x2) - int(x1)
                height = int(y2) - int(y1)
                if width > W_standard or height > H_standard:
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
                x1, y1, x2, y2 = e['xyxy']
                valid = True
                for j, ee in enumerate(_elements):
                    if i != j:
                        xx1, yy1, xx2, yy2 = ee['xyxy']
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
                x1, y1, x2, y2 = element['xyxy']
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
                patch_orig = np.zeros((H_standard, W_standard, 3), dtype=page.dtype)
                hc = H_standard // 2
                wc = W_standard // 2
                h = int(y2)-int(y1)
                w = int(x2)-int(x1)
                patch_orig[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = page[int(y1):int(y2), int(x1):int(x2)]
                patches_orig.append(patch_orig)
                patch_mask = np.zeros((H_standard, W_standard), dtype=page.dtype)
                patch_mask[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = 255
                patch_masks.append(patch_mask)
            # background image
            background_orig_path = str(fname).replace('layout_imgs_6w_ocr', 'LaMa_3x_stringOnly_inpainted_background_images').replace('.json', '_inpainted.png')
            assert os.path.isfile(background_orig_path)
            background_orig = PIL.Image.open(background_orig_path)
            if background_orig.width != W_standard or background_orig.height != H_standard:
                background_orig = background_orig.resize((W_standard, H_standard), resample=PIL.Image.BILINEAR)
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

            attr = {'name': fname.name, 'width': W_standard, 'height': H_standard, 'num_bbox_labels': len(label_list), 'filtered': filtered, 'has_canvas_element': False}
            yield dict(attr=attr, bboxes=bboxes, labels=labels, texts=texts, patches=patches, patches_orig=patches_orig, patch_masks=patch_masks, background_orig=background_orig, page_label=page_label)
            if idx >= max_idx-1:
                break
    return max_idx, iterate_samples()

#----------------------------------------------------------------------------

def open_cgl_dataset_with_saliency_map(source_dir, *, max_samples: Optional[int]):
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
        'Logo',
        'Text',
        'Underlay',
        'Embellishment',
        'Highlighted text'
        ]

    label2index = dict()
    for idx, label in enumerate(label_list):
        label2index[label] = idx

    max_idx = maybe_min(len(input_samples), max_samples)

    W_standard = 513
    H_standard = 750
    def iterate_samples():
        for idx, fname in enumerate(input_samples):
            page = PIL.Image.open(str(fname).replace('layout_imgs_6w_ocr', 'layout_imgs_6w_png').replace('.json', '.png'))
            W_page = page.size[0]
            H_page = page.size[1]
            if float(W_page) / float(H_page) != float(W_standard) / float(H_standard):
                continue

            # Load page label
            arch_fname = os.path.relpath(str(fname), source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            page_label = page_labels.get(arch_fname)
            
            with fname.open() as f:
                ann = json.load(f)

            def is_valid(element):
                if 'label' not in element or element['label'] not in label_list:
                    return False
                if 'str' not in element or len(element['str']) == 0 or len(element['str']) >= 64:
                    return False
                x1, y1, x2, y2 = element['xyxy']
                if x1 < 0 or y1 < 0 or W_page < x2 or H_page < y2:
                    return False
                if x2 <= x1 or y2 <= y1:
                    return False
                width = int(x2) - int(x1)
                height = int(y2) - int(y1)
                if width > W_standard or height > H_standard:
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
                x1, y1, x2, y2 = e['xyxy']
                valid = True
                for j, ee in enumerate(_elements):
                    if i != j:
                        xx1, yy1, xx2, yy2 = ee['xyxy']
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
                x1, y1, x2, y2 = element['xyxy']
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
                patch_orig = np.zeros((H_standard, W_standard, 3), dtype=page.dtype)
                hc = H_standard // 2
                wc = W_standard // 2
                h = int(y2)-int(y1)
                w = int(x2)-int(x1)
                patch_orig[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = page[int(y1):int(y2), int(x1):int(x2)]
                patches_orig.append(patch_orig)
                patch_mask = np.zeros((H_standard, W_standard), dtype=page.dtype)
                patch_mask[hc-h//2:hc+h-h//2, wc-w//2:wc+w-w//2] = 255
                patch_masks.append(patch_mask)
            # background image
            background_orig_path = str(fname).replace('layout_imgs_6w_ocr', 'LaMa_3x_stringOnly_inpainted_background_images').replace('.json', '_inpainted.png')
            assert os.path.isfile(background_orig_path)
            background_orig = PIL.Image.open(background_orig_path)
            if background_orig.width != W_standard or background_orig.height != H_standard:
                background_orig = background_orig.resize((W_standard, H_standard), resample=PIL.Image.BILINEAR)
            background_orig = np.array(background_orig)
            assert background_orig.ndim == 3 and background_orig.shape[2] == 3
            # saliency map
            saliency_orig_path = str(fname).replace('layout_imgs_6w_ocr', 'LaMa_3x_stringOnly_inpainted_background_images_saliency_map').replace('.json', '_saliency_map.png')
            assert os.path.isfile(saliency_orig_path)
            saliency_orig = PIL.Image.open(saliency_orig_path)
            if saliency_orig.width != W_standard or saliency_orig.height != H_standard:
                saliency_orig = saliency_orig.resize((W_standard, H_standard), resample=PIL.Image.BILINEAR)
            saliency_orig = np.array(saliency_orig)
            assert saliency_orig.ndim == 2

            # Lexicographic sort
            sort_idx = lexicographic_sort_idx(bboxes)
            bboxes = [bboxes[i] for i in sort_idx]
            labels = [labels[i] for i in sort_idx]
            texts = [texts[i] for i in sort_idx]
            patches = [patches[i] for i in sort_idx]
            patches_orig = [patches_orig[i] for i in sort_idx]
            patch_masks = [patch_masks[i] for i in sort_idx]

            attr = {'name': fname.name, 'width': W_standard, 'height': H_standard, 'num_bbox_labels': len(label_list), 'filtered': filtered, 'has_canvas_element': False}
            yield dict(attr=attr, bboxes=bboxes, labels=labels, texts=texts, patches=patches, patches_orig=patches_orig, patch_masks=patch_masks, background_orig=background_orig, saliency_orig=saliency_orig, page_label=page_label)
            if idx >= max_idx-1:
                break
    return max_idx, iterate_samples()

#----------------------------------------------------------------------------

def open_dataset(source, *, max_samples: Optional[int]):
    if 'enrico' in source:
        if os.path.isdir(source):
            return open_enrico(source, max_samples=max_samples)
        error('Missing input directory')
    elif 'clay' in source:
        if os.path.isdir(source):
            return open_clay(source, max_samples=max_samples)
        error('Missing input directory')
    elif 'ads_banner_collection' in source:
        if os.path.isdir(source):
            if 'ads_banner_collection_manual' in source:
                #return open_ads_banner_collection_manual_gt(source, max_samples=max_samples)
                #return open_ads_banner_collection_manual_gt_header_consolidated(source, max_samples=max_samples)
                return open_ads_banner_collection_manual_gt_header_consolidated_3labels(source, max_samples=max_samples)
            return open_ads_banner_collection(source, max_samples=max_samples)
        error('Missing input directory')
    elif 'AMT_uploaded_ads_banners_plus_final' in source:
        if os.path.isdir(source):
            return open_ads_banner_collection_AMT_uploaded_ads_banners(source, max_samples=max_samples)
        error('Missing input directory')
    elif 'floors_product_samples' in source:
        if os.path.isdir(source):
            return open_floors_product_samples(source, max_samples=max_samples)
        error('Missing input directory')
    elif 'cgl_dataset' in source:
        if os.path.isdir(source):
            if 'with_saliency_map' in source:
                return open_cgl_dataset_with_saliency_map(source, max_samples=max_samples)
            return open_cgl_dataset(source, max_samples=max_samples)
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
@click.option('--transform', help='Input crop/resize mode', type=click.Choice(['center-crop', 'center-crop-wide']))
@click.option('--resolution', help='Output resolution (e.g., \'512x512\')', metavar='WxH', type=parse_tuple)
def convert_dataset(
    ctx: click.Context,
    source: str,
    dest: str,
    max_samples: Optional[int],
    transform: Optional[str],
    resolution: Optional[Tuple[int, int]]
):
    """Convert an image dataset into a dataset archive usable with StyleGAN2 ADA PyTorch.

    The input dataset format is guessed from the --source argument:

    \b
    --source *_lmdb/                    Load LSUN dataset
    --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
    --source train-images-idx3-ubyte.gz Load MNIST dataset
    --source path/                      Recursively load all images from path/
    --source dataset.zip                Recursively load all images from dataset.zip

    Specifying the output format and path:

    \b
    --dest /path/to/dir                 Save output files under /path/to/dir
    --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip

    The output dataset format can be either an image folder or an uncompressed zip archive.
    Zip archives makes it easier to move datasets around file servers and clusters, and may
    offer better training performance on network file systems.

    Images within the dataset archive will be stored as uncompressed PNG.
    Uncompresed PNGs can be efficiently decoded in the training loop.

    Class labels are stored in a file called 'dataset.json' that is stored at the
    dataset root folder.  This file has the following structure:

    \b
    {
        "labels": [
            ["00000/img00000000.png",6],
            ["00000/img00000001.png",9],
            ... repeated for every image in the datase
            ["00049/img00049999.png",1]
        ]
    }

    If the 'dataset.json' file cannot be found, the dataset is interpreted as
    not containing class labels.

    Image scale/crop and resolution requirements:

    Output images must be square-shaped and they must all have the same power-of-two
    dimensions.

    To scale arbitrary input image size to a specific width and height, use the
    --resolution option.  Output resolution will be either the original
    input resolution (if resolution was not specified) or the one specified with
    --resolution option.

    Use the --transform=center-crop or --transform=center-crop-wide options to apply a
    center crop transform on the input image.  These options should be used with the
    --resolution option.  For example:

    \b
    python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \\
        --transform=center-crop-wide --resolution=512x384
    """

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
        if 'saliency_orig' in sample:
            saliency_orig = sample['saliency_orig']
            saliency_orig = PIL.Image.fromarray(saliency_orig, 'L')
            image_bits = io.BytesIO()
            saliency_orig.save(image_bits, format='png', compress_level=0, optimize=False)
            if idx < s:
                save_bytes_train(os.path.join(archive_root_dir_train, archive_fname + '_saliency_orig.png'), image_bits.getbuffer())
            else:
                save_bytes_val(os.path.join(archive_root_dir_val, archive_fname + '_saliency_orig.png'), image_bits.getbuffer())

    close_dest_train()
    close_dest_val()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
