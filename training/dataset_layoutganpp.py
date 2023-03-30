import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

import seaborn as sns

#----------------------------------------------------------------------------

def to_dense_batch(data, is_str=False):
    if not is_str:
        shape_temp = list(data.shape)
        if shape_temp[0] == 9:
            data_batch = np.array(data, dtype=data.dtype)
        else:
            data_batch = np.zeros([9]+shape_temp[1:], dtype=data.dtype)
            data_batch[:shape_temp[0]] = data
        mask = np.array([1] * shape_temp[0] + [0] * (9-shape_temp[0]), dtype=np.bool)
    else:
        data_batch = list(data) + [''] * (9-len(data))
        mask = np.array([1] * len(data) + [0] * (9-len(data)), dtype=np.bool)
    return data_batch, mask

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        num_bbox_labels,       # Number of bbox labels.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        #xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        background_size = 1024,     # Background image resolution for training.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._num_bbox_labels = num_bbox_labels
        self._colors = None
        self._use_labels = use_labels
        self.background_size = background_size
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        #self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        #if xflip:
        #    self._raw_idx = np.tile(self._raw_idx, 2)
        #    self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_data(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        sample = self._load_raw_data(self._raw_idx[idx])
        assert isinstance(sample['bboxes'], np.ndarray)
        assert isinstance(sample['labels'], np.ndarray)
        assert isinstance(sample['texts'], list)
        assert isinstance(sample['patches'], np.ndarray)
        assert isinstance(sample['patches_orig'], np.ndarray)
        assert isinstance(sample['patch_masks'], np.ndarray)
        assert isinstance(sample['mask'], np.ndarray)
        assert isinstance(sample['background'], np.ndarray)
        assert isinstance(sample['background_orig'], np.ndarray)
        assert list(sample['patches'].shape) == [self.patch_shape[0], 3, 256, 256]
        assert list(sample['patches_orig'].shape) == self.patch_shape
        assert list(sample['background'].shape) == [3, self.background_size, self.background_size]
        assert list(sample['background_orig'].shape) == self.patch_shape[1:]
        #if self._xflip[idx]:
        #    assert sample['patches'].ndim == 4 # NCHW
        #    sample['patches'] = sample['patches'][:, :, ::-1]
        return sample, self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        #d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def patch_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_assets(self):
        assert len(self.patch_shape) == 4 # NCHW
        return self.patch_shape[0]

    @property
    def num_channels(self):
        assert len(self.patch_shape) == 4 # NCHW
        return self.patch_shape[1]

    @property
    def height(self):
        assert len(self.patch_shape) == 4 # NCHW
        return self.patch_shape[2]

    @property
    def width(self):
        assert len(self.patch_shape) == 4 # NCHW
        return self.patch_shape[3]

    @property
    def background_size_for_training(self):
        return self.background_size

    @property
    def num_bbox_labels(self):
        return self._num_bbox_labels

    @property
    def colors(self):
        if self._colors is None:
            colors = sns.color_palette('husl', n_colors=self._num_bbox_labels)
            self._colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
        return self._colors

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class LayoutDataset(Dataset):
    def __init__(self,
        path,                       # Path to directory or zip.
        xflip           = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        background_size = 1024,     # Background image resolution for training.
        **super_kwargs,             # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self.background_size = background_size
        self._zipfile = None

        if self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a zip')

        PIL.Image.init()

        assert 'non_image.json' in self._all_fnames
        with self._open_file('non_image.json') as f:
            self._samples = json.load(f)['samples']

        name = self._path.split('/')[-3]
        raw_shape = [len(self._samples)] + list(self._load_raw_data(0)['patches_orig'].shape)
        num_bbox_labels = self._samples[0][1]['attr']['num_bbox_labels']
        super().__init__(name=name, raw_shape=raw_shape, num_bbox_labels=num_bbox_labels, background_size=background_size, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_data(self, raw_idx):
        rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
        rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))
        # Load bboxes
        bboxes = np.array(self._samples[raw_idx][1]['bboxes'])
        bboxes_batch, mask = to_dense_batch(bboxes)
        bboxes_batch = bboxes_batch.astype(np.float32)
        # Load bbox labels
        labels = np.array(self._samples[raw_idx][1]['labels'])
        labels_batch, _ = to_dense_batch(labels)
        labels_batch = labels_batch.astype(np.int64)
        # Load bbox texts
        texts = self._samples[raw_idx][1]['texts']
        texts_batch, _ = to_dense_batch(texts, is_str=True)
        # Load bbox patches
        base_fname = self._samples[raw_idx][0]
        patches = []
        for i in range(bboxes.shape[0]):
            patch_fname = base_fname + '_%d_patch.png' % i
            with self._open_file(patch_fname) as f:
                patch_orig = PIL.Image.open(f)
                width = patch_orig.width
                height = patch_orig.height
                if width > height:
                    width_new = 256
                    height_new = int(float(height) / float(width) * 256.0) // 2 * 2
                else:
                    height_new = 256
                    width_new = int(float(width) / float(height) * 256.0) // 2 * 2
                patch_temp = np.array(patch_orig.resize((width_new,height_new), PIL.Image.ANTIALIAS))
            assert patch_temp.ndim == 3 and patch_temp.shape[2] == 3
            patch_temp = (patch_temp.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
            patch = np.zeros((256, 256, 3)).astype(np.float32)
            patch[128-height_new//2:128+height_new//2, 128-width_new//2:128+width_new//2] = patch_temp
            patch = patch.transpose(2, 0, 1) # HWC => CHW
            patches.append(patch)
        patches = np.stack(patches, axis=0)
        patches_batch, _ = to_dense_batch(patches)
        # Load bbox patches (original)
        patches_orig = []
        for i in range(bboxes.shape[0]):
            patch_orig_fname = base_fname + '_%d_patch_orig.png' % i
            with self._open_file(patch_orig_fname) as f:
                patch_orig = np.array(PIL.Image.open(f))
            assert patch_orig.ndim == 3 and patch_orig.shape[2] == 3
            patch_orig = (patch_orig.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
            patch_orig = patch_orig.transpose(2, 0, 1) # HWC => CHW
            patches_orig.append(patch_orig)
        patches_orig = np.stack(patches_orig, axis=0)
        patches_orig_batch, _ = to_dense_batch(patches_orig)
        # Load bbox patch masks
        patch_masks = []
        for i in range(bboxes.shape[0]):
            patch_mask_fname = base_fname + '_%d_patch_mask.png' % i
            with self._open_file(patch_mask_fname) as f:
                patch_mask = np.array(PIL.Image.open(f))[:,:,np.newaxis]
            assert patch_mask.ndim == 3
            patch_mask = patch_mask.astype(np.float32) / 255.0
            patch_mask = patch_mask.transpose(2, 0, 1) # HWC => CHW
            patch_masks.append(patch_mask)
        patch_masks = np.stack(patch_masks, axis=0)
        patch_masks_batch, _ = to_dense_batch(patch_masks)
        # Load background image
        background_orig_fname = base_fname + '_background_orig.png'
        with self._open_file(background_orig_fname) as f:
            background_orig = PIL.Image.open(f)
            background = np.array(background_orig.resize((self.background_size, self.background_size), PIL.Image.ANTIALIAS))
            background_orig = np.array(background_orig)
        assert background.ndim == 3 and background.shape[2] == 3
        background = (background.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
        background = background.transpose(2, 0, 1) # HWC => CHW
        assert background_orig.ndim == 3 and background_orig.shape[2] == 3
        background_orig = (background_orig.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
        background_orig = background_orig.transpose(2, 0, 1) # HWC => CHW
        return dict(name=self._samples[raw_idx][1]['attr']['name'], W_page=self._samples[raw_idx][1]['attr']['width'], H_page=self._samples[raw_idx][1]['attr']['height'],
                    bboxes=bboxes_batch, labels=labels_batch, texts=texts_batch, patches=patches_batch, patches_orig=patches_orig_batch, patch_masks=patch_masks_batch, mask=mask, background=background, background_orig=background_orig)

    def _load_raw_labels(self):
        labels = [[sample[0], sample[1]['page_label']] for sample in self._samples]
        if any([label[1] is None for label in labels]):
            return None
        #labels = dict(labels)
        #labels = [labels[fname.replace('\\', '/')] for fname in self._patch_fnames]
        #labels = np.array(labels)
        #labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        #return labels

#----------------------------------------------------------------------------
