import copy

import torch.utils.data as data
import torch
import os
import os.path
import glob
from torchvision import transforms
import torchvision.transforms.functional as transFunc
import random
import numpy as np
import torch.nn.functional as F
import math
import scipy.io as scio
import torch.utils.data as data
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
import cv2
from imageio.v2 import imread
from skimage.color import rgb2gray, gray2rgb
from torchvision.transforms import ToTensor
from .medical_preprocessing import load_medical_image, read_manifest, rtv_structure
from torchvision.transforms import InterpolationMode
from utils.mask_group_split import group_mask


## TODO: choose with or without transformation at test mode
class Dataset(data.Dataset):
    def __init__(self, gt_file, structure_file, config, mask_file=None, evaluation=False):
        self.config = config
        self.evaluation = evaluation or config.MODE in ("test", "eval")
        self.gt_image_files = self.load_file_list(gt_file)
        self.structure_image_files = self.load_file_list(structure_file)
        self.model = config.MODEL
        self._evaluation_indices = None

        if len(self.gt_image_files) == 0:
            raise (RuntimeError("Found 0 images in the input files " + "\n"))

        if self.evaluation:
            self.transform_opt = {'crop': False,
                                  'flip': False,
                                  'resize': config.DATA_TEST_SIZE,
                                  'random_load_mask': False}
            self.mask_type = 'from_file' if mask_file is not None else config.DATA_MASK_TYPE
        else:
            self.transform_opt = {'crop': config.DATA_CROP, 'flip': config.DATA_FLIP,
                                  'resize': config.DATA_TRAIN_SIZE, 'random_load_mask': True}

        if not self.evaluation:
            self.mask_type = config.DATA_MASK_TYPE
        # generate random rectangle mask
        if self.mask_type == 'random_bbox':
            self.mask_setting = config.DATA_RANDOM_BBOX_SETTING
        # generate random free form mask
        elif self.mask_type == 'random_free_form':
            self.mask_setting = config.DATA_RANDOM_FF_SETTING
        # read masks from files
        elif self.mask_type == 'from_file':
            self.mask_image_files = self.load_file_list(mask_file)
            if not self.mask_image_files: raise ValueError('No mask files.')
        if self.evaluation:
            if self.mask_type != 'from_file':
                raise ValueError('Formal evaluation requires saved masks for reproducible filtering.')
            self._evaluation_indices=[]
            self.evaluation_mask_groups=[]
            for original_index,item in enumerate(self.gt_image_files):
                # Geometry after image processing determines actual mask area.
                image=load_medical_image(item,self.config)
                if self.transform_opt['resize']:
                    image=transFunc.resize(image,self.transform_opt['resize'])
                mask=self.load_mask(original_index,torch.empty(1,image.height,image.width))
                group=group_mask(mask[0].numpy())
                if group is not None:
                    self._evaluation_indices.append(original_index)
                    self.evaluation_mask_groups.append(group)
            if not self._evaluation_indices: raise ValueError('No evaluation masks in the 1%-60% range.')

    def __getitem__(self, index):
        if not self.evaluation:
            return self.load_item(index)
        index = self._evaluation_indices[index]
        # Preserve original image/mask pairing after exclusion.
        np_state, py_state = np.random.get_state(), random.getstate()
        np.random.seed((self.config.EVAL_MASK_SEED or 0) + index)
        random.seed((self.config.EVAL_MASK_SEED or 0) + index)
        try:
            return self.load_item(index)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

    def __len__(self):
        return len(self._evaluation_indices) if self._evaluation_indices is not None else len(self.gt_image_files)

    def load_file_list(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                if flist.lower().endswith('.json'):
                    return read_manifest(flist)
                if flist.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    return [flist]
                return np.atleast_1d(np.genfromtxt(flist, dtype=str, encoding='utf-8')).tolist()
        return []

    def load_item(self, index):
        gt_image = load_medical_image(self.gt_image_files[index], self.config)
        if self.model in (1, 3):
            if self.config.RTV_SOURCE == 'files':
                if len(self.structure_image_files) != len(self.gt_image_files):
                    raise ValueError('Structure file list must match the image list one-to-one.')
                structure_image = loader(self.structure_image_files[index])
                if structure_image.size != gt_image.size:
                    raise ValueError('Cached RTV structure geometry differs from the source image.')
            else:
                structure_image = rtv_structure(gt_image, self.config.RTV_LAMBDA or .015,
                                               self.config.RTV_SIGMA or 3, self.config.RTV_ITERATIONS or 30)
        else:
            structure_image = Image.new('RGB', gt_image.size)
        # Priors are extracted from the complete image before any mask is applied.
        texture_image = Image.fromarray(self.load_lbp(np.asarray(gt_image.convert('L'))), mode='L')
        params = get_params(gt_image.size, self.transform_opt)
        images = []
        for image, nearest in ((gt_image, False), (structure_image, False), (texture_image, True)):
            if params['crop']:
                x, y, w, h = params['crop']
                image = image.crop((x, y, x+w, y+h))
            if params['resize']:
                image = transFunc.resize(image, params['resize'], interpolation=
                                         InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR)
            if params['flip']:
                image = transFunc.hflip(image)
            images.append(transFunc.to_tensor(image))
        gt_image, structure_image, texture_image = images
        mask = self.load_mask(index, gt_image)
        return gt_image * (1-mask), structure_image, texture_image, gt_image, mask

    def load_mask(self, index, img):
        _, w, h = img.shape
        image_shape = [w, h]
        if self.mask_type == 'random_bbox':
            bboxs = []
            for i in range(self.mask_setting['num']):
                bbox = random_bbox(self.mask_setting, image_shape)
                bboxs.append(bbox)
            mask = bbox2mask(bboxs, image_shape, self.mask_setting)
            return torch.from_numpy(mask)

        elif self.mask_type == 'random_free_form':
            mask = random_ff_mask(self.mask_setting, image_shape)
            return torch.from_numpy(mask)

        elif self.mask_type == 'from_file':
            if self.transform_opt['random_load_mask']:
                index = np.random.randint(0, len(self.mask_image_files))
                mask = gray_loader(self.mask_image_files[index])
                # if random.random() > 0.5:
                #     mask = transFunc.hflip(mask)
                # if random.random() > 0.5:
                #     mask = transFunc.vflip(mask)
            else:
                mask = gray_loader(self.mask_image_files[index % len(self.mask_image_files)])
            mask = transFunc.resize(mask, size=image_shape, interpolation=InterpolationMode.NEAREST)
            mask = transFunc.to_tensor(mask)
            mask = (mask > 0).float()
            return mask
        else:
            raise (RuntimeError("No such mask type: %s" % self.mask_type))

    def load_name(self, index, add_mask_name=False):
        if self._evaluation_indices is not None: index=self._evaluation_indices[index]
        name = self.gt_image_files[index]
        # name = self.gt_image_files[index]
        if isinstance(name, dict):
            name = name.get('name', Path(name['path']).name.split('.')[0] + '_slice_' + str(name.get('slice_index', 0)) + '.png')
        name = os.path.basename(name)

        if not add_mask_name:
            return name
        else:
            if len(self.mask_image_files) == 0:
                return name
            else:
                mask_name = os.path.basename(self.mask_image_files[index])
                mask_name, _ = os.path.splitext(mask_name)
                name, ext = os.path.splitext(name)
                name = name + '_' + mask_name + ext
                return name

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def load_lbp(self, img_gray):
        h, w = img_gray.shape
        padded = np.pad(img_gray, 1, mode='constant')
        valid = np.pad(np.ones((h, w), dtype=bool), 1, mode='constant')
        out = np.zeros((h, w), dtype=np.uint8)
        offsets = [(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0)]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
            inside = valid[1+dy:1+dy+h, 1+dx:1+dx+w]
            out |= ((neighbor >= img_gray) & inside).astype(np.uint8) << bit
        return out

    # lbp计算像素
    def lbp_calculated_pixel(self, img, x, y):
        '''
         64 | 128 |   1      -->对应位置      x - 1, y - 1 | x - 1, y | x - 1, y + 1
        ----------------                  ------------------------------------------------
         32 |   0 |   2                        x, y - 1  |center(x,y)| x, y + 1
        ----------------                  ------------------------------------------------
         16 |   8 |   4                    x + 1, y - 1 |  x + 1, y  | x + 1, y + 1
        '''
        center = img[x][y]
        val_ar = []
        # 计算中心点一圈的LBP特征
        val_ar.append(self.get_pixel(img, center, x - 1, y + 1))  # top_right
        val_ar.append(self.get_pixel(img, center, x, y + 1))  # right
        val_ar.append(self.get_pixel(img, center, x + 1, y + 1))  # bottom_right
        val_ar.append(self.get_pixel(img, center, x + 1, y))  # bottom
        val_ar.append(self.get_pixel(img, center, x + 1, y - 1))  # bottom_left
        val_ar.append(self.get_pixel(img, center, x, y - 1))  # left
        val_ar.append(self.get_pixel(img, center, x - 1, y - 1))  # top_left
        val_ar.append(self.get_pixel(img, center, x - 1, y))  # top
        # 按照上面顺序所对应的LBP权重
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            # val = 每个点和对应位置权重相乘相加
            val += val_ar[i] * power_val[i]
        return val

    def get_pixel(self, img, center, x, y):
        # 临近点像素和中心相比较 如果比中心大 那就是1 否则为0
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = np.array(Image.fromarray(img).resize((height, width)))
        # img = resize(img, [height, width])

        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = transFunc.to_tensor(img).float()
        return img_t


def random_bbox(config, shape):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including DATA_NEW_SHAPE,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    img_height = shape[0]
    img_width = shape[1]
    height, width = config['shape']
    ver_margin, hor_margin = config['margin']
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    t = np.random.randint(low=ver_margin, high=maxt)
    l = np.random.randint(low=hor_margin, high=maxl)
    h = height
    w = width
    return (t, l, h, w)


def random_ff_mask(config, shape):
    """Generate a random free form mask with configuration.

    Args:
        config: Config should have configuration including DATA_NEW_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)
    """

    h, w = shape
    mask = np.zeros((h, w))
    num_v = 12 + np.random.randint(
        config['mv'])  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(config['ma'])
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(config['ml'])
            brush_w = 10 + np.random.randint(config['mbw'])
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    return mask.reshape((1,) + mask.shape).astype(np.float32)


def bbox2mask(bboxs, shape, config):
    """Generate mask tensor from bbox.

    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including DATA_NEW_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

    Returns:
        tf.Tensor: output with shape [1, H, W, 1]

    """
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    # print(mask.shape)
    for bbox in bboxs:
        if config['random_size']:
            h = int(0.1 * bbox[2]) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(0.1 * bbox[3]) + np.random.randint(int(bbox[3] * 0.2) + 1)
        else:
            h = 0
            w = 0
        mask[bbox[0] + h:bbox[0] + bbox[2] - h,
        bbox[1] + w:bbox[1] + bbox[3] - w] = 1.
    # print("after", mask.shape)
    return mask.reshape((1,) + mask.shape).astype(np.float32)


def gray_loader(path):
    return Image.open(path).convert('L')


def loader(path):
    return Image.open(path).convert('RGB')


def get_params(size, transform_opt):
    w, h = size
    if transform_opt['flip']:
        flip = random.random() > 0.5
    else:
        flip = False
    if transform_opt['crop']:
        transform_crop = transform_opt['crop'] \
        if w >= transform_opt['crop'][0] and h >= transform_opt['crop'][1] else [h, w]
        x = random.randint(0, np.maximum(0, w - transform_crop[0]))
        y = random.randint(0, np.maximum(0, h - transform_crop[1]))
        crop = [x, y, transform_crop[0], transform_crop[1]]
    else:
        crop = False
    if transform_opt['resize']:
        resize = [transform_opt['resize'], transform_opt['resize'], ]
    else:
        resize = False
    param = {'crop': crop, 'flip': flip, 'resize': resize}
    return param


def transform_image(transform_param, gt_image, structure_image, normalize=True, toTensor=True):
    transform_list = []

    if transform_param['crop']:
        crop_position = transform_param['crop'][:2]
        crop_size = transform_param['crop'][2:]
        transform_list.append(transforms.Lambda(lambda img: __crop(img, crop_position, crop_size)))
    if transform_param['resize']:
        transform_list.append(transforms.Resize(transform_param['resize']))
    if transform_param['flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, True)))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    trans = transforms.Compose(transform_list)
    if gt_image.size != structure_image.size:
        structure_image = transFunc.resize(structure_image, size=gt_image.size)

    gt_image = trans(gt_image)
    structure_image = trans(structure_image)

    return gt_image, structure_image


def transform_texture_image(transform_param, gt_image, texture_image, normalize=True, toTensor=True):
    transform_list = []
    transform_list1 = []
    if transform_param['crop']:
        crop_position = transform_param['crop'][:2]
        crop_size = transform_param['crop'][2:]
        transform_list.append(transforms.Lambda(lambda img: __crop(img, crop_position, crop_size)))
    if transform_param['resize']:
        transform_list.append(transforms.Resize(transform_param['resize']))
    if transform_param['flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, True)))
    if toTensor:
        transform_list += [transforms.ToTensor()]
        transform_list1 += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list1 += [transforms.Normalize(0.5, 0.5)]
    trans = transforms.Compose(transform_list)
    trans1 = transforms.Compose(transform_list1)
    if gt_image.size != texture_image.size:
        texture_image = transFunc.resize(texture_image, size=gt_image.size)

    gt_image = trans(gt_image)
    texture_image = trans1(texture_image)

    return gt_image, texture_image


def transform_all_image(transform_param, gt_image, structure_image, texture_image, normalize=True, toTensor=True):
    transform_list = []
    transform_list1 = []
    if transform_param['crop']:
        crop_position = transform_param['crop'][:2]
        crop_size = transform_param['crop'][2:]
        transform_list.append(transforms.Lambda(lambda img: __crop(img, crop_position, crop_size)))
    if transform_param['resize']:
        transform_list.append(transforms.Resize(transform_param['resize']))
    if transform_param['flip']:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, True)))

    if toTensor:
        transform_list += [transforms.ToTensor()]
        transform_list1 += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
        transform_list1 += [transforms.Normalize(0.5, 0.5)]
    trans = transforms.Compose(transform_list)
    trans1 = transforms.Compose(transform_list1)
    if gt_image.size != texture_image.size:
        texture_image = transFunc.resize(texture_image, size=gt_image.size)

    gt_image = trans(gt_image)
    structure_image = trans(structure_image)
    texture_image = trans1(texture_image)

    return gt_image, structure_image, texture_image


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
