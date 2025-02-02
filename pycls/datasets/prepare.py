#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
from skimage import color

import pycls.datasets.transforms as transforms
from pycls.core.config import cfg


def prepare_rot(im, dataset, split, mean, sd, eig_vals=None, eig_vecs=None):
    im = prepare_im(im, dataset, split, mean, sd, eig_vals=eig_vals, eig_vecs=eig_vecs)   # HWC→CHW， 图像增广，翻转，crop等
    rot_im = []
    for i in range(4):
        rot_im.append(np.rot90(im, i, (1, 2))) #i指转90度的次数，(1,2)指 H，W轴旋转i*90°
    im = np.stack(rot_im, axis=0)   # [(C,H,W),(C,H,W),(C,H,W),(C,H,W)] → (4,C,H,W)
    label = np.array([0, 1, 2, 3])
    return im, label


def prepare_col(im, dataset, split, nbrs, mean, sd, eig_vals=None, eig_vecs=None):
    # (32, 32, 3)
    if "cifar" not in dataset:
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            if "imagenet" in dataset:
                im = transforms.random_sized_crop(im=im, size=train_size, area_frac=0.08)
            elif "cityscapes" in dataset:
                random_scale = np.power(2, -1 + 2 * np.random.uniform())
                random_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
                im = transforms.scale(random_size, im)
                im = transforms.random_crop(im, train_size, order="HWC")
        else:
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
    if split == "train":
        im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")  # Best before rgb2lab because otherwise need to flip together with label
    im_lab = color.rgb2lab(im.astype(np.uint8, copy=False)) #(32, 32, 3)
    im = transforms.HWC2CHW(im_lab[:, :, 0:1]).astype(np.float32, copy=False)   # (1,32,32),把im_lab图像的L通道作为im, 剩下的ab通道作为label
    
    # Ad hoc normalization of the L channel 
    # im是L通道，用rgb三个值的平均作为mean，三个值的sd的平均作sd来标准化 只有L通道的im
    im = im / 100.0
    im = im - np.mean(mean)
    im = im / np.mean(sd)
    
    # 1) 取(32,32,3）后两个ab通道，reshape成 (1024,2), 送入到knn中找到代表的量化值（从313中选择一个值）
    # 2) knn输入(1024,2)，1024代表像素点，2代表每个像素点的ab两个通道值
    # 3）knn的nbr设置是1，即输出的特征维度，就是1个代表颜色的值
    label = nbrs.kneighbors(im_lab[:, :, 1:].reshape(-1, 2),
            return_distance=False).reshape(im_lab.shape[0], im_lab.shape[1])
    
    return im, label # im (1,32,32) label(32,32)


def prepare_jig(im, dataset, split, perms, mean, sd, eig_vals=None, eig_vecs=None):
    # im = HWC
    if "cifar" not in dataset:
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            if "imagenet" in dataset:
                target_size = cfg.TEST.IM_SIZE
            elif "cityscapes" in dataset:
                random_scale = np.power(2, -1 + 2 * np.random.uniform())
                target_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
            im = transforms.scale(target_size, im)
            im = transforms.random_crop(im, train_size, order="HWC")
        else:
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
    if split == "train":
        im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        if np.random.uniform() < cfg.TRAIN.GRAY_PERCENTAGE:
            # 1)图片的像素必须是整数，不是float；
            # 2)转换出来的灰度图是(32,32), 归一在0~1范围内，需要乘上255到0~255范围内
            im = color.rgb2gray(im.astype(np.uint8, copy=False)) * 255.0
            im = np.expand_dims(im, axis=2) # (HWC)=(32,32,1)
            im = np.tile(im, (1, 1, 3)) # (32,32,3),每个像素的3个通道都是相同的灰度值
    im = transforms.HWC2CHW(im)
    im = im / 255.0
    if "cifar" not in dataset:
        im = im[:, :, ::-1]  # RGB -> BGR
        # PCA jitter
        if split == "train":
            im = transforms.lighting(im, 0.1, eig_vals, eig_vecs)
    # Color normalization
    im = transforms.color_norm(im, mean, sd)
    # Random permute
    label = np.random.randint(len(perms))
    perm = perms[label]
    # Crop tiles
    psz = int(cfg.TRAIN.IM_SIZE / cfg.JIGSAW_GRID)  # Patch size
    tsz = int(psz * 0.76)  # Tile size; int(16 * 0.76) = 12
    tiles = np.zeros((cfg.JIGSAW_GRID ** 2, 3, tsz, tsz)).astype(np.float32) # (4,3,12,12)
    # CHW
    for i in range(cfg.JIGSAW_GRID):
        for j in range(cfg.JIGSAW_GRID):
            patch = im[:, psz * i:psz * (i+1), psz * j:psz * (j+1)]
            # Gap
            h = np.random.randint(psz - tsz + 1) # [0,5)
            w = np.random.randint(psz - tsz + 1)
            tile = patch[:, h:h+tsz, w:w+tsz]   # 从(16,16)中抽出一个tile
            # Normalization
            mu, sigma = np.mean(tile), np.std(tile)
            tile = tile - mu
            tile = tile / sigma
            tiles[perm[cfg.JIGSAW_GRID * i + j]] = tile
    return tiles, label # tiles(4,3,12,12)是按照label对应的排列排序的拼图块，预测


def prepare_im(im, dataset, split, mean, sd, is_cutout=False, eig_vals=None, eig_vecs=None):
    if "imagenet" in dataset:
        # Train and test setups differ
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            # Scale and aspect ratio then horizontal flip
            im = transforms.random_sized_crop(im=im, size=train_size, area_frac=0.08)
            im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        else:
            # Scale and center crop
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(train_size, im)
    elif "cityscapes" in dataset:
        train_size = cfg.TRAIN.IM_SIZE
        if split == "train":
            # Scale
            random_scale = np.power(2, -1 + 2 * np.random.uniform())
            random_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
            im = transforms.scale(random_size, im)
            # Crop
            im = transforms.random_crop(im, train_size, order="HWC")
            # Flip
            im = transforms.horizontal_flip(im=im, p=0.5, order="HWC")
        else:
            # Scale
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            # Crop
            im = transforms.center_crop(train_size, im)
    # above (HWC, RGB) below (CHW, RGB)
    im = transforms.HWC2CHW(im)
    im = im / 255.0
    if "cifar" not in dataset:  # ！！！当数据集为cifar时，归一用的mean和sd为rgb order，当为其他时，应该为BGR
        im = im[:, :, ::-1]  # RGB -> BGR
        # PCA jitter
        if split == "train":
            im = transforms.lighting(im, 0.1, eig_vals, eig_vecs)
    # Color normalization
    im = transforms.color_norm(im, mean, sd)    # 这里的mean 和sd是rgb for cifar，bgr for imagenet
    if "cifar" in dataset:
        if split == "train":
            im = transforms.horizontal_flip(im=im, p=0.5)
            im = transforms.random_crop(im=im, size=cfg.TRAIN.IM_SIZE, pad_size=4)  # Best after color_norm because of zero padding
            if is_cutout:
                im = transforms.cutout(im, length=cfg.TRAIN.CUTOUT_LENGTH)  
    return im


def prepare_seg(im, label, split, mean, sd, eig_vals=None, eig_vecs=None):
    if split == "train":
        train_size = cfg.TRAIN.IM_SIZE
        # Scale
        random_scale = np.power(2, -1 + 2 * np.random.uniform())
        random_size = int(max(train_size, cfg.TEST.IM_SIZE * random_scale))
        im = transforms.scale(random_size, im)
        label = transforms.scale(random_size, label, interpolation=cv2.INTER_NEAREST, dtype=np.int64)
        # Crop
        h, w = im.shape[:2]
        y = 0
        if h > train_size:
            y = np.random.randint(0, h - train_size)
        x = 0
        if w > train_size:
            x = np.random.randint(0, w - train_size)
        im = im[y : (y + train_size), x : (x + train_size), :]
        label = label[y : (y + train_size), x : (x + train_size)]
        # Flip
        if np.random.uniform() < 0.5:
            im = im[:, ::-1, :].copy()
            label = label[:, ::-1].copy()
    im = transforms.HWC2CHW(im)
    im = im / 255.0
    im = im[:, :, ::-1]  # RGB -> BGR
    # PCA jitter
    if split == "train":
        im = transforms.lighting(im, 0.1, eig_vals, eig_vecs)
    # Color normalization
    im = transforms.color_norm(im, mean, sd)
    if split != "train":
        # 1025 x 2049; cfg.TEST.IM_SIZE is not used here
        im = np.pad(im, ((0, 0), (0, 1), (0, 1)), "constant", constant_values=0)  # Best after color_norm because of zero padding
        label = np.pad(label, ((0, 1), (0, 1)), "constant", constant_values=255)
    return im, label
