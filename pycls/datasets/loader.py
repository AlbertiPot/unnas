#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

import os

import torch
from pycls.core.config import cfg
from pycls.datasets.cifar10 import Cifar10
from pycls.datasets.cifar100 import Cifar100
from pycls.datasets.imagenet import ImageNet
from pycls.datasets.imagenet22k import ImageNet22k
from pycls.datasets.cityscapes import Cityscapes
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


# Supported datasets
_DATASETS = {"cifar10": Cifar10, 
             "cifar100": Cifar100,
             "imagenet": ImageNet,
             "imagenet22k": ImageNet22k,
             "cityscapes": Cityscapes}

# Default data directory (/path/pycls/pycls/datasets/data)
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data") # dirname获取当前的绝对路径， 即.../your project/pycls/datasets/ join 本文件夹下的data

# Relative data paths to default data directory
_PATHS = {"cifar10": "cifar10", 
          "cifar100": "cifar100",
          "imagenet": "imagenet",
          "imagenet22k": "imagenet22k",
          "cityscapes": "cityscapes"}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last, portion=None, side=None):
    """Constructs the data loader for the given dataset."""
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    
    # Retrieve the data path for the dataset
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    # Construct the dataset
    dataset = _DATASETS[dataset_name](data_path, split, portion, side)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create collate_fn
    if cfg.TASK == "rot":
        def _collate_fn(batch):
            batch = torch.utils.data.dataloader.default_collate(batch)
            # print(batch[0],batch[0].shape)
            # print("Before collate ", batch[0].shape, batch[1].shape)
            assert(len(batch) == 2)
            b, r, c, h, w = batch[0].size()             # batch_sz, rotation四个角度，通道，h，w
            batch[0] = batch[0].view([b * r, c, h, w])  # 将bsz和旋转的4个角度相乘，即一个batch的数量是4倍的原bsz
            batch[1] = batch[1].view([b * r])           # 标签
            # print("After collate ", batch[0].shape, batch[1].shape)
            return batch
        collate_fn = _collate_fn
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,    # dataset中的prepare环节，由于涉及多线程，无法打断点验证
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
