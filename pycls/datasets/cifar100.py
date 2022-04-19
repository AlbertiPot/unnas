# modified from CIFAR10

"""CIFAR100 dataset."""

import os
import pickle
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg
from pycls.datasets.prepare import prepare_rot
from pycls.datasets.prepare import prepare_col
from pycls.datasets.prepare import prepare_jig
from pycls.datasets.prepare import prepare_im


logger = logging.get_logger(__name__)
folder = os.path.dirname(os.path.realpath(__file__))

# Per-channel mean and SD values in RGB order
_MEAN = [0.5071, 0.4867, 0.4408]
_SD = [0.2675, 0.2565, 0.2761]


class Cifar100(torch.utils.data.Dataset):
    """CIFAR-100 dataset."""

    def __init__(self, data_path, split, portion=None, side=None):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        splits = ["train", "test"]
        assert split in splits, "Split '{}' not supported for cifar".format(split)
        logger.info("Constructing CIFAR-100 {}...".format(split))
        self._data_path, self._split = data_path, split
        self._portion, self._side = portion, side
        
        # TODO: 测试if elif语句是不是符合cifar100
        if cfg.TASK == 'col':
            # Color centers in ab channels; numpy array; shape (313, 2)
            self._pts = np.load(os.path.join(folder, "files", "pts_in_hull.npy"))
            self._nbrs = NearestNeighbors(n_neighbors=1).fit(self._pts)
        elif cfg.TASK == 'jig':
            assert cfg.JIGSAW_GRID == 2
            assert cfg.MODEL.NUM_CLASSES == 24
            # Jigsaw permutations; numpy array; shape (24, 4)   分成2*2的grid，共4个patch 4!=24个排列，即24行，每行4个patch
            self._perms = np.load(os.path.join(folder, "files", "permutations_24.npy"))
        
        self._inputs, self._labels = self._load_data()

    def _load_data(self):
        """Loads data into memory."""
        logger.info("{} data path: {}".format(self._split, self._data_path))
        # Compute data batch names
        if self._split == "train":
            batch_names = "train"
        else:
            batch_names = "test"
        # Load data batches
        inputs, labels = [], []
        batch_path = os.path.join(self._data_path, batch_names)
        with open(batch_path, "rb") as f:
                data = pickle.load(f, encoding="bytes")
        inputs.append(data[b'data'])
        labels += data[b'fine_labels']
        
        # Combine and reshape the inputs
        inputs = np.vstack(inputs).astype(np.float32) # 这里将图片转化为(50000, 3072)，3072 = 1024R 1024G 1024B
        inputs = inputs.reshape((-1, 3, cfg.TRAIN.IM_SIZE, cfg.TRAIN.IM_SIZE))  # datasize*c*h*w
        
        # 对训练集再分固定的portion
        if self._portion:
            # CIFAR-100 data are random, so no need to shuffle
            pos = int(self._portion * len(inputs))
            if self._side == "l":
                return inputs[:pos], labels[:pos]
            else:  # self._side == "r"
                return inputs[pos:], labels[pos:]
        else:
            return inputs, labels

    def __getitem__(self, index):
        # TODO: 【检查以下】是不是适合cifar100，prepare中加上cifar100
        im, label = self._inputs[index, ...].copy(), self._labels[index]    # 将第index个数据copy出来
        im = transforms.CHW2HWC(im)  # CHW, RGB -> HWC, RGB
        if cfg.TASK == 'rot':
            im, label = prepare_rot(im,
                                    dataset="cifar100",
                                    split=self._split,
                                    mean=_MEAN,
                                    sd=_SD)
        elif cfg.TASK == 'col':
            im, label = prepare_col(im,
                                    dataset="cifar100",
                                    split=self._split,
                                    nbrs=self._nbrs,
                                    mean=_MEAN,
                                    sd=_SD)
        elif cfg.TASK == 'jig':
            im, label = prepare_jig(im,
                                    dataset="cifar100",
                                    split=self._split,
                                    perms=self._perms,
                                    mean=_MEAN,
                                    sd=_SD)
        else:
            im = prepare_im(im,
                            dataset="cifar100",
                            split=self._split,
                            mean=_MEAN,
                            sd=_SD)
        return im, label

    def __len__(self):
        return self._inputs.shape[0]
