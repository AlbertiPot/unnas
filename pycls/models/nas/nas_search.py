#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from pycls.models.common import Preprocess
from pycls.models.common import Classifier
from pycls.models.nas.operations import *
from pycls.models.nas.genotypes import DARTS_OPS as PRIMITIVES
from pycls.models.nas.genotypes import Genotype
from pycls.core.config import cfg
import pycls.core.logging as logging

logger = logging.get_logger(__name__)

# 两个节点直接的全部连接
class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:                # PRIMITIVES存DARTs的8个candidates operation
            op = OPS[primitive](C, stride, False)   # OPS存算子的对象
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False)) # 池化后加个bn
            
            self._ops.append(op)    # 将两个node之间的全部算子存起来

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))  # weights是连边的α


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        # 输入两个节点的处理
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)   # 如果前一个cell是reductioncell，对前前个处理，空间解析度减半，并且用卷积处理到指定的通道数
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):    # 遍历中间4个节点
            for j in range(2+i):        # 遍历每个节点与之前节点的连接
                stride = 2 if reduction and j < 2 else 1    # reduction只在前两个节点处
                op = MixedOp(C, stride)
                self._ops.append(op)                        # 将节点的全部nodes之间的连接存起来

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)   # 对输入两个节点的处理
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1) # cell内的全部中间节点的feature，仅仅保留最后的multiplier个feature送给output节点


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        if 'cifar' in cfg.TRAIN.DATASET:
            logger.info('Using CIFAR10 stem')
            C_curr = stem_multiplier*C  # 输出channel
            # stem for cifar：处理cifar10数据集图片
            self.stem = nn.Sequential(
                nn.Conv2d(cfg.MODEL.INPUT_CHANNELS, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            reduction_prev = False
        else:
            logger.info('Using ImageNet stem')
            C_curr = C
            self.stem0 = nn.Sequential(
                nn.Conv2d(cfg.MODEL.INPUT_CHANNELS, C // 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            reduction_prev = True

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C # 前两个是前两个cell输出的channel，后一个是当前cell的输出channel
        self.cells = nn.ModuleList()
        reduction_layers = [layers//3] if cfg.TASK == 'seg' else [layers//3, 2*layers//3]   # 分割仅仅在第1/3层是reduction层

        for i in range(layers):
            if i in reduction_layers:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.classifier = Classifier(C_prev, num_classes)

        self._initialize_alphas()

    def forward(self, input):
        input = Preprocess(input)
        if 'cifar' in cfg.TRAIN.DATASET:
            s0 = s1 = self.stem(input)
        else:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1) # shape(14,8)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        logits = self.classifier(s1, input.shape[2:])
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2+i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops))
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            # 遍历中间节点，遍历其与之前节点的连边
            for i in range(self._steps):
                end = start + n         # 每个中间节点与之前全部节点的连边：第一个节点与前两个输入节点的连边index是0，1，第二个是2，3，4
                W = weights[start:end].copy()
                
                # edges[]存的是每个节点与之前节点的连边：0表示第一个输入节点，1表示第二个输入节点，2表示第一个中间节点
                # eg: 第一个中间节点，与输入0和输入1连接（edges[0,1]）
                # 首先保留每个连边中权重最大的算子，然后根据这个算子的权重（负值），对所有与之前节点的连接排序，最后仅保留权重最大的前两个
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                
                # 遍历权重中每个节点中的两个连边，j的形式是前序节点的序号
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):  # k:遍历每个连接中不同算子的α
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:    # 找α权值最高的
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))    # 将最高的保留下的算子名称，以及连接的前序节点号j存入list
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2+self._steps-self._multiplier, self._steps+2)   # multiplier指定每个cell输出倒数multiplier个feature
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


class NAS_Search(nn.Module):
    """NAS net wrapper (delegates to nets from DARTS)."""

    def __init__(self):
        assert cfg.TRAIN.DATASET in ['cifar10', 'imagenet', 'imagenet22k', 'cityscapes'], \
            'Training on {} is not supported'.format(cfg.TRAIN.DATASET)
        super(NAS_Search, self).__init__()
        logger.info('Constructing NAS_Search: {}'.format(cfg.NAS))
        
        # 本类包含了模型 以及模型任务所用的loss
        if cfg.TASK == 'seg':
            criterion = nn.CrossEntropyLoss(ignore_index=255).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()
        
        self.net_ = Network(
            C=cfg.NAS.WIDTH,
            num_classes=cfg.MODEL.NUM_CLASSES,
            layers=cfg.NAS.DEPTH,
            criterion=criterion
        )

    def _loss(self, input, target):
        return self.net_._loss(input, target)

    def arch_parameters(self):
        return self.net_.arch_parameters()

    def genotype(self):
        return self.net_.genotype()

    def forward(self, x):
        return self.net_.forward(x)
