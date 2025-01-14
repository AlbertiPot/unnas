#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""NAS genotypes (adopted from DARTS)."""

from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


# NASNet ops
NASNET_OPS = [
    'skip_connect',
    'conv_3x1_1x3',
    'conv_7x1_1x7',
    'dil_conv_3x3',
    'avg_pool_3x3',
    'max_pool_3x3',
    'max_pool_5x5',
    'max_pool_7x7',
    'conv_1x1',
    'conv_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
]

# ENAS ops
ENAS_OPS = [
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'avg_pool_3x3',
    'max_pool_3x3',
]

# AmoebaNet ops
AMOEBA_OPS = [
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'avg_pool_3x3',
    'max_pool_3x3',
    'dil_sep_conv_3x3',
    'conv_7x1_1x7',
]

# NAO ops
NAO_OPS = [
    'skip_connect',
    'conv_1x1',
    'conv_3x3',
    'conv_3x1_1x3',
    'conv_7x1_1x7',
    'max_pool_2x2',
    'max_pool_3x3',
    'max_pool_5x5',
    'avg_pool_2x2',
    'avg_pool_3x3',
    'avg_pool_5x5',
]

# PNAS ops
PNAS_OPS = [
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
    'skip_connect',
    'avg_pool_3x3',
    'max_pool_3x3',
    'dil_conv_3x3',
]

# DARTS ops
DARTS_OPS = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]


NASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 1),
        ('sep_conv_7x7', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('skip_connect', 3),
        ('avg_pool_3x3', 2),
        ('sep_conv_3x3', 2),
        ('max_pool_3x3', 1),
    ],
    reduce_concat=[4, 5, 6],
)


PNASNet = Genotype(
    normal=[
        ('sep_conv_5x5', 0),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 1),
        ('max_pool_3x3', 1),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 4),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 1),
    ],
    normal_concat=[2, 3, 4, 5, 6],
    reduce=[
        ('sep_conv_5x5', 0),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 1),
        ('max_pool_3x3', 1),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 4),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 1),
    ],
    reduce_concat=[2, 3, 4, 5, 6],
)


AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 3),
        ('sep_conv_3x3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_7x7', 2),
        ('sep_conv_7x7', 0),
        ('avg_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('conv_7x1_1x7', 0),
        ('sep_conv_3x3', 5),
    ],
    reduce_concat=[3, 4, 6]
)

mytrain = Genotype(
    normal=[
        ('sep_conv_3x3', 1), 
        ('dil_conv_3x3', 0), 
        ('dil_conv_5x5', 0), 
        ('sep_conv_5x5', 1), 
        ('dil_conv_3x3', 0), 
        ('sep_conv_3x3', 1), 
        ('sep_conv_3x3', 0), 
        ('sep_conv_3x3', 1)], 
    
    normal_concat=range(2, 6), 
    
    reduce=[
        ('max_pool_3x3', 0), 
        ('max_pool_3x3', 1), 
        ('max_pool_3x3', 0), 
        ('sep_conv_3x3', 2), 
        ('max_pool_3x3', 0), 
        ('skip_connect', 2), 
        ('max_pool_3x3', 0), 
        ('dil_conv_3x3', 2)], 
    
    reduce_concat=range(2, 6))

DARTS_V1 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('avg_pool_3x3', 0)
    ],
    reduce_concat=[2, 3, 4, 5]
)


DARTS_V2 = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)

PDARTS = Genotype(
    normal=[
        ('skip_connect', 0),
        ('dil_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 4)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('avg_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0),
        ('dil_conv_3x3', 1),
        ('dil_conv_3x3', 1),
        ('dil_conv_5x5', 3)
    ],
    reduce_concat=range(2, 6)
)

PCDARTS_C10 = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 0),
        ('dil_conv_3x3', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 1),
        ('avg_pool_3x3', 0),
        ('dil_conv_3x3', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_5x5', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('sep_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 2)
    ],
    reduce_concat=range(2, 6)
)

PCDARTS_IN1K = Genotype(
    normal=[
        ('skip_connect', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0),
        ('skip_connect', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1),
        ('dil_conv_5x5', 4)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_3x3', 0),
        ('skip_connect', 1),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 1),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 3)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET_CLS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('skip_connect', 1),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 4),
        ('dil_conv_5x5', 3)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET_ROT = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 4),
        ('sep_conv_5x5', 2)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET_COL = Genotype(
    normal=[
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_3x3', 1),
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 3),
        ('max_pool_3x3', 0),
        ('sep_conv_3x3', 4)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET_JIG = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 1)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET22K_CLS = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('dil_conv_5x5', 2),
        ('max_pool_3x3', 0),
        ('dil_conv_5x5', 3),
        ('dil_conv_5x5', 2),
        ('dil_conv_5x5', 4),
        ('dil_conv_5x5', 3)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET22K_ROT = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('dil_conv_5x5', 2),
        ('sep_conv_5x5', 0),
        ('dil_conv_5x5', 3),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 4),
        ('sep_conv_3x3', 3)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET22K_COL = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('skip_connect', 1),
        ('dil_conv_5x5', 2),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 4),
        ('sep_conv_5x5', 1)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_IMAGENET22K_JIG = Genotype(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 4)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_5x5', 0),
        ('skip_connect', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_5x5', 0),
        ('sep_conv_5x5', 3),
        ('sep_conv_5x5', 0),
        ('sep_conv_5x5', 4)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_CITYSCAPES_SEG = Genotype(
    normal=[
        ('skip_connect', 0),
        ('sep_conv_5x5', 1),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('sep_conv_3x3', 0),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 4),
        ('sep_conv_5x5', 2)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_CITYSCAPES_ROT = Genotype(
    normal=[
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 3),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('max_pool_3x3', 0),
        ('sep_conv_5x5', 1),
        ('sep_conv_5x5', 2),
        ('sep_conv_5x5', 1),
        ('sep_conv_5x5', 3),
        ('dil_conv_5x5', 2),
        ('sep_conv_5x5', 2),
        ('sep_conv_5x5', 0)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_CITYSCAPES_COL = Genotype(
    normal=[
        ('dil_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('skip_connect', 0),
        ('sep_conv_5x5', 2),
        ('dil_conv_3x3', 3),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('sep_conv_3x3', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 1),
        ('avg_pool_3x3', 0),
        ('avg_pool_3x3', 1),
        ('skip_connect', 4)
    ],
    reduce_concat=range(2, 6)
)

UNNAS_CITYSCAPES_JIG = Genotype(
    normal=[
        ('dil_conv_5x5', 1),
        ('sep_conv_5x5', 0),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 2),
        ('sep_conv_3x3', 0),
        ('dil_conv_5x5', 1)
    ],
    normal_concat=range(2, 6),
    reduce=[
        ('avg_pool_3x3', 0),
        ('skip_connect', 1),
        ('dil_conv_5x5', 1),
        ('dil_conv_5x5', 2),
        ('dil_conv_5x5', 2),
        ('dil_conv_5x5', 0),
        ('dil_conv_5x5', 3),
        ('dil_conv_5x5', 2)
    ],
    reduce_concat=range(2, 6)
)

# rot
v1rotc100seed2      = Genotype(normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1rotc100seed5555   = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('skip_connect', 3), ('skip_connect', 2)], reduce_concat=range(2, 6))
v1rotc100seed7777   = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 1), ('dil_conv_3x3', 0)], reduce_concat=range(2, 6))
v1rotc100seed9999   = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_5x5', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

v1rotc10seed2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 3), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
v1rotc10seed5555 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 3), ('avg_pool_3x3', 1), ('sep_conv_3x3', 0)], reduce_concat=range(2, 6))
v1rotc10seed7777 = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1rotc10seed9999 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

# jig
v1jigc100seed2      = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 4), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
v1jigc100seed5555   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_3x3', 3), ('skip_connect', 1)], reduce_concat=range(2, 6))
v1jigc100seed7777   = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_3x3', 3), ('dil_conv_5x5', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
v1jigc100seed9999   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))

v1jigc10seed2 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 3), ('dil_conv_5x5', 2), ('dil_conv_5x5', 4), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
v1jigc10seed5555 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('skip_connect', 0), ('sep_conv_5x5', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
v1jigc10seed7777 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 3), ('skip_connect', 0), ('dil_conv_3x3', 4), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
v1jigc10seed9999 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('skip_connect', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 2), ('dil_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))

# col
v1colc100seed2      = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 3)], reduce_concat=range(2, 6))
v1colc100seed5555   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 3), ('max_pool_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 1)], reduce_concat=range(2, 6))
v1colc100seed7777   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))
v1colc100seed9999   = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 3), ('skip_connect', 0), ('skip_connect', 1), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))

v1colc10seed2 = Genotype(normal=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 4), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 1), ('skip_connect', 1), ('sep_conv_3x3', 3), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=range(2, 6))
v1colc10seed5555 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 1)], reduce_concat=range(2, 6))
v1colc10seed7777 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('skip_connect', 1), ('dil_conv_3x3', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
v1colc10seed9999 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 1), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
# cls_col
v1cls_colc100seed2      = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
v1cls_colc100seed5555   = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_colc100seed7777   = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_colc100seed9999   = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

v1cls_colc10seed2      = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
v1cls_colc10seed5555   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
v1cls_colc10seed7777   = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 3), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_colc10seed9999   = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_5x5', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

# cls_jig
v1cls_jigc100seed2      = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('dil_conv_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ('dil_conv_5x5', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
v1cls_jigc100seed5555   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 4), ('skip_connect', 2)], reduce_concat=range(2, 6))
v1cls_jigc100seed7777   = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 3), ('max_pool_3x3', 1), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
v1cls_jigc100seed9999   = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 4), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

v1cls_jigc10seed2      = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 1), ('sep_conv_3x3', 3), ('avg_pool_3x3', 1), ('sep_conv_3x3', 4), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
v1cls_jigc10seed5555   = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('dil_conv_5x5', 4), ('dil_conv_3x3', 1)], reduce_concat=range(2, 6))
v1cls_jigc10seed7777   = Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 4), ('sep_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('avg_pool_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
v1cls_jigc10seed9999   = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_5x5', 4), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

# cls_rot
v1cls_rotc100seed2      = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_rotc100seed5555   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_rotc100seed7777   = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_rotc100seed9999   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

v1cls_rotc10seed2      = Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))
v1cls_rotc10seed5555   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_3x3', 2), ('dil_conv_5x5', 3), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_rotc10seed7777   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))
v1cls_rotc10seed9999   = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 1), ('dil_conv_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('dil_conv_5x5', 1), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

# Supported genotypes
GENOTYPES = {
    'nas': NASNet,
    'pnas': PNASNet,
    'amoeba': AmoebaNet,
    'darts_v1': DARTS_V1,
    'darts_v2': DARTS_V2,
    'pdarts': PDARTS,
    'pcdarts_c10': PCDARTS_C10,
    'pcdarts_in1k': PCDARTS_IN1K,
    'unnas_imagenet_cls': UNNAS_IMAGENET_CLS,
    'unnas_imagenet_rot': UNNAS_IMAGENET_ROT,
    'unnas_imagenet_col': UNNAS_IMAGENET_COL,
    'unnas_imagenet_jig': UNNAS_IMAGENET_JIG,
    'unnas_imagenet22k_cls': UNNAS_IMAGENET22K_CLS,
    'unnas_imagenet22k_rot': UNNAS_IMAGENET22K_ROT,
    'unnas_imagenet22k_col': UNNAS_IMAGENET22K_COL,
    'unnas_imagenet22k_jig': UNNAS_IMAGENET22K_JIG,
    'unnas_cityscapes_seg': UNNAS_CITYSCAPES_SEG,
    'unnas_cityscapes_rot': UNNAS_CITYSCAPES_ROT,
    'unnas_cityscapes_col': UNNAS_CITYSCAPES_COL,
    'unnas_cityscapes_jig': UNNAS_CITYSCAPES_JIG,
    'custom': None,
    'v1rotc100seed2':v1rotc100seed2,
    'v1rotc100seed5555':v1rotc100seed5555,
    'v1rotc100seed7777':v1rotc100seed7777,
    'v1rotc100seed9999':v1rotc100seed9999,
    'v1jigc100seed2':v1jigc100seed2,
    'v1jigc100seed5555':v1jigc100seed5555,
    'v1jigc100seed7777':v1jigc100seed7777,
    'v1jigc100seed9999':v1jigc100seed9999,
    'v1colc100seed2':v1colc100seed2,
    'v1colc100seed5555':v1colc100seed5555,
    'v1colc100seed7777':v1colc100seed7777,
    'v1colc100seed9999':v1colc100seed9999,
    'v1cls_colc100seed2':v1cls_colc100seed2,
    'v1cls_colc100seed5555':v1cls_colc100seed5555,
    'v1cls_colc100seed7777':v1cls_colc100seed7777,
    'v1cls_colc100seed9999':v1cls_colc100seed9999,
    'v1cls_colc10seed2':v1cls_colc10seed2,
    'v1cls_colc10seed5555':v1cls_colc10seed5555,
    'v1cls_colc10seed7777':v1cls_colc10seed7777,
    'v1cls_colc10seed9999':v1cls_colc10seed9999,
    'v1cls_jigc100seed2':v1cls_jigc100seed2,
    'v1cls_jigc100seed5555':v1cls_jigc100seed5555,
    'v1cls_jigc100seed7777':v1cls_jigc100seed7777,
    'v1cls_jigc100seed9999':v1cls_jigc100seed9999,
    'v1cls_jigc10seed2':v1cls_jigc10seed2,
    'v1cls_jigc10seed5555':v1cls_jigc10seed5555,
    'v1cls_jigc10seed7777':v1cls_jigc10seed7777,
    'v1cls_jigc10seed9999':v1cls_jigc10seed9999,
    'v1cls_rotc100seed2':v1cls_rotc100seed2,
    'v1cls_rotc100seed5555':v1cls_rotc100seed5555,
    'v1cls_rotc100seed7777':v1cls_rotc100seed7777,
    'v1cls_rotc100seed9999':v1cls_rotc100seed9999,
    'v1cls_rotc10seed2':v1cls_rotc10seed2,
    'v1cls_rotc10seed5555':v1cls_rotc10seed5555,
    'v1cls_rotc10seed7777':v1cls_rotc10seed7777,
    'v1cls_rotc10seed9999':v1cls_rotc10seed9999,
    'v1jigc10seed2':v1jigc10seed2,
    'v1jigc10seed5555':v1jigc10seed5555,
    'v1jigc10seed7777':v1jigc10seed7777,
    'v1jigc10seed9999':v1jigc10seed9999,
    'v1colc10seed2':v1colc10seed2,
    'v1colc10seed5555':v1colc10seed5555,
    'v1colc10seed7777':v1colc10seed7777,
    'v1colc10seed9999':v1colc10seed9999,
    'v1rotc10seed2':v1rotc10seed2,
    'v1rotc10seed5555':v1rotc10seed5555,
    'v1rotc10seed7777':v1rotc10seed7777,
    'v1rotc10seed9999':v1rotc10seed9999
}
