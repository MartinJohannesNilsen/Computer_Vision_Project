import torchvision
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import ToTensor, Normalize, Resize, GroundTruthBoxesToAnchors
from .utils import get_dataset_dir, get_output_dir
from .task2_1 import (
    model,
    gpu_transform,
    data_train,
    data_val,
    train,
    optimizer,
    schedulers,
    backbone,
    train_cpu_transform,
    val_cpu_transform,
    loss_objective,
    label_map,
)
import sys
import re

# Optimize anchorboxes
"""
Iterations:
 0. No changes to anchor boxes
 1. Decrease of smallest min_sizes for optimizing detection of smaller objects
 2. Increase of largest min_sizes for optimizing detection of larger objects
 3. Both decrease the smallest and increase the largest for testing purposes
"""

# Find iteration based on file name
expression = re.compile("\d+(?=\.\w)")
matches = expression.findall(__file__)
if not matches:
    print(
        "Resolve file name to format 'task2_2_iterX.py', where X is in the list of augmentation iterations!"
    )
    sys.exit(1)
ITERATION = int(matches[0])

if ITERATION == 0:
    anchors = L(AnchorBoxes)(
        feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
        strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
        min_sizes=[
            [16, 16],
            [32, 32],
            [48, 48],
            [64, 64],
            [86, 86],
            [128, 128],
            [128, 400],
        ],
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        image_shape="${train.imshape}",
        scale_center_variance=0.1,
        scale_size_variance=0.2,
    )
elif ITERATION == 1:
    anchors = L(AnchorBoxes)(
        feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
        strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
        # min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]], # Original
        min_sizes=[
            [12, 12],
            [26, 26],
            [48, 48],
            [64, 64],
            [86, 86],
            [128, 128],
            [128, 400],
        ],
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        image_shape="${train.imshape}",
        scale_center_variance=0.1,
        scale_size_variance=0.2,
    )
elif ITERATION == 2:
    anchors = L(AnchorBoxes)(
        feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
        strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
        # min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]], # Original
        min_sizes=[
            [12, 12],
            [26, 26],
            [48, 48],
            [64, 64],
            [92, 92],
            [148, 148],
            [148, 432],
        ],
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        image_shape="${train.imshape}",
        scale_center_variance=0.1,
        scale_size_variance=0.2,
    )
elif ITERATION == 3:
    anchors = L(AnchorBoxes)(
        feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
        strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
        # min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]], # Original
        min_sizes=[
            [8, 8],
            [24, 24],
            [48, 48],
            [64, 64],
            [94, 94],
            [172, 172],
            [172, 520],
        ],
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        image_shape="${train.imshape}",
        scale_center_variance=0.1,
        scale_size_variance=0.2,
    )
