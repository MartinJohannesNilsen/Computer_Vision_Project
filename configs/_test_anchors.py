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
    # aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    # aspect_ratios=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
    aspect_ratios=[[2], [2], [2], [2], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2,
)
