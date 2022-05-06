from pprint import pprint
import re
import sys
from glob import glob
import torchvision
import torch
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    Resize,
    ToTensor,
    Normalize,
    GroundTruthBoxesToAnchors,
    RandomSampleCrop,
    RandomHorizontalFlip,
    RandomRotation,
    RandomColorJitter,
    RandomGrayscale,
    RandomAdjustSharpness,
)
from .task2_1 import (
    model,
    data_train,
    data_val,
    train,
    optimizer,
    schedulers,
    backbone,
    anchors,
    loss_objective,
    label_map,
    gpu_transform,
    val_cpu_transform,
)
from .utils import get_dataset_dir

transforms = [
    # L(RandomSampleCrop)(),
    L(ToTensor)(),  # Convert to tensor
    # L(RandomHorizontalFlip)(p=1),
    # L(RandomRotation)(rotation=3),
    # L(RandomColorJitter)(brightness=0, contrast=0.5, saturation=0.5, hue=0.5),  # All defaults to 0
    L(RandomGrayscale)(p=1),
    # L(RandomAdjustSharpness)(sharpness_factor=0, p=1),  # sf = 0, 1, 2 (default 1 for no change, 0 blur and 2 sharpen)
    # L(RandomAdjustSharpness)(sharpness_factor=1.5, p=1),  # sf = 0, 1, 2 (default 1 for no change, 0 blur and 2 sharpen)
]
transforms.append(L(Resize)(imshape="${train.imshape}"))
transforms.append(L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5))
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=transforms)
