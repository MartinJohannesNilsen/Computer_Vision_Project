# Inherit configs from the default ssd300
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

"""
Iterations:
 0. None
 1. RandomSampleCrop - crop
 2. RandomHorizontalFlip - flip
 3. RandomRotation - rotate
 4. RandomColorJitter/RandomGrayScale - change colors
 5. RandomAdjustSharpness - sharpen/blur
 6. Combination of RandomSampleCrop, RandomHorizontalFlip and RandomAdjustSharpness
 7. Combination of RandomHorizontalFlip and RandomAdjustSharpness
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

# Select iteration
if ITERATION == 0:
    transforms = [
        L(ToTensor)(),
    ]
elif ITERATION == 1:
    transforms = [
        L(RandomSampleCrop)(),
        L(ToTensor)(),
    ]
elif ITERATION == 2:
    transforms = [
        L(ToTensor)(),
        L(RandomHorizontalFlip)(),
    ]
elif ITERATION == 3:
    transforms = [
        L(ToTensor)(),
        L(RandomRotation)(rotation=3),
    ]
elif ITERATION == 4:
    transforms = [
        L(ToTensor)(),
        L(RandomColorJitter)(
            brightness=0, contrast=0.5, saturation=0.5, hue=0.5
        ),  # All defaults to 0
        L(RandomGrayscale)(p=0.5),
    ]
elif ITERATION == 5:
    transforms = [
        L(ToTensor)(),
        L(RandomAdjustSharpness)(
            sharpness_factor=0, p=0.25
        ),  # sf = 0, 1, 2 (default 1 for no change, 0 blur and 2 sharpen)
        L(RandomAdjustSharpness)(
            sharpness_factor=1.5, p=0.25
        ),  # sf = 0, 1, 2 (default 1 for no change, 0 blur and 2 sharpen)
    ]
elif ITERATION == 6:
    transforms = [
        L(RandomSampleCrop)(),
        L(ToTensor)(),
        L(RandomHorizontalFlip)(),
        L(RandomAdjustSharpness)(sharpness_factor=0, p=0.25),
        L(RandomAdjustSharpness)(sharpness_factor=1.5, p=0.25),
    ]
elif ITERATION == 7:
    transforms = [
        L(ToTensor)(),
        L(RandomHorizontalFlip)(),
        L(RandomAdjustSharpness)(sharpness_factor=0, p=0.25),
        L(RandomAdjustSharpness)(sharpness_factor=1.5, p=0.25),
    ]

pprint(transforms)
transforms.append(L(Resize)(imshape="${train.imshape}"))
transforms.append(L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5))
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=transforms)
