# Inherit configs from the default ssd300
from glob import glob
import os
import torchvision
import torch
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    Resize, ToTensor, Normalize, GroundTruthBoxesToAnchors,
    RandomSampleCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
)
from .ssd300 import train, anchors, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective
from .utils import get_dataset_dir

"""
Possible transforms, which all of them have been added in the ssd.data.transforms file.
The methods taken from Pytorch own library have been implemented with wrappers.
- RandomSampleCrop
- RandomHorizontalFlip
- RandomRotation
- ColorJitter
- GrayScale / RandomGrayscale
- RandomAffine
- RandomEqualize
- RandomAutocontrast
"""
augmentation_transforms = [
    # L(RandomSampleCrop)(),
    L(ToTensor)(),  # Convert to tensor
    # L(RandomHorizontalFlip)(p=1),
    # L(RandomRotation)(rotation=3),
    L(ColorJitter)(brightness=0, contrast=0, saturation=0, hue=0), # All defaults to 0
]

# Keep all the other settings!
train.imshape = (128, 1024)
train.image_channels = 3
model.num_classes = 8 + 1  # Add 1 for background class
transforms = [
    L(Resize)(imshape="${train.imshape}"),  # Reassure all images are correct size
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),  # Draw boxes
]
for elem in reversed(augmentation_transforms):
    transforms.insert(0, elem)  # Insert all augmentation transforms after ToTensor(), before resize and gt boxes, in correct order
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=transforms)
val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])


data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"))

data_val.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${val_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"))
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
