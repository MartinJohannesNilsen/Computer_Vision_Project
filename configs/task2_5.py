# Inherit configs from the default ssd300
import torchvision
import torch
from ssd.data import TDT4265Dataset
from ssd.modeling import RetinaNet
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir
from .task2_1 import train, optimizer, schedulers, loss_objective, model, backbone, data_train, data_val, train_cpu_transform, val_cpu_transform, gpu_transform, label_map, anchors

data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")
