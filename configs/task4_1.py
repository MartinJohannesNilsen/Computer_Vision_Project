# Inherit configs from the default ssd300
import torchvision
import torch
from ssd.data import TDT4265Dataset
from ssd.modeling import backbones
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir
from .task2_1 import train_cpu_transform, val_cpu_transform,model, gpu_transform,data_train, data_val, train, anchors, optimizer, schedulers, model, loss_objective, label_map


feature_extractor = L(backbones.BiFPNModel)(
    input_channels=[64, 128, 256, 512, 256, 256],
    output_channels=[256, 256, 256, 256, 256, 256],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)
model.feature_extractor = feature_extractor
