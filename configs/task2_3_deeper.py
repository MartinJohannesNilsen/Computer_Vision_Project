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
from .task2_3_focal_loss import train_cpu_transform, val_cpu_transform, model, gpu_transform, data_train, data_val, train, optimizer, schedulers, loss_objective, label_map, loss_objective, anchors, backbone
from ssd.modeling import AnchorBoxes


model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    use_improved_weight_init=False
)

anchors.aspect_ratios=[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3],