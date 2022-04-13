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
from .task2_3_focal_loss import train_cpu_transform, val_cpu_transform,model, gpu_transform,data_train, data_val, train, anchors, optimizer, schedulers, model, loss_objective, label_map, feature_extractor, loss_objective


model = L(RetinaNet) (
    feature_extractor="${feature_extractor}",
    anchors="${anchors}",
    loss_objective= "${loss_objective) ",
    num_classes=8 + 1, # Add 1 for background
    anchor_prob_initialization=False
)

