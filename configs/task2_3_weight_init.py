# Inherit configs from the default ssd300
import torchvision
import torch
from ssd.data import TDT4265Dataset
from ssd.modeling import RetinaNet
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor,
    RandomHorizontalFlip,
    RandomSampleCrop,
    Normalize,
    Resize,
    GroundTruthBoxesToAnchors,
)
from .utils import get_dataset_dir
from .task2_3_deeper import (
    train_cpu_transform,
    val_cpu_transform,
    model,
    gpu_transform,
    data_train,
    data_val,
    train,
    anchors,
    optimizer,
    schedulers,
    loss_objective,
    label_map,
    loss_objective,
    backbone,
)


model.use_improved_weight_init = True
