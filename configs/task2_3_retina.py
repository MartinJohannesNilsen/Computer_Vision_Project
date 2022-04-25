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
from .task2_3_focal_loss import train_cpu_transform, val_cpu_transform,model, gpu_transform,data_train, data_val, train, optimizer, schedulers, model, loss_objective, label_map, feature_extractor, loss_objective
from ssd.modeling import AnchorBoxes


anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes= [[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

model = L(RetinaNet)(
    feature_extractor="${feature_extractor}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1, # Add 1 for background
    use_improved_weight_init=False
)

