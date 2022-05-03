# Inherit configs from the default ssd300
import sys
import torchvision
from ssd.data import TDT4265Dataset
from ssd.modeling import RetinaNet
from tops.config import LazyCall as L
from ssd.data.transforms import (GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir
from .task2_3_weight_init import train, optimizer, schedulers, loss_objective, model, backbone, data_train, data_val, label_map, anchors
from .task2_2_iter6 import train_cpu_transform, val_cpu_transform, gpu_transform, transforms
from .task2_4_5 import anchors

# Train for a long time
train["epochs"] = 150
EXTEND = True

# Need to send in new anchors
backbone.output_feature_sizes = "${anchors.feature_sizes}"
loss_objective.anchors = "${anchors}"
model.feature_extractor = "${backbone}"
model.anchors = "${anchors}"
model.loss_objective = "${loss_objective}"
train.imshape = (128, 1024)
train.image_channels = 3
model.num_classes = 8 + 1  # Add 1 for background class
transforms = transforms[:-1]
transforms.append(L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5))
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=transforms)

# Augmentations
data_train.dataset.transform = "${train_cpu_transform}"
data_val.dataset.transform = "${val_cpu_transform}"
data_val.gpu_transform = gpu_transform
data_train.gpu_transform = gpu_transform

# Dataset expansion
if EXTEND:
    data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
    data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
    data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
    data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")
