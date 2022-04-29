# Inherit configs from the default ssd300
import torchvision
import torch
from ssd.data import TDT4265Dataset
from ssd.modeling import backbones, FocalLoss
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, RandomHorizontalFlip, RandomSampleCrop, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir
from .task2_3_fpn import train_cpu_transform, val_cpu_transform,model, gpu_transform,data_train, data_val, train, anchors, optimizer, schedulers, model, loss_objective, label_map, feature_extractor, backbone

loss_objective = L(FocalLoss)(anchors = "${model.anchors}", alpha = torch.as_tensor([0.01,*[1.0 for i in range (model.num_classes-1)]]).to("cuda"), gamma = 2.0,num_classes = model.num_classes)

feature_extractor = L(backbones.FPNModel)(
    input_channels=[64, 128, 256, 512, 1024, 2048],
    output_channels=[256, 256, 256, 256, 256, 256],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
)

# model.loss_objective = loss_objective

# We struggled with NaN values
model.use_improved_weight_init = True

