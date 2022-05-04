import torch
import torchvision
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors)
from .utils import get_dataset_dir, get_output_dir
from .task2_1 import model, gpu_transform, data_train, data_val, train, optimizer, schedulers, backbone, train_cpu_transform, val_cpu_transform, loss_objective, label_map, anchors

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = L(model)
print(type(model))
train["epochs"] = 150
