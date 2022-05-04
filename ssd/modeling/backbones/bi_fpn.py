import torch.nn as nn
import torchvision
import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchsummary import summary
from typing import Tuple, List
from collections import OrderedDict


class BiFPNModel(nn.Module):
    def __init__(
            self,
            output_channels: List[int],
            image_channels: int,
            input_channels: List[int],
            output_feature_sizes: List[Tuple[int]],
        
        ):
        super().__init__()
        
        self.out_channels = output_channels
        self.input_channels = input_channels
        self.output_feature_shape = output_feature_sizes 
        self.image_channels = image_channels
        self.model = torchvision.models.resnet34(pretrained=True).to('cuda')
        
        self.layer5 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=1024,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),    
        ).to("cuda")
        self.layer6 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=1024,
                out_channels=2048,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),    
        ).to("cuda")
        
        features = torch.nn.ModuleList(self.model.children())[:-2]
        model_features = torch.nn.Sequential(*features) 
        
        model_features.add_module("8", self.layer5)
        model_features.add_module("9", self.layer6)
        
        self.model = model_features
        print(self.model)
        exit()
        self.body = create_feature_extractor(
            self.model, return_nodes={f'{k}': str(v)
                             for v, k in enumerate([i for i in range(4,10)])})
        
        self.fpn_channels = self.out_channels[0]
        
        # self.fpn = torchvision.ops.FeaturePyramidNetwork(
        #     self.input_channels, out_channels=self.fpn_channels)

    def fpn(self, x, out_features):
    #     # print(f"{[a.shape for a in out_features]=}") same as x - [torch.Size([1, 64, 32, 256]), torch.Size([1, 128, 16, 128]), torch.Size([1, 256, 8, 64]), torch.Size([1, 512, 4, 32]), torch.Size([1, 1024, 2, 16]), torch.Size([1, 2048, 1, 8])]
    #     # print(f"{self.fpn_channels=}") - 256
    #     # print(f"{self.input_channels=}") - [64, 128, 256, 512, 1024, 2048]
    #     # print(f"{self.out_channels=}") - [256, 256, 256, 256, 256, 256]
        exit()
        
        
    def forward(self, x):
        x = self.body(x)
        out_features = list(x.values())
        self.res_test(out_features)
        
        x = self.fpn(x, out_features)
        out_features = list(x.values())
        # print(f"{[a.shape for a in out_features]=}") - FPN - [torch.Size([1, 256, 32, 256]), torch.Size([1, 256, 16, 128]), torch.Size([1, 256, 8, 64]), torch.Size([1, 256, 4, 32]), torch.Size([1, 256, 2, 16]), torch.Size([1, 256, 1, 8])]
        self.fpn_test(out_features)
        
        return tuple(out_features)
        


    def res_test(self, out_features):
        for idx, feature in enumerate(out_features):
            out_channel = self.input_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert (
                feature.shape[1:] == expected_shape
            ), f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
            assert len(out_features) == len(
            self.output_feature_shape
                ), f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
                  
    def fpn_test(self, out_features):
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert (
                feature.shape[1:] == expected_shape
            ), f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
            assert len(out_features) == len(
            self.output_feature_shape
        ), f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"