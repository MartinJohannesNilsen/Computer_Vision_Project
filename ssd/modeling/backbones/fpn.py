import torch.nn as nn
import torchvision
import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchsummary import summary
from typing import Tuple, List
from collections import OrderedDict

LOWEST_BACKBONE_LVL = 2   # E.g., "conv2"-like level
HIGHEST_BACKBONE_LVL = 5  # E.g., "conv5"-like level

class FPNModel(nn.Module):
    def __init__(
            self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]],
        ):
        super().__init__()
        
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes 
        self.image_channels = image_channels
        self.model = torchvision.models.resnet34(pretrained=True).to('cuda')
        self.layers = {'layer1': 'layer1','layer2': 'layer2','layer3': 'layer3','layer4': 'layer4',}
        self.body = create_feature_extractor(
            self.model, return_nodes=self.layers)
        
    
    def forward(self, x):
        out_features = self.body(x).values()
        
        
        
        
        
        
        
        '''
        x = OrderedDict()
        for idx, feat in enumerate(out_features):
            x[f"feat{idx}"] = feat
        m = torchvision.ops.FeaturePyramidNetwork(in_channels,256)
        m = m.to("cuda")
        
        output = m(x)
        
        out_features = list(output.values())
        '''
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
        return tuple(list(output.values()))

