import torch.nn as nn
import torchvision
import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torchsummary import summary
from typing import Tuple, List

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
        resnet = torchvision.models.resnet34(pretrained=True)
        self.body = create_feature_extractor(resnet, 
                                             return_nodes={
                                                 f'layer{k}': str(v) for v, k in enumerate([1, 2, 3, 4])})
        
        self.extra = nn.ModuleList([
            torch.nn.Sequential(
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
                    out_channels=256,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=2,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                torch.nn.ReLU(),
            )
        ])

        
    def init_from_pretrain(self, state_dict):
        self.resnet.load_state_dict(state_dict)
    
    def forward(self, x):
        a = self.body(x)
        out_features=list(a.values())
        x = out_features[-1]
#         print("START EXTRA")
        for layer in self.extra:
            x = layer(x)
            out_features.append(x)
#         print("END EXTRA")
#         print("START OUTFEATURES")
#         for f in out_features:
#             print(f.shape)
#         print("END OUTFEATURES")
            
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
        return tuple(out_features)

