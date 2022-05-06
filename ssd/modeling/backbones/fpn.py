import torch.nn as nn
import torchvision
import torch
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)
from torchsummary import summary
from typing import Tuple, List
from collections import OrderedDict


class FPNModel(nn.Module):
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
        self.model = torchvision.models.resnet34(pretrained=True).to("cuda" if torch.cuda.is_available() else "cpu")

        self.layer5 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_channels[-3],
                out_channels=self.input_channels[-3],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=self.input_channels[-3],
                out_channels=self.input_channels[-2],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.layer6 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_channels[-2],
                out_channels=self.input_channels[-2],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=self.input_channels[-2],
                out_channels=self.input_channels[-1],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        features = torch.nn.ModuleList(self.model.children())[:-2]
        model_features = torch.nn.Sequential(*features)

        model_features.add_module("8", self.layer5)
        model_features.add_module("9", self.layer6)

        self.model = model_features
        self.body = create_feature_extractor(
            self.model,
            return_nodes={
                f"{k}": str(v) for v, k in enumerate([i for i in range(4, 10)])
            },
        )

        self.fpn_channels = self.out_channels[0]

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            self.input_channels, out_channels=self.fpn_channels
        )

    def forward(self, x):
        x = self.body(x)
        out_features = list(x.values())
        self.res_test(out_features)

        x = self.fpn(x)
        out_features = list(x.values())

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
