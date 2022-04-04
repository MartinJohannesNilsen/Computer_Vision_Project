import torch.nn as nn
import torchvision
import torch

class FPNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fpn = nn.ModuleList(list(torchvision.models.resnet18(pretrained=True).children())[:-2])
        self.out_channels = [128, 256, 128, 128, 64, 64]
        self.image_channels = 3
        
        self.extras = nn.ModuleList([
#             nn.Sequential(
#                 torch.nn.Conv2d(
#                     in_channels=self.image_channels,
#                     out_channels=32,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                 ),
#                 torch.nn.ReLU(),
#                 torch.nn.MaxPool2d(kernel_size=2, stride=2),
#                 torch.nn.Conv2d(
#                     in_channels=32,
#                     out_channels=64,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                 ),
#                 torch.nn.ReLU(),
#                 # torch.nn.MaxPool2d(kernel_size=2, stride=2),
#                 torch.nn.Conv2d(
#                     in_channels=64,
#                     out_channels=64,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                 ),
#                 torch.nn.ReLU(),
#                 torch.nn.Conv2d(
#                     in_channels=64,
#                     out_channels=self.out_channels[0],
#                     kernel_size=3,
#                     stride=2,
#                     padding=1,
#                 ),
#                 torch.nn.ReLU(),
#             ),
            nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.out_channels[0],
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=self.out_channels[1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(),
            ),
            nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.out_channels[1],
                    out_channels=256,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=256,
                    out_channels=self.out_channels[2],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(),
            ),
            nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.out_channels[2],
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=self.out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(),
            ),
            nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.out_channels[3],
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=self.out_channels[4],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(),
            ),
            nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=self.out_channels[4],
                    out_channels=128,
                    kernel_size=2,  # Changed from 3 (originally) to 2
                    stride=1,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=self.out_channels[5],
                    kernel_size=2,  # Changed from 3 (originally) to 2
                    stride=2, # Changed from 1 to 2
                    padding=0,
                ),
                torch.nn.ReLU(),
            )
        ])
        self.init_parameters()


    def init_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def init_from_pretrain(self, state_dict):
        self.fpn.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        
        print("START FPN")
        for layer in self.fpn:
            x = layer(x)
        print("END FPN")
        print(x.shape)
        print("START EXTRA")

        for extra in self.extras:
            x = extra(x)
            features.append(x)
        print("END EXTRA")

        for f in features:
            print(f.shape)
            
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

