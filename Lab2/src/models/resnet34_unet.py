import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    """
    A ResNet-34 encoder, modified so that:
     - the first 7x7 conv uses stride=1 (so we keep full input spatial size at x0),
     - then a 3x3 stride-2 maxpool produces x1 at 1/2 size,
     - layers 1-4 follow ResNet-34's block counts [3,4,6,3] with down-sampling at the start of layers 2–4.
    We return five feature maps for skip-connections at scales:
      x0:    [B,  64, H,   W  ]   ← after conv1
      x2:    [B,  64, H/2, W/2]   ← after layer1
      x3:    [B, 128, H/4, W/4]   ← after layer2
      x4:    [B, 256, H/8, W/8]   ← after layer3
      x5:    [B, 512, H/16,W/16]  ← after layer4
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.inplanes = 64
        # instead of stride=2, use stride=1 here
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # first downsample
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-34 layers: blocks per layer: [3,4,6,3]
        self.layer1 = self._make_layer(64,  3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_channels, blocks, stride):
        """Create one layer of BasicBlocks, with optional downsampling."""
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        # first block may downsample
        layers.append(BasicBlock(self.inplanes, out_channels, stride, downsample))
        self.inplanes = out_channels
        # remaining blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # [B,64, H,   W  ]
        x1 = self.maxpool(x0)                    # [B,64, H/2, W/2]
        x2 = self.layer1(x1)                     # [B,64, H/2, W/2]
        x3 = self.layer2(x2)                     # [B,128,H/4, W/4]
        x4 = self.layer3(x3)                     # [B,256,H/8, W/8]
        x5 = self.layer4(x4)                     # [B,512,H/16,W/16]
        return [x0, x2, x3, x4, x5]


class UpBlock(nn.Module):
    """Upsampling + double-conv for decoder."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # learnable upsampling
        self.up = nn.ConvTranspose2d(in_channels, out_channels,
                                     kernel_size=2, stride=2)
        # followed by two 3×3 convs (with BN+ReLU)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # if due to rounding we miss the exact size, resize
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        # channel-wise concat
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34_UNet(nn.Module):
    """
    Combines the above ResNet-34 encoder with a U-Net decoder, using 4 UpBlocks to
    return a segmentation map of the same spatial size as the input.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)

        # decoder: note channel counts match the encoder skips
        self.up4 = UpBlock(512, 256, 256)  # 1/16 → 1/8
        self.up3 = UpBlock(256, 128, 128)  # 1/8  → 1/4
        self.up2 = UpBlock(128, 64,  64)   # 1/4  → 1/2
        self.up1 = UpBlock(64,  64,  64)   # 1/2  →   1

        # final 1×1 conv to map to the desired number of classes
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # collect encoder features
        enc_feats = self.encoder(x)
        # bottom
        x = enc_feats[-1]        # [B,512,H/16,W/16]
        # 4 up-blocks, each halves the feature-map channels and doubles spatial size
        x = self.up4(x, enc_feats[-2])  # +skip @ H/8
        x = self.up3(x, enc_feats[-3])  # +skip @ H/4
        x = self.up2(x, enc_feats[-4])  # +skip @ H/2
        x = self.up1(x, enc_feats[-5])  # +skip @ H
        # final conv
        x = self.final(x)
        return torch.sigmoid(x)  # for binary mask in [0,1]

