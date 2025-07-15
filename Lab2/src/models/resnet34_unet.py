import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet34_UNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        resnet = resnet34(weights=None)

        # Encoder
        self.inconv = nn.Sequential(
            resnet.conv1,  # 64
            resnet.bn1,
            resnet.relu,
        )  # Output: 64, 112x112 if input is 224x224
        self.pool = resnet.maxpool     # 64, 56x56
        self.enc1 = resnet.layer1      # 64, 56x56
        self.enc2 = resnet.layer2      # 128, 28x28
        self.enc3 = resnet.layer3      # 256, 14x14
        self.enc4 = resnet.layer4      # 512, 7x7

        # Decoder (UNet style)
        self.up1 = UpBlock(512, 256, 256)  # 7x7 → 14x14
        self.up2 = UpBlock(256, 128, 128)  # 14x14 → 28x28
        self.up3 = UpBlock(128, 64, 64)    # 28x28 → 56x56
        self.up4 = UpBlock(64, 64, 64)     # 56x56 → 112x112

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 112x112 → 224x224
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.pool(x1)
        e1 = self.enc1(x2)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.up1(e4, e3)
        d2 = self.up2(d1, e2)
        d3 = self.up3(d2, e1)
        d4 = self.up4(d3, x1)
        out = self.final_up(d4)
        out = self.final_conv(out)
        return torch.sigmoid(out)
