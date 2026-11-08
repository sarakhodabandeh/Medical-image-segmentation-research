
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolution layers with batch normalization and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate for filtering encoder skip features."""

    def __init__(self, gate_channels, skip_channels, intermediate_channels):
        super().__init__()

        self.gate_projection = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                intermediate_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.skip_projection = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                intermediate_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )

        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                intermediate_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        gate_features = self.gate_projection(gate)
        skip_features = self.skip_projection(skip)

        attention_map = self.attention(
            gate_features + skip_features
        )

        return skip * attention_map


class AttentionUNet(nn.Module):
    """Attention U-Net for binary breast ultrasound lesion segmentation."""

    def __init__(self, in_channels=3, out_channels=1, base_channels=16):
        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.att4 = AttentionGate(c * 8, c * 8, c * 4)
        self.dec4 = DoubleConv(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.att3 = AttentionGate(c * 4, c * 4, c * 2)
        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.att2 = AttentionGate(c * 2, c * 2, c)
        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.att1 = AttentionGate(c, c, c // 2)
        self.dec1 = DoubleConv(c * 2, c)

        # One logit per pixel
        self.output = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention-gated skip connections
        d4 = self.up4(b)
        a4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))

        d3 = self.up3(d4)
        a3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        return self.output(d1)
