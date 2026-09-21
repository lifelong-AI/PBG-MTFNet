# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=(15, 1, 1), padding=(7, 0, 0), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FineConvBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        hidden = max(1, in_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, T, H, W]
        b, c, _, _, _ = x.shape
        s = self.avg_pool(x).view(b, c)           # [B,C]
        w = self.fc(s).view(b, c, 1, 1, 1)        # [B,C,1,1,1]
        return x * w


class MSCNN_CAM(nn.Module):
    def __init__(self, input_channels=5, output_dim=2):
        super().__init__()
        self.input_channels = input_channels

        self.coarse_branch = CoarseConvBranch(input_channels, 16)
        self.fine_branch   = FineConvBranch(input_channels, 16)

        self.attn = ChannelAttention(32)


        self.pool2d = nn.AdaptiveAvgPool2d(1)


        self.classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, space_mask=None, return_internals=False):
        ""
        B, T, F, H, W = x.shape
        assert F == self.input_channels, f"Input F={F}, expected {self.input_channels}"


        Af = None
        if return_internals:
            with torch.no_grad():
                Af = x.abs().mean(dim=(1, 3, 4))  # [B,F] over (T,H,W)
                Af = torch.softmax(Af, dim=1)


        x = x.permute(0, 2, 1, 3, 4).contiguous()


        feat_coarse = self.coarse_branch(x)   # [B,16,T,H,W]
        feat_fine   = self.fine_branch(x)     # [B,16,T,H,W]
        feat = torch.cat([feat_coarse, feat_fine], dim=1)  # [B,32,T,H,W]


        feat = self.attn(feat)                # [B,32,T,H,W]


        feat = feat.mean(dim=2, keepdim=False)


        pooled = self.pool2d(feat).view(B, -1)


        logits = self.classifier(pooled)

        if return_internals:
            return logits, {"A_f": Af}
        return logits



if __name__ == '__main__':
    model = MSCNN_CAM(input_channels=5, output_dim=2).cuda()
    for T in [8, 16]:
        dummy_x = torch.randn(2, T, 5, 8, 9).cuda()
        y, aux = model(dummy_x, return_internals=True)
        print(f"T={T}  logits:", y.shape, "  A_f:", aux["A_f"].shape)
