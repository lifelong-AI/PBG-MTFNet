# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 仅沿时间维做 3D 卷积（kT=3, kH=kW=1）
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1),
                              padding=(1, 0, 0), bias=False)
        # 在 [B*T, C, H, W] 上做 2D BN
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x_out = self.conv(x)  # [B, C_out, T, H, W]
        x_bn = x_out.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)  # [B*T, C_out, H, W]
        x_bn = self.relu(self.bn(x_bn))
        x = x_bn.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4).contiguous()  # [B, C_out, T, H, W]
        return x


class ESTCNN(nn.Module):
    def __init__(self, num_bands=5, H=8, W=9):
        super().__init__()
        self.H, self.W = H, W
        self.num_bands = num_bands

        self.block1 = nn.Sequential(
            ConvBlock(num_bands, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 16)
        )
        self.block2 = nn.Sequential(
            ConvBlock(16, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32)
        )
        self.block3 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64)
        )

        # 前两次沿时间维做 2× 下采样；最后一步改为动态平均（见 forward）
        self.pool1 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.pool2 = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # 分类头（输出二分类 logits）
        self.fc1 = nn.Linear(64 * H * W, 50)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(50, 2)  # logits [B,2]

    def forward(self, x, space_mask=None, return_internals=False):
        """
        x: [B, T, F, H, W]
        space_mask: 保留形参以兼容训练脚本，这里不使用
        return_internals: True -> 返回 (logits, {"A_f":[B,F]})
        """
        B, T, F, H, W = x.shape
        assert F == self.num_bands and H == self.H and W == self.W, \
            f"Input shape mismatch: got F/H/W=({F},{H},{W}), expect ({self.num_bands},{self.H},{self.W})"

        # ===== 观测用 A_f（不参与训练）=====
        Af = None
        if return_internals:
            with torch.no_grad():
                # mean(|x|) over (T,H,W) -> [B,F]
                Af = x.abs().mean(dim=(1, 3, 4))
                Af = torch.softmax(Af, dim=1)

        # 变换到 [B, F, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, T, H, W]

        # 三个阶段的时序卷积；前两次固定下采样，最后动态平均到 T=1
        for layer in self.block1:
            x = layer(x)
        x = self.pool1(x)  # T: /2

        for layer in self.block2:
            x = layer(x)
        x = self.pool2(x)  # T: /2

        for layer in self.block3:
            x = layer(x)

        # ★ 动态时间池化：无论当前 T 为多少，直接在时间维取均值 -> T=1
        x = x.mean(dim=2, keepdim=False)  # [B, 64, H, W]

        # 展平 + 分类头（logits）
        x = x.view(B, -1)                 # [B, 64*H*W]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)              # [B, 2]

        if return_internals:
            return logits, {"A_f": Af}
        return logits


# -------------------- 测试 -------------------- #
if __name__ == "__main__":
    for T in [8, 16]:
        B, F, H, W = 4, 5, 8, 9
        input_data = torch.randn(B, T, F, H, W).cuda()
        model = ESTCNN(num_bands=F, H=H, W=W).cuda()
        logits, aux = model(input_data, return_internals=True)
        print(f"T={T} -> logits shape:", logits.shape, "| A_f:", aux["A_f"].shape)
