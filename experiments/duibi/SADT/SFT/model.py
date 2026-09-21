# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn   = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        hidden = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, C, T, H, W]
        B, C, _, _, _ = x.shape
        y = x.mean(dim=(2, 3, 4))         # [B,C]
        attn = self.fc(y).view(B, C, 1, 1, 1)
        return x * attn



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):  # x: [B, C, T, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out   = x.mean(dim=1, keepdim=True)
        attn = torch.cat([max_out, mean_out], dim=1)   # [B,2,T,H,W]
        attn = torch.sigmoid(self.conv(attn))          # [B,1,T,H,W]
        return x * attn



class FrequencyAttention(nn.Module):
    ""
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)
        self.sigm = nn.Sigmoid()

    def forward(self, x):  # [B,C,T,H,W]
        s = x.mean(dim=(3, 4), keepdim=True)           # [B,C,T,1,1]
        a = self.fc2(self.relu(self.fc1(s)))           # [B,C,T,1,1]
        a = self.sigm(a)
        return x * a



class ConvBranch(nn.Module):
    def __init__(self, in_ch=1, out_ch=16):
        super().__init__()
        self.block = Conv3DBlock(in_ch, out_ch)

    def forward(self, x):
        return self.block(x)



class SFTNet(nn.Module):
    def __init__(self, input_dim=(8, 5, 8, 9), pred_steps=1):
        ""
        super().__init__()
        T, F, H, W = input_dim
        self.T, self.F, self.H, self.W = T, F, H, W
        self.C = 16


        self.freq_branch    = ConvBranch(1, self.C)
        self.temp_branch    = ConvBranch(1, self.C)
        self.spatial_branch = ConvBranch(1, self.C)


        self.freq_attn    = FrequencyAttention(self.C)
        self.spatial_attn = SpatialAttention()
        self.channel_attn = ChannelAttention(self.C)


        self.input_step_dim = self.F * self.C * self.H * self.W
        self.lstm = nn.LSTM(input_size=self.input_step_dim, hidden_size=128, batch_first=True)


        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, space_mask=None, return_internals=False):
        ""
        B, T, F, H, W = x.shape
        assert (F, H, W) == (self.F, self.H, self.W), \
            f"Input shape mismatch: got (F,H,W)=({F},{H},{W}), expected ({self.F},{self.H},{self.W})"


        Af = None
        if return_internals:
            with torch.no_grad():
                Af = x.abs().mean(dim=(1, 3, 4))  # [B,F] over (T,H,W)
                Af = torch.softmax(Af, dim=1)


        x_bf = x.permute(0, 2, 1, 3, 4).contiguous().view(B * F, 1, T, H, W)

        f1 = self.freq_branch(x_bf)     # [B*F, C, T, H, W]
        f2 = self.temp_branch(x_bf)     # [B*F, C, T, H, W]
        f3 = self.spatial_branch(x_bf)  # [B*F, C, T, H, W]
        feat = f1 + f2 + f3


        feat = self.freq_attn(feat)
        feat = self.spatial_attn(feat)
        feat = self.channel_attn(feat)


        feat = feat.view(B, F, self.C, T, H, W).permute(0, 3, 1, 2, 4, 5).contiguous()
        feat = feat.view(B, T, -1)


        out, _ = self.lstm(feat)        # [B, T, 128]
        logits = self.head(out[:, -1, :])  # [B, 2]

        if return_internals:
            return logits, {"A_f": Af}
        return logits



if __name__ == '__main__':

    for T in [8, 30]:
        model = SFTNet(input_dim=(T, 5, 8, 9)).cuda()
        dummy_x = torch.randn(2, T, 5, 8, 9).cuda()
        y, aux = model(dummy_x, return_internals=True)
        print(f"T={T} -> logits:", y.shape, "| A_f:", aux["A_f"].shape)
