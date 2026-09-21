# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- 多尺度卷积分支（GN 版本） ---------------- #
class MultiScaleBranch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, gn_groups=4):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        # 用 GroupNorm 替代 BatchNorm，跨被试更稳健
        # groups 需能整除 out_ch，默认 4；若不整除则退化为 LayerNorm 风格的 GN(groups=1)
        if out_ch % gn_groups != 0:
            gn_groups = 1
        self.gn = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


# ---------------- 通道注意（SE） ---------------- #
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=8):
        super().__init__()
        hidden = max(1, in_ch // ratio)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B,C,T,H,W]
        B, C, _, _, _ = x.shape
        y = self.pool(x).view(B, C)        # [B,C]
        y = self.fc(y).view(B, C, 1, 1, 1) # [B,C,1,1,1]
        return x * y


# ---------------- 空间注意（CAM-3D） ---------------- #
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):  # x: [B,C,T,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)   # [B,1,T,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)     # [B,1,T,H,W]
        attn = torch.cat([max_out, avg_out], dim=1)      # [B,2,T,H,W]
        attn = torch.sigmoid(self.conv(attn))            # [B,1,T,H,W]
        return x * attn


# ---------------- 核心特征：不使用 mask（轻量+正则） ---------------- #
class _APMCNNCore(nn.Module):
    """
    输入 : x[B,T,F,H,W]
    输出 : feat_seq[B,T,D]  (D=F*3*out_ch*H*W)，以及 A_f[B,F]（仅用于观测）
    """
    def __init__(self, out_ch=8, bands=5, H=8, W=9, gn_groups=4,
                 p_spatial_drop3d=0.1, p_band_dropout=0.3):
        super().__init__()
        self.out_ch = out_ch
        self.bands = bands
        self.H, self.W = H, W
        self.p_band_dropout = float(p_band_dropout)

        self.branch3 = MultiScaleBranch(1, out_ch, kernel_size=3, gn_groups=gn_groups)
        self.branch5 = MultiScaleBranch(1, out_ch, kernel_size=5, gn_groups=gn_groups)
        self.branch7 = MultiScaleBranch(1, out_ch, kernel_size=7, gn_groups=gn_groups)

        self.ca = ChannelAttention(out_ch * 3)
        self.sa = SpatialAttention(kernel_size=7)
        self.dropout3d = nn.Dropout3d(p=p_spatial_drop3d)

    def forward(self, x, space_mask_ignored=None):
        """
        x: [B,T,F,H,W]
        space_mask_ignored: 为兼容 train.py 的调用，这里不使用
        """
        B, T, F, H, W = x.shape
        assert F == self.bands, f"bands mismatch: got {F}, expect {self.bands}"
        assert (H, W) == (self.H, self.W), f"HW mismatch: {(H,W)} vs {(self.H,self.W)}"

        # [B,F,T,H,W] -> [B*F,1,T,H,W]
        x_bfthw = x.permute(0, 2, 1, 3, 4).contiguous().view(B * F, 1, T, H, W)

        out3 = self.branch3(x_bfthw)                            # [B*F, C', T, H, W]
        out5 = self.branch5(x_bfthw)
        out7 = self.branch7(x_bfthw)
        feat = torch.cat([out3, out5, out7], dim=1)             # [B*F, 3*C', T, H, W]

        feat = self.ca(feat)
        feat = self.sa(feat)
        feat = self.dropout3d(feat)

        # 还原到 [B,T,F,3*C',H,W]
        C3 = feat.size(1)
        feat = feat.view(B, F, C3, T, H, W).permute(0, 3, 1, 2, 4, 5).contiguous()  # [B,T,F,C3,H,W]

        # 观测用：频段能量 softmax（不参与训练）
        with torch.no_grad():
            Af = feat.abs().mean(dim=(1, 3, 4, 5))              # [B,F]
            Af = torch.softmax(Af, dim=1)                       # [B,F]

        # -------- 频段级随机失活（band dropout）---------
        # 训练阶段随机将某些频段整段置零（所有时刻、通道、空间）
        if self.training and self.p_band_dropout > 0.0:
            # 以 p_band_dropout 的概率丢弃每个 band（独立伯努利）
            drop_mask = (torch.rand(B, F, device=feat.device) > self.p_band_dropout).float()
            drop_mask = drop_mask.view(B, 1, F, 1, 1, 1)        # [B,1,F,1,1,1]
            feat = feat * drop_mask.permute(0, 1, 2, 3, 4, 5)  # 广播到 [B,T,F,C3,H,W]

        # 展平到序列，送 LSTM
        B, T, F, C3, H, W = feat.shape
        feat_seq = feat.view(B, T, F * C3 * H * W)              # [B,T,D]
        return feat_seq, Af


# ---------------- 顶层：对齐 train.py 的类名与接口 ---------------- #
class AMSCNN(nn.Module):
    """
    - 输入: x[B,T,F,H,W], space_mask(被忽略)
    - 输出: logits[B,2]
    - 支持 return_internals=True -> (logits, {'A_f': [B,F]})
    """
    def __init__(self, input_shape=(8, 5, 8, 9),
                 lstm_hidden=64, lstm_layers=1, num_classes=2,
                 out_ch=8, gn_groups=4, p_spatial_drop3d=0.1, p_band_dropout=0.3):
        super().__init__()
        T, F, H, W = input_shape
        self.T, self.F, self.H, self.W = T, F, H, W

        self.core = _APMCNNCore(out_ch=out_ch, bands=F, H=H, W=W,
                                gn_groups=gn_groups,
                                p_spatial_drop3d=p_spatial_drop3d,
                                p_band_dropout=p_band_dropout)

        D_in = F * (out_ch * 3) * H * W            # = F * 3*out_ch * H * W
        self.lstm = nn.LSTM(
            input_size=D_in, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, space_mask=None, return_internals=False):
        # 兼容 train.py 的签名，但忽略 space_mask
        feat_seq, Af = self.core(x, space_mask_ignored=None)    # [B,T,D], [B,F]
        out, _ = self.lstm(feat_seq)                            # [B,T,Hid]
        logits = self.head(out[:, -1, :])                       # [B,2]
        if return_internals:
            return logits, {"A_f": Af}
        return logits


# -------------------- 最小自测 -------------------- #
if __name__ == '__main__':
    model = AMSCNN(input_shape=(8, 5, 8, 9)).cuda()
    dummy = torch.randn(4, 8, 5, 8, 9).cuda()
    y, aux = model(dummy, return_internals=True)
    print("logits:", y.shape)      # [4,2]
    print("A_f   :", aux["A_f"].shape)  # [4,5]
