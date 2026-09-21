# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 3D 残差块（GroupNorm 版本） ----------
class BasicCSFBlock(nn.Module):
    def __init__(self, ch, gn_groups=4):
        super().__init__()
        g = gn_groups if ch % gn_groups == 0 else 1  # 不能整除时退化为 GN(groups=1)
        self.conv1 = nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn1   = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.gn2   = nn.GroupNorm(g, ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):  # [B,C,T,H,W]
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + x)
        return out


class CSFG(nn.Module):
    """
    - 输入:  x[B,T,F,H,W]；space_mask 仅为兼容，内部不使用
    - 输出:  logits[B,2]
    - return_internals=True: (logits, {"A_f":[B,F]})
    设计改动（轻量化+稳健）：
      * hidden_channels 默认 8，LSTM 隐藏 64
      * 残差块个数可配（n_blocks），默认 1
      * BN -> GroupNorm
      * 频段/时间步 dropout 正则
    """
    def __init__(self, input_dim=(8, 5, 8, 9),
                 hidden_channels=8,            # ↓ 从 16 降到 8
                 lstm_hidden=64,               # ↓ 从 128 降到 64
                 lstm_layers=1,
                 n_blocks=1,                   # ↓ 从固定3改为默认1
                 num_classes=2,
                 gn_groups=4,
                 p_drop3d=0.15,
                 p_banddrop=0.20,
                 p_tdrop=0.15):
        super().__init__()
        T, F, H, W = input_dim
        self.F, self.H, self.W = F, H, W
        self.C = hidden_channels
        self.p_banddrop = float(p_banddrop)
        self.p_tdrop    = float(p_tdrop)

        # 逐频段共享骨干：Initial + n_blocks×Residual
        self.initial = nn.Conv3d(1, self.C, kernel_size=3, padding=1, bias=False)
        g = gn_groups if self.C % gn_groups == 0 else 1
        self.init_gn  = nn.GroupNorm(g, self.C)
        self.init_act = nn.ReLU(inplace=True)

        self.csf = nn.Sequential(*[BasicCSFBlock(self.C, gn_groups=gn_groups) for _ in range(n_blocks)])
        self.drop3d = nn.Dropout3d(p=p_drop3d)

        # 时序建模：LSTM
        self.lstm = nn.LSTM(input_size=self.C * F, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=False)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def _backbone_per_band(self, x_bfthw):  # [B*F,1,T,H,W] -> [B*F,C,T,H,W]
        x = self.init_act(self.init_gn(self.initial(x_bfthw)))
        if len(self.csf) > 0:
            x = self.csf(x)
        x = self.drop3d(x)
        return x

    def forward(self, x, space_mask=None, return_internals=False):
        B, T, F, H, W = x.shape
        assert F == self.F and (H, W) == (self.H, self.W), \
            f"Input shape mismatch: got F/H/W=({F},{H},{W}), expect ({self.F},{self.H},{self.W})"

        # ---------- 正则 1：频段 dropout（训练时随机屏蔽若干频段的全部时空） ----------
        if self.training and self.p_banddrop > 0:
            # Bernoulli(keep) for each band per-sample
            keep = (torch.rand(B, F, device=x.device) > self.p_banddrop).float()  # [B,F]
            x = x * keep.view(B, 1, F, 1, 1)  # 广播到 [B,T,F,H,W]

        # [B,F,T,H,W] -> [B*F,1,T,H,W]
        x_bfthw = x.permute(0, 2, 1, 3, 4).contiguous().view(B * F, 1, T, H, W)

        feat = self._backbone_per_band(x_bfthw)   # [B*F,C,T,H,W]

        # 观测用 Af（不参与反传）
        with torch.no_grad():
            energy = feat.abs().mean(dim=(1, 2, 3, 4))  # [B*F]
            Af = torch.softmax(energy.view(B, F), dim=1)

        # 仅对空间池化，保留时间维
        feat = feat.mean(dim=(3, 4), keepdim=False)     # [B*F,C,T]
        feat = feat.view(B, F, self.C, T).permute(0, 3, 1, 2).contiguous()  # [B,T,F,C]
        feat = feat.view(B, T, F * self.C)             # [B,T,F*C]

        # ---------- 正则 2：时间步 dropout（随机丢弃若干帧） ----------
        if self.training and self.p_tdrop > 0 and T > 1:
            keep_t = (torch.rand(B, T, device=feat.device) > self.p_tdrop).float()  # [B,T]
            # 至少保留最后一帧，避免全丢
            keep_t[:, -1] = 1.0
            feat = feat * keep_t.unsqueeze(-1)  # [B,T,F*C]

        out, _ = self.lstm(feat)                # [B,T,hidden]
        logits = self.head(out[:, -1, :])       # [B,2]

        if return_internals:
            return logits, {"A_f": Af}
        return logits


# -------------------- 最小自测 -------------------- #
if __name__ == '__main__':
    model = CSFG(input_dim=(8, 5, 8, 9)).cuda()
    dummy = torch.randn(2, 8, 5, 8, 9).cuda()
    y, aux = model(dummy, return_internals=True)
    print("logits:", y.shape)      # [2,2]
    print("A_f   :", aux["A_f"].shape)  # [2,5]
