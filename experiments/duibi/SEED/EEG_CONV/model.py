# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGCONV(nn.Module):
    """
    适配 data_loader.py & train.py 的轻量 2D-CNN + LSTM 分类模型
    - 输入:  x[B,T,F,H,W]
    - 输出:  logits[B,2]  (用于 CrossEntropyLoss)
    - 兼容:  forward(x, space_mask=None, return_internals=False)
             当 return_internals=True 时，返回 (logits, {"A_f":[B,F]})
             这里的 A_f 基于原始输入 x 的频段能量（mean|x| over T,H,W）做 softmax，仅用于观测
    """
    def __init__(self, input_shape=(8, 5, 8, 9)):
        super().__init__()
        T, F, H, W = input_shape
        self.T, self.F, self.H, self.W = T, F, H, W

        # 以 “频段作为通道” 做 2D 卷积（每个时间步独立）
        self.conv1 = nn.Conv2d(self.F, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        # 空间自适应池化 -> 2x2（每时间步），随后展平作为 LSTM 的每步输入
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # 输出: [B*T, 64, 2, 2]
        self.flatten_dim = 64 * 2 * 2             # = 256

        # 时间建模
        self.lstm = nn.LSTM(
            input_size=self.flatten_dim,
            hidden_size=128,
            batch_first=True,
            bidirectional=False
        )

        # 分类头（2 类 logits）
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, space_mask=None, return_internals=False):
        """
        x: [B,T,F,H,W]
        space_mask: 为兼容 train.py 形参，这里不使用
        """
        B, T, Freq, H, W = x.shape
        assert Freq == self.F and (H, W) == (self.H, self.W), \
            f"Input shape mismatch: got F/H/W=({Freq},{H},{W}), expect ({self.F},{self.H},{self.W})"

        # ====== 频段注意观测 A_f（不参与训练，仅返回给 analyze_attention）======
        with torch.no_grad():
            # mean(|x|) over (T,H,W) -> [B,F]
            Af = x.abs().mean(dim=(1, 3, 4))      # [B,F]
            Af = torch.softmax(Af, dim=1)         # 归一化

        # ====== 逐时间步 2D-CNN，频段作通道 ======
        x_bt_fhw = x.view(B * T, Freq, H, W)      # [B*T, F, H, W]
        x_bt = F.relu(self.bn1(self.conv1(x_bt_fhw)))  # [B*T, 32, H, W]
        x_bt = F.relu(self.bn2(self.conv2(x_bt)))      # [B*T, 64, H, W]
        x_bt = self.pool(x_bt)                     # [B*T, 64, 2, 2]

        # 重组回序列 [B, T, 256]
        x_seq = x_bt.view(B, T, -1)               # [B, T, 256]

        # LSTM 时间建模
        out, _ = self.lstm(x_seq)                 # [B, T, 128]

        # 分类头（取最后时间步）
        logits = self.head(out[:, -1, :])         # [B, 2]

        if return_internals:
            return logits, {"A_f": Af}
        return logits


# -------------------- 自测 -------------------- #
if __name__ == '__main__':
    model = FatigueDetection_FreqHybridGate_ConvGRU(input_shape=(8, 5, 8, 9)).cuda()
    dummy_x = torch.randn(2, 8, 5, 8, 9).cuda()
    y, aux = model(dummy_x, space_mask=None, return_internals=True)
    print("logits:", y.shape)            # [2,2]
    print("A_f   :", aux["A_f"].shape)   # [2,5]
