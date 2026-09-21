# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, T=8, F=5, H=8, W=9):
        super(EEGNet, self).__init__()
        # 这些属性仅用于形状断言/卷积超参，不再用于 forward 的展平长度
        self.T, self.F, self.H, self.W = T, F, H, W

        # Block 1: Temporal 卷积 + BN
        # 输入会被 reshape 为 [B, 1, F, L]，在 L 维上进行一维样式的卷积（用 Conv2d 实现）
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # Block 2: 空间卷积（Depthwise）
        # ZeroPad2d(pad) 的顺序是 (left, right, top, bottom)
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(F, 32), groups=8, bias=False)  # Depthwise
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4))

        # Block 3: “Pointwise” 卷积（按你原始实现保留 kernel=(1,4)）
        self.padding2 = nn.ZeroPad2d((2, 1, 0, 0))
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(1, 4), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4))

        # 线性分类头：LazyLinear 会在首次前向时自动确定 in_features
        self.fc = nn.LazyLinear(2)

        self.drop_p = 0.25

    def forward(self, x, space_mask=None, return_internals=False):
        """
        x: [B, T, F, H, W]
        space_mask: 兼容形参，这里不使用
        return_internals: True 时返回 (logits, {'A_f':[B,F]})
        """
        B, T_cur, F_cur, H_cur, W_cur = x.shape
        # 频段数量通常固定为 5，如你的卷积核用到了 F；这里做一致性检查
        assert F_cur == self.F, f"Got F={F_cur}, but model was built with F={self.F}"

        # ===== 可选：频段注意观测 A_f（不参与训练）=====
        Af = None
        if return_internals:
            with torch.no_grad():
                Af = x.abs().mean(dim=(1, 3, 4))   # mean(|x|) over (T,H,W) -> [B,F]
                Af = torch.softmax(Af, dim=1)

        # 预处理到 [B, 1, F, L]，其中 L = T*H*W（动态）
        L = T_cur * H_cur * W_cur
        x = x.permute(0, 2, 1, 3, 4).contiguous()      # [B, F, T, H, W]
        x = x.reshape(B, 1, F_cur, L)                  # [B, 1, F, L]

        # Block 1
        x = self.conv1(x)                              # [B, 8, F, L]
        x = self.bn1(x)
        x = F.elu(x, inplace=True)
        x = F.dropout(x, p=self.drop_p, training=self.training)

        # Block 2
        x = self.padding1(x)
        x = self.conv2(x)                              # [B, 8, 1, L']
        x = self.bn2(x)
        x = F.elu(x, inplace=True)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.pool2(x)

        # Block 3
        x = self.padding2(x)
        x = self.conv3(x)                              # [B, 16, 1, L'']
        x = self.bn3(x)
        x = F.elu(x, inplace=True)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.pool3(x)

        # 展平 + 分类头（输出 logits）
        x = x.view(B, -1)                              # [B, C*L_final]
        logits = self.fc(x)                            # [B, 2]

        if return_internals:
            return logits, {"A_f": Af}
        return logits


# ========== ✅ 自测 ==========
if __name__ == "__main__":
    # 测试 T=8
    model = EEGNet(T=8, F=5, H=8, W=9)
    dummy = torch.randn(8, 8, 5, 8, 9)
    y, aux = model(dummy, return_internals=True)
    print("Logits (T=8):", y.shape, "A_f:", aux["A_f"].shape)

    # 测试 T=16（验证动态 T 兼容）
    dummy2 = torch.randn(4, 16, 5, 8, 9)
    y2 = model(dummy2)
    print("Logits (T=16):", y2.shape)
