# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, T=8, F=5, H=8, W=9):
        super(EEGNet, self).__init__()

        self.T, self.F, self.H, self.W = T, F, H, W



        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)



        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(8, 8, kernel_size=(F, 32), groups=8, bias=False)  # Depthwise
        self.bn2 = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4))


        self.padding2 = nn.ZeroPad2d((2, 1, 0, 0))
        self.conv3 = nn.Conv2d(8, 16, kernel_size=(1, 4), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 4))


        self.fc = nn.LazyLinear(2)

        self.drop_p = 0.25

    def forward(self, x, space_mask=None, return_internals=False):
        ""
        B, T_cur, F_cur, H_cur, W_cur = x.shape

        assert F_cur == self.F, f"Got F={F_cur}, but model was built with F={self.F}"


        Af = None
        if return_internals:
            with torch.no_grad():
                Af = x.abs().mean(dim=(1, 3, 4))   # mean(|x|) over (T,H,W) -> [B,F]
                Af = torch.softmax(Af, dim=1)


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


        x = x.view(B, -1)                              # [B, C*L_final]
        logits = self.fc(x)                            # [B, 2]

        if return_internals:
            return logits, {"A_f": Af}
        return logits



if __name__ == "__main__":

    model = EEGNet(T=8, F=5, H=8, W=9)
    dummy = torch.randn(8, 8, 5, 8, 9)
    y, aux = model(dummy, return_internals=True)
    print("Logits (T=8):", y.shape, "A_f:", aux["A_f"].shape)


    dummy2 = torch.randn(4, 16, 5, 8, 9)
    y2 = model(dummy2)
    print("Logits (T=16):", y2.shape)
