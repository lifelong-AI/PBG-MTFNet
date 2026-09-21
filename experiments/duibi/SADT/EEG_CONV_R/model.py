# model.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEG_CONV_Residual(nn.Module):
    def __init__(self, input_shape=(8, 5, 8, 9)):
        super().__init__()
        self.T, self.F, self.H, self.W = input_shape


        self.conv1 = nn.Conv2d(self.F, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.res_conv = nn.Conv2d(32, 64, kernel_size=1, bias=False)


        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # [B*T, 64, 2, 2]
        self.flatten_dim = 64 * 2 * 2             # 256


        self.lstm = nn.LSTM(self.flatten_dim, 128, batch_first=True)


        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, space_mask=None, return_internals=False):
        ""
        B, T, Freq, H, W = x.shape
        assert Freq == self.F and (H, W) == (self.H, self.W), \
            f"Input shape mismatch: got F/H/W=({Freq},{H},{W}), expect ({self.F},{self.H},{self.W})"


        Af = None
        if return_internals:
            with torch.no_grad():
                # mean(|x|) over (T,H,W) -> [B,F]
                Af = x.abs().mean(dim=(1, 3, 4))
                Af = torch.softmax(Af, dim=1)


        x_bt_fhw = x.view(B * T, Freq, H, W)                 # [B*T, F, H, W]

        x1 = F.relu(self.bn1(self.conv1(x_bt_fhw)), inplace=True)  # [B*T, 32, H, W]
        residual = self.res_conv(x1)                               # [B*T, 64, H, W]
        x2 = F.relu(self.bn2(self.conv2(x1)), inplace=True)        # [B*T, 64, H, W]
        x_feat = x2 + residual

        x_feat = self.pool(x_feat)                                 # [B*T, 64, 2, 2]
        x_seq = x_feat.view(B, T, -1)                              # [B, T, 256]


        out, _ = self.lstm(x_seq)                                  # [B, T, 128]
        feat = out[:, -1, :]                                       # [B, 128]
        logits = self.classifier(feat)                             # [B, 2]

        if return_internals:
            return logits, {"A_f": Af}
        return logits



if __name__ == '__main__':
    model = EEG_CONV_Residual().cuda()
    dummy_x = torch.randn(4, 8, 5, 8, 9).cuda()  # B=4
    y, aux = model(dummy_x, space_mask=None, return_internals=True)
    print("logits:", y.shape)           # [4, 2]
    print("A_f   :", aux["A_f"].shape)  # [4, 5]
