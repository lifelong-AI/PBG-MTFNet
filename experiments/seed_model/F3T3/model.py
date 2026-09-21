# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(A: torch.Tensor, add_self_loop: bool = True, eps: float = 1e-6) -> torch.Tensor:
    ""
    E = A.size(0)
    if add_self_loop:
        A = A + torch.eye(E, dtype=A.dtype, device=A.device)
    deg = A.sum(dim=1).clamp_min(eps)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt



class GraphConvSimple(nn.Module):
    ""
    def __init__(self, c_in, c_out, dropout=0.3):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out, bias=False)
        self.bn  = nn.BatchNorm1d(c_out)
        self.do  = nn.Dropout(dropout)

    def forward(self, x, A):
        # x: [B',E,C_in]
        x_agg = torch.matmul(A, x)              # [B',E,C_in]
        h = self.lin(x_agg)                     # [B',E,C_out]
        Bp, E, C = h.shape
        h = self.bn(h.view(Bp * E, C)).view(Bp, E, C)
        h = F.relu(h, inplace=True)
        h = self.do(h)
        return h



class PerBandGCN_NoAlpha(nn.Module):
    ""
    def __init__(self, A_hat, Freq, Cb, c_mid, dropout=0.3, normalize_A=True):
        super().__init__()
        assert A_hat.dim() == 2 and A_hat.size(0) == A_hat.size(1), "A_hat must have shape [E,E]"
        self.register_buffer("A_hat", A_hat.clone())
        self.Freq, self.Cb, self.c_mid = Freq, Cb, c_mid
        self.normalize_A = bool(normalize_A)
        if self.normalize_A:
            self.register_buffer("A_norm", normalize_adj(A_hat))
        else:
            self.A_norm = None

        self.embeds = nn.ModuleList([nn.Linear(1, Cb, bias=False) for _ in range(Freq)])
        self.gcns   = nn.ModuleList([
            GraphConvSimple(c_in=Cb, c_out=c_mid, dropout=dropout)
            for _ in range(Freq)
        ])

    def _get_A(self):
        return self.A_norm if self.A_norm is not None else self.A_hat

    def forward(self, x_BTFE):
        """
        x_BTFE: [B, T, Freq, E]
        """
        B, T, Freq, E = x_BTFE.shape
        A_use = self._get_A()                  # [E,E]
        outs = []
        for f in range(self.Freq):
            xf = x_BTFE[:, :, f].unsqueeze(-1)             # [B,T,E,1]
            z  = self.embeds[f](xf)                        # [B,T,E,Cb]
            z  = z.permute(0, 1, 3, 2).reshape(B * T, E, self.Cb)  # [B*T,E,Cb]
            hf = self.gcns[f](z, A_use).view(B, T, E, self.c_mid)   # [B,T,E,Cmid]
            outs.append(hf.unsqueeze(2))                   # [B,T,1,E,Cmid]
        return torch.cat(outs, dim=2)                      # [B,T,Freq,E,Cmid]



class TF_Conv_5x5(nn.Module):
    ""
    def __init__(self, C_in, C_out, pdrop=0.2):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=(3,3), padding=(1,1), bias=False)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(C_out)
        self.drop = nn.Dropout(pdrop)

    def forward(self, H_stack):
        # H_stack: [B, T, F, E, C_in]  →  [B, T, E, C_out]
        B, T, F, E, C = H_stack.shape


        x = H_stack.permute(0, 3, 4, 1, 2).contiguous().view(B * E, C, T, F)

        y = self.conv(x)                               # [B*E, C_out, T, F]
        y = self.act(y)


        y = y.permute(0, 2, 3, 1).contiguous().view(B * E, T * F, -1)  # [B*E, T*F, C_out]
        y = self.norm(y).view(B * E, T, F, -1)                         # [B*E, T, F, C_out]
        y = self.drop(y)


        y_tf = y.mean(dim=2)


        y_tf = y_tf.view(B, E, T, -1).permute(0, 2, 1, 3).contiguous()
        return y_tf



class STGCN_PB_TFConv5x5_FC(nn.Module):
    ""
    def __init__(self, A_hat, Freq=5, Cb=16, Cmid=32, Ctf=48,
                 dropout=0.5, normalize_A=True,
                 num_classes=2, head_hidden=48, head_dropout=0.5):
        super().__init__()
        self.Freq, self.Cmid, self.Ctf = Freq, Cmid, Ctf

        # 1) Per-Band GCN
        self.pbgcn = PerBandGCN_NoAlpha(
            A_hat, Freq, Cb, Cmid, dropout=dropout, normalize_A=normalize_A
        )


        self.tf = TF_Conv_5x5(C_in=Cmid, C_out=Ctf, pdrop=0.5)


        self._head = None
        self._head_cfg = dict(num_classes=num_classes,
                              head_hidden=head_hidden,
                              head_dropout=head_dropout)

    def _ensure_head(self, E: int, device: torch.device):
        if self._head is None:
            in_dim = E * self.Ctf
            self._head = nn.Sequential(
                nn.Linear(in_dim, self._head_cfg["head_hidden"], bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(self._head_cfg["head_dropout"]),
                nn.Linear(self._head_cfg["head_hidden"], self._head_cfg["num_classes"], bias=True),
            ).to(device)
        else:
            if next(self._head.parameters()).device != device:
                self._head = self._head.to(device)
        return self._head

    def forward(self, x_BTFE):
        """
        x_BTFE: [B, T, Freq, E]
        """
        B, T, F, E = x_BTFE.shape
        device = x_BTFE.device


        H_stack = self.pbgcn(x_BTFE)     # [B,T,F,E,Cmid]


        H_tf = self.tf(H_stack)          # [B,T,E,Ctf]


        feat = H_tf.mean(dim=1)          # [B,E,Ctf]


        flat = feat.reshape(B, E * self.Ctf)  # [B, E*Ctf]
        head = self._ensure_head(E=E, device=device)
        logits = head(flat)                   # [B,num_classes]
        return logits
