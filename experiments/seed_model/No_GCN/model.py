# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 工具：对称规范化 (A+I) =====================
def normalize_adj(A: torch.Tensor, add_self_loop: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """
    输入:  A [E,E]（非负、对称）
    输出:  A_norm = D^{-1/2} (A + I) D^{-1/2}
    （本无GCN版本中未使用，保留以保持接口一致）
    """
    E = A.size(0)
    if add_self_loop:
        A = A + torch.eye(E, dtype=A.dtype, device=A.device)
    deg = A.sum(dim=1).clamp_min(eps)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


# ===================== Per-Band（移除GCN；仅逐节点点运算） =====================
class PerBand_NoGCN(nn.Module):
    """
    输入 x_BTFE: [B, T, Freq, E]
    输出 H_stack: [B, T, Freq, E, c_mid]
    说明：
      - 不做空间传播（无 GCN、无邻接），每个频段在每个节点上做点运算：1→Cb→Cmid。
      - 为对齐原模型接口，仍保留 Cb 与 c_mid 两级映射。
    """
    def __init__(self, A_hat, Freq, Cb, c_mid, dropout=0.3, normalize_A=True):
        super().__init__()
        self.Freq, self.Cb, self.c_mid = Freq, Cb, c_mid

        # 频段独立：标量 → Cb 嵌入
        self.embeds = nn.ModuleList([nn.Linear(1, Cb, bias=False) for _ in range(Freq)])
        # 频段独立：Cb → Cmid（点运算），带 BN/GELU/Dropout
        self.proj  = nn.ModuleList([nn.Linear(Cb, c_mid, bias=False) for _ in range(Freq)])
        self.bns   = nn.ModuleList([nn.BatchNorm1d(c_mid) for _ in range(Freq)])
        self.do    = nn.Dropout(dropout)

    def forward(self, x_BTFE):
        """
        x_BTFE: [B, T, Freq, E]
        """
        B, T, Freq, E = x_BTFE.shape
        outs = []
        for f in range(self.Freq):
            # 标量→Cb
            xf = x_BTFE[:, :, f].unsqueeze(-1)                        # [B,T,E,1]
            z  = self.embeds[f](xf)                                   # [B,T,E,Cb]
            # Cb→Cmid（点运算 + BN/GELU/Dropout）
            zf = z.reshape(B * T * E, self.Cb)                        # [B*T*E,Cb]
            hf = self.proj[f](zf)                                     # [B*T*E,Cmid]
            hf = self.bns[f](hf)                                      # BN over Cmid
            hf = F.gelu(hf)
            hf = self.do(hf)
            hf = hf.view(B, T, E, self.c_mid)                         # [B,T,E,Cmid]
            outs.append(hf.unsqueeze(2))                               # [B,T,1,E,Cmid]
        return torch.cat(outs, dim=2)                                  # [B,T,Freq,E,Cmid]


# ===================== 时频卷积（多尺度：T核=3/5；F核固定=5；E 不动；无残差） =====================
class TF_MSConvBank_T35_F5(nn.Module):
    """
    输入:  H_stack [B, T, F, E, C_in]
    做法:  在 (T,F) 上做两条并联卷积（保持 T,F 尺寸不变），频段核=5，时间核∈{3,5}：
           - 支路1: Conv2d(kernel=(3,5))
           - 支路2: Conv2d(kernel=(5,5))
          两支路通道拼接后，用 1×1 压回 C_out；按通道做 LayerNorm；
          频段内做均值融合到 [T,E,C_out]。
    输出:  H_tf [B, T, E, C_out]
    """
    def __init__(self, C_in, C_out, pdrop=0.2):
        super().__init__()
        c1 = C_out // 2
        c2 = C_out - c1

        self.br1 = nn.Sequential(
            nn.Conv2d(C_in, c1, kernel_size=(3,3), padding=(1,1), bias=False),
            nn.GELU(),
        )
        self.br2 = nn.Sequential(
            nn.Conv2d(C_in, c2, kernel_size=(5,5), padding=(2,2), bias=False),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(c1 + c2, C_out, kernel_size=1, bias=False),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(C_out)  # 在展平的 (T*F) 上按通道归一化
        self.drop = nn.Dropout(pdrop)

    def forward(self, H_stack):
        # H_stack: [B, T, F, E, C_in]  →  [B, T, E, C_out]
        B, T, F, E, C = H_stack.shape

        # 按节点独立处理：折叠 B、E → [B*E, C_in, T, F]
        x = H_stack.permute(0, 3, 4, 1, 2).contiguous().view(B * E, C, T, F)

        y1 = self.br1(x)                       # [B*E, c1, T, F]
        y2 = self.br2(x)                       # [B*E, c2, T, F]
        y  = torch.cat([y1, y2], dim=1)        # [B*E, c1+c2, T, F]
        y  = self.proj(y)                      # [B*E, C_out, T, F]

        # LayerNorm：把 (T,F) 展平后按通道归一化
        y  = y.permute(0, 2, 3, 1).contiguous().view(B * E, T * F, -1)  # [B*E, T*F, C_out]
        y  = self.norm(y).view(B * E, T, F, -1)                         # [B*E, T, F, C_out]
        y  = self.drop(y)

        # 融合频段（均值） → [B*E, T, C_out]
        y_tf = y.mean(dim=2)

        # 还原回 [B, T, E, C_out]
        y_tf = y_tf.view(B, E, T, -1).permute(0, 2, 1, 3).contiguous()
        return y_tf


# ===================== 顶层：Per-Band(无GCN) → TF_MSConvBank_T35_F5 → (T池化) → FC =====================
class ST_NoGCN_TFConv_T35_F5_FC(nn.Module):
    """
    输入:  x_BTFE [B, T, Freq, E]
           A_hat  [E, E]（此处不使用，仅为接口对齐）
    流程:  Per-Band（无GCN，仅点运算）→ TF_MSConvBank_T35_F5（时频多尺度）
          → 时间池化（mean）→ 展平节点 → FC
    输出:  logits [B, num_classes]
    """
    def __init__(self, A_hat, Freq=5, Cb=16, Cmid=32, Ctf=48,
                 dropout=0.5, normalize_A=True,
                 num_classes=2, head_hidden=48, head_dropout=0.5):
        super().__init__()
        self.Freq, self.Cmid, self.Ctf = Freq, Cmid, Ctf

        # 1) Per-Band（无GCN）
        self.pbbare = PerBand_NoGCN(
            A_hat, Freq, Cb, Cmid, dropout=dropout, normalize_A=normalize_A
        )

        # 2) 时频多尺度卷积（T×F；E 不动；无残差）
        self.tfbank = TF_MSConvBank_T35_F5(C_in=Cmid, C_out=Ctf, pdrop=0.5)

        # 3) 分类头（懒构建）
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

        # 1) 逐节点点运算：Per-band 无GCN
        H_stack = self.pbbare(x_BTFE)                # [B,T,F,E,Cmid]

        # 2) 时频：T核=3/5；F核=5
        H_tf = self.tfbank(H_stack)                  # [B,T,E,Ctf]

        # 3) 时间池化（mean over T）
        feat = H_tf.mean(dim=1)                      # [B,E,Ctf]

        # 4) 展平节点 → FC
        flat = feat.reshape(B, E * self.Ctf)         # [B, E*Ctf]
        head = self._ensure_head(E=E, device=x_BTFE.device)
        logits = head(flat)                          # [B,num_classes]
        return logits
