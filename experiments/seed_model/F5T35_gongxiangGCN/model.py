# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================== 工具：对称规范化 (A+I) (无变动) =====================
def normalize_adj(A: torch.Tensor, add_self_loop: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """
    输入:  A [E,E]（非负、对称）
    输出:  A_norm = D^{-1/2} (A + I) D^{-1/2}
    """
    E = A.size(0)
    if add_self_loop:
        A = A + torch.eye(E, dtype=A.dtype, device=A.device)
    deg = A.sum(dim=1).clamp_min(eps)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


# ===================== GraphConv（无 α；标准 GCN：Â X W） (无变动) =====================
class GraphConvSimple(nn.Module):
    """
    标准图卷积（单分支）：H = ReLU( BN( (Â @ X) @ W ) )，其中 X形状 [B',E,C_in]。
    输入 x:  [B', E, C_in]
         A:  [E, E]（已规范化的 Â）
    输出 h:  [B', E, C_out]
    """
    def __init__(self, c_in, c_out, dropout=0.3):
        super().__init__()
        self.lin = nn.Linear(c_in, c_out, bias=False)
        self.bn  = nn.BatchNorm1d(c_out)
        self.do  = nn.Dropout(dropout)

    def forward(self, x, A):
        # x: [B',E,C_in]
        x_agg = torch.matmul(A, x)               # [B',E,C_in]
        h = self.lin(x_agg)                      # [B',E,C_out]
        Bp, E, C = h.shape
        h = self.bn(h.view(Bp * E, C)).view(Bp, E, C)
        h = F.relu(h, inplace=True)
        h = self.do(h)
        return h

# ===================== 【新模块】Shared GCN (所有频段共享一套GCN) =====================
class SharedGCN_NoAlpha(nn.Module):
    """
    输入 x_BTFE: [B, T, Freq, E]
    输出 H_stack: [B, T, Freq, E, c_mid]
    说明：
      - 所有频段共享一套 GCN 和输入嵌入层。
      - 先用一个共享的线性层把标量幅值嵌入到 Cb，再做一次共享的标准 GCN 到 c_mid。
      - 仅使用固定先验邻接 A_hat 的对称规范化（可选关闭）。
    """
    def __init__(self, A_hat, Cb, c_mid, dropout=0.5, normalize_A=True):
        super().__init__()
        assert A_hat.dim() == 2 and A_hat.size(0) == A_hat.size(1), "A_hat 必须 [E,E]"
        self.register_buffer("A_hat", A_hat.clone())
        self.Cb, self.c_mid = Cb, c_mid
        self.normalize_A = bool(normalize_A)
        if self.normalize_A:
            self.register_buffer("A_norm", normalize_adj(A_hat))
        else:
            self.A_norm = None

        # 共享的频段嵌入层：把每个频段的标量输入 (1) 映到 Cb
        self.embed = nn.Linear(1, Cb, bias=False)
        
        # 共享的 GCN 层
        self.gcn = GraphConvSimple(c_in=Cb, c_out=c_mid, dropout=dropout)

    def _get_A(self):
        return self.A_norm if self.A_norm is not None else self.A_hat

    def forward(self, x_BTFE):
        """
        x_BTFE: [B, T, Freq, E]
        """
        B, T, Freq, E = x_BTFE.shape
        A_use = self._get_A()  # [E,E]

        # 1. 嵌入：[B,T,F,E] -> [B,T,F,E,1] -> [B,T,F,E,Cb]
        x_emb = self.embed(x_BTFE.unsqueeze(-1))
        
        # 2. 重塑以批处理：将 B, T, Freq 合并到批次维度
        # [B,T,F,E,Cb] -> [B*T*F, E, Cb]
        # 这样，每个时间步、每个频段的图都成为 GCN 的一个独立样本
        x_reshaped = x_emb.reshape(B * T * Freq, E, self.Cb)

        # 3. 应用共享的GCN
        # 输入: [B*T*F, E, Cb], 输出: [B*T*F, E, c_mid]
        h = self.gcn(x_reshaped, A_use)
        
        # 4. 还原形状
        # [B*T*F, E, c_mid] -> [B, T, F, E, c_mid]
        H_stack = h.view(B, T, Freq, E, self.c_mid)
        
        return H_stack


# ===================== 时频卷积 (无变动) =====================
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
        # 均分到两支路
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
        # 拼接后 1×1 压回
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

        y1 = self.br1(x)                  # [B*E, c1, T, F]
        y2 = self.br2(x)                  # [B*E, c2, T, F]
        y  = torch.cat([y1, y2], dim=1)   # [B*E, c1+c2, T, F]
        y  = self.proj(y)                 # [B*E, C_out, T, F]

        # LayerNorm：把 (T,F) 展平后按通道归一化
        y  = y.permute(0, 2, 3, 1).contiguous().view(B * E, T * F, -1)  # [B*E, T*F, C_out]
        y  = self.norm(y).view(B * E, T, F, -1)                         # [B*E, T, F, C_out]
        y  = self.drop(y)

        # 融合频段（均值） → [B*E, T, C_out]
        y_tf = y.mean(dim=2)

        # 还原回 [B, T, E, C_out]
        y_tf = y_tf.view(B, E, T, -1).permute(0, 2, 1, 3).contiguous()
        return y_tf


# ===================== 【新】顶层模型 (使用共享GCN) =====================
class STGCN_PB_TFConv_FC_SharedGCN(nn.Module):
    """
    输入:  x_BTFE [B, T, Freq, E]
           A_hat  [E, E]
    流程:  Shared GCN（空间）→ TF_MSConvBank_T35_F5（时频多尺度）
         → 时间池化（mean）→ 展平节点 → FC
    输出:  logits [B, num_classes]
    """
    def __init__(self, A_hat, Freq=5, Cb=16, Cmid=32, Ctf=48,
                 dropout=0.5, normalize_A=True,
                 num_classes=2, head_hidden=48, head_dropout=0.5):
        super().__init__()
        # self.Freq = Freq # Freq 不再是 GCN 模块的配置，但其他地方可能需要
        self.Cmid, self.Ctf = Cmid, Ctf

        # 1) 共享的GCN
        self.shared_gcn = SharedGCN_NoAlpha(
            A_hat, Cb, Cmid, dropout=dropout, normalize_A=normalize_A
        )

        # 2) 时频多尺度卷积（无变动）
        self.tfbank = TF_MSConvBank_T35_F5(C_in=Cmid, C_out=Ctf, pdrop=0.5)

        # 3) 分类头（无变动）
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

        # 1) 空间：Shared GCN
        H_stack = self.shared_gcn(x_BTFE)             # [B,T,F,E,Cmid]

        # 2) 时频：T核=3/5；F核=5
        H_tf = self.tfbank(H_stack)                   # [B,T,E,Ctf]

        # 3) 时间池化（mean over T）
        feat = H_tf.mean(dim=1)                       # [B,E,Ctf]

        # 4) 展平节点 → FC
        flat = feat.reshape(B, E * self.Ctf)          # [B, E*Ctf]
        head = self._ensure_head(E=E, device=x_BTFE.device)
        logits = head(flat)                           # [B,num_classes]
        return logits