# -*- coding: utf-8 -*-
# data_loader.py (已修改为按被试独立归一化并接受 min_samples_per_class 参数)
import os, glob, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ------------ SEED 8x9 网格（True=有电极；沿用你的“上面那版”） ------------
ELECTRODE_MATRIX = [
    ['', '', '', 'FP1', '', 'FP2', '', '', ''],
    ['F7', '', 'F3', '', 'FZ', '', 'F4', '', 'F8'],
    ['FT7', '', 'FC3', '', 'FCZ', '', 'FC4', '', 'FT8'],
    ['T7', '', 'C3', '', 'CZ', '', 'C4', '', 'T8'],
    ['TP7', '', 'CP3', '', 'CPZ', '', 'CP4', '', 'TP8'],
    ['P7', '', 'P3', '', 'PZ', '', 'P4', '', 'P8'],
    ['', '', '', '', '', '', '', '', ''],
    ['', '', '', 'O1', 'OZ', 'O2', '', '', ''],
]


# -------------------- 工具函数 --------------------
def _space_mask_hw():
    """8x9 网格上哪些格子是真实电极。"""
    return np.array([[bool(c.strip()) for c in row] for row in ELECTRODE_MATRIX], dtype=bool)  # [H,W]


def _nodes_from_mask(space_mask):
    """返回: idx_flat[E], coords[E,2](y,x), (H,W)"""
    H, W = space_mask.shape
    coords = np.argwhere(space_mask)  # [E,2]
    idx_flat = coords[:, 0] * W + coords[:, 1]
    return idx_flat.astype(np.int64), coords.astype(np.int64), (H, W)


def _adj_from_coords(coords, device):
    """8 邻域 + 自环 + 对称归一化 → A_hat:[E,E] (torch.float32)"""
    E = coords.shape[0]
    A0 = torch.zeros((E, E), dtype=torch.float32, device=device)
    for i in range(E):
        yi, xi = coords[i]
        for j in range(i + 1, E):
            yj, xj = coords[j]
            if abs(yi - yj) <= 1 and abs(xi - xj) <= 1 and not (yi == yj and xi == xj):
                A0[i, j] = 1.0
                A0[j, i] = 1.0
    A = A0 + torch.eye(E, device=device)
    deg = A.sum(dim=1).clamp(min=1e-6)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    A_hat = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)
    return A_hat  # [E,E]


# ---------------- 数据加载器（读取 HxW 网格数据 → 节点化为 [T,Freq,E]） ----------------
class EEGDataLoader:
    """
    目录结构（与 train.py 保持一致）:
      root/
        ├─ data_16589/subject_xx/sample_0000.npz  (键 sample -> [T,Freq,H,W])
        └─ labels/subject_xx.csv                 (含 'CLASS' 列，0/1)

    输出：
      - get_all() -> (all_samples, subject_data, space_mask_hw, mask_5)
      - get_graph(device) -> (A_hat[E,E], E, idx_flat[E], (H,W))
    """

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 修复点 1：添加 min_samples_per_class 参数 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    def __init__(self, root_path, subjects=23, sample_key='sample', label_col='CLASS', min_samples_per_class=50):
        self.root = root_path
        self.sample_key = sample_key
        self.label_col = label_col
        self.min_samples_per_class = min_samples_per_class  # 保存参数
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        # 固定电极集合（一次性）
        space_mask = _space_mask_hw()
        self.idx_flat, self.coords_hw, self.hw = _nodes_from_mask(space_mask)
        self.E = len(self.idx_flat)

        # 载入样本缓存（先按 H×W 读入，Dataset 再节点化为 [T,Freq,E]）
        self.subject_ids = [f"subject_{i:02d}" for i in range(subjects)]
        self.subject_data, self.all_samples = {}, []
        self._load()

        # Z-score 用的 H×W 掩码
        self.space_mask_hw = space_mask
        # 频段掩码（5 个频段时便于广播；若你的数据 F≠5，不用它也没关系）
        self.mask_5 = np.stack([space_mask] * 5, 0).astype(bool)

    def _load(self):
        print(f"  - Attempting to load {len(self.subject_ids)} subjects.")
        loaded_subjects_count = 0
        excluded_subjects_info = []

        for sid in self.subject_ids:
            data_dir = os.path.join(self.root, 'data_16589', sid)
            label_file = os.path.join(self.root, 'labels', f"{sid}.csv")
            if not os.path.isdir(data_dir) or not os.path.isfile(label_file):
                continue

            df = pd.read_csv(label_file)
            if self.label_col not in df.columns:
                raise KeyError(f"{label_file} 未发现列 '{self.label_col}'")
            labels_cls = df[self.label_col].astype(int).to_numpy()

            # 样本筛选逻辑
            n_class_0 = np.sum(labels_cls == 0)
            n_class_1 = np.sum(labels_cls == 1)

            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 修复点 2：使用 self.min_samples_per_class ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            if n_class_0 < self.min_samples_per_class or n_class_1 < self.min_samples_per_class:
                reason = f"Sample imbalance (Class 0: {n_class_0}, Class 1: {n_class_1})"
                excluded_subjects_info.append(f"    - {sid}: Excluded (Reason: {reason}).")
                continue
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            files = sorted(
                glob.glob(os.path.join(data_dir, "sample_*.npz")),
                key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
            )
            if len(files) == 0:
                warnings.warn(f"[{sid}] 未发现样本文件，跳过");
                continue

            xs = []
            for f in files:
                with np.load(f, allow_pickle=True) as z:
                    if self.sample_key not in z:
                        raise KeyError(f"{f} 缺少键 '{self.sample_key}'")
                    x = z[self.sample_key].astype(np.float32)
                    xs.append(x)
            arr = np.stack(xs, axis=0).astype(np.float32)
            assert len(labels_cls) == arr.shape[0], f"{sid} 标签与样本数不一致"

            # 对当前被试的所有数据进行 Z-score 归一化
            mean = np.mean(arr, axis=(0, 1, 2), keepdims=True)
            std = np.std(arr, axis=(0, 1, 2), keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)  # 避免除以零
            arr = (arr - mean) / std

            # 存储已经归一化好的数据
            self.subject_data[sid] = (arr, labels_cls)
            for i in range(arr.shape[0]):
                self.all_samples.append({'subject': sid, 'index': i})

            loaded_subjects_count += 1

        print(f"  - Successfully loaded {loaded_subjects_count} subjects.")
        if excluded_subjects_info:
            print(
                f"  - Excluded {len(excluded_subjects_info)} subjects who did not meet the criteria (min samples per class: {self.min_samples_per_class}):")
            for info in excluded_subjects_info:
                print(info)
        print("=" * 80)

    def get_all(self):
        return self.all_samples, self.subject_data, self.space_mask_hw, self.mask_5

    def get_graph(self, device='cpu'):
        A_hat = _adj_from_coords(self.coords_hw, device=torch.device(device))
        return A_hat, self.E, self.idx_flat, self.hw


# ---------------- Dataset（不再执行归一化） ----------------
class EEGDataset(Dataset):
    """
    返回：
      - input_e: [T, Freq, E]（仅真实电极）
      - target : int64 (0/1)
      - （可选）input_hw: [T, Freq, H, W]（可视化/其他模型）
    """

    def __init__(self, samples, subject_data, idx_flat, hw, return_hw: bool = False):
        self.samples = samples
        self.subject_data = subject_data
        self.idx_flat = idx_flat.astype(np.int64)
        self.H, self.W = hw
        self.return_hw = return_hw

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        info = self.samples[i]
        x_hw, y_cls = self.subject_data[info['subject']]
        x_hw_sample = x_hw[info['index']].astype(np.float32)

        xp = np.nan_to_num(x_hw_sample, copy=False)

        xp_flat = xp.reshape(xp.shape[0], xp.shape[1], self.H * self.W)
        xp_e = xp_flat[:, :, self.idx_flat]

        out = {
            "input_e": torch.from_numpy(xp_e).float(),
            "target": torch.tensor(y_cls[info['index']], dtype=torch.long),
        }
        if self.return_hw:
            out["input_hw"] = torch.from_numpy(xp).float()
        return out

