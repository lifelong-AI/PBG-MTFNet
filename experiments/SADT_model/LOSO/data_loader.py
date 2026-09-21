# -*- coding: utf-8 -*-

import os, glob, warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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



def _space_mask_hw():
    ""
    return np.array([[bool(c.strip()) for c in row] for row in ELECTRODE_MATRIX], dtype=bool)  # [H,W]


def _nodes_from_mask(space_mask):
    ""
    H, W = space_mask.shape
    coords = np.argwhere(space_mask)  # [E,2]
    idx_flat = coords[:, 0] * W + coords[:, 1]
    return idx_flat.astype(np.int64), coords.astype(np.int64), (H, W)


def _adj_from_coords(coords, device):
    ""
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



class EEGDataLoader:
    ""


    def __init__(self, root_path, subjects=23, sample_key='sample', label_col='CLASS', min_samples_per_class=50):
        self.root = root_path
        self.sample_key = sample_key
        self.label_col = label_col
        self.min_samples_per_class = min_samples_per_class
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


        space_mask = _space_mask_hw()
        self.idx_flat, self.coords_hw, self.hw = _nodes_from_mask(space_mask)
        self.E = len(self.idx_flat)


        self.subject_ids = [f"subject_{i:02d}" for i in range(subjects)]
        self.subject_data, self.all_samples = {}, []
        self._load()


        self.space_mask_hw = space_mask

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
                raise KeyError(f"{label_file}: missing column '{self.label_col}'")
            labels_cls = df[self.label_col].astype(int).to_numpy()


            n_class_0 = np.sum(labels_cls == 0)
            n_class_1 = np.sum(labels_cls == 1)


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
                warnings.warn(f"[{sid}] No sample files found; skipping");
                continue

            xs = []
            for f in files:
                with np.load(f, allow_pickle=True) as z:
                    if self.sample_key not in z:
                        raise KeyError(f"{f}: missing key '{self.sample_key}'")
                    x = z[self.sample_key].astype(np.float32)
                    xs.append(x)
            arr = np.stack(xs, axis=0).astype(np.float32)
            assert len(labels_cls) == arr.shape[0], f"{sid}: label and sample counts do not match"


            mean = np.mean(arr, axis=(0, 1, 2), keepdims=True)
            std = np.std(arr, axis=(0, 1, 2), keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            arr = (arr - mean) / std


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



class EEGDataset(Dataset):
    ""

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

