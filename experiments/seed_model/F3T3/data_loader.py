# data_loader_persubject_norm.py
import os, glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ------------ 电极在 8x9 网格上的位置（True=有电极） ------------
ELECTRODE_MATRIX = [
    ['', '', '', '', '', '', '', '', ''],
    ['', '', '', '', '', '', '', '', ''],
    ['FT7', '', '', '', '', '', '', '', 'FT8'],
    ['T7', '', '', '', '', '', '', '', 'T8'],
    ['TP7', '', '', 'CP1', '', 'CP2', '', '', 'TP8'],
    ['', '', '', 'P1', 'PZ', 'P2', '', '', ''],
    ['', '', '', 'PO3', 'POZ', 'PO4', '', '', ''],
    ['', '', '', 'O1', 'OZ', 'O2', '', '', '']
]


def _space_mask_hw():
    return np.array([[bool(c.strip()) for c in row] for row in ELECTRODE_MATRIX], dtype=bool)


def _nodes_from_mask(space_mask):
    H, W = space_mask.shape
    coords = np.argwhere(space_mask)
    idx_flat = coords[:, 0] * W + coords[:, 1]
    return idx_flat.astype(np.int64), coords.astype(np.int64), (H, W)


def _adj_from_coords(coords, device):
    import torch
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
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
    A_hat = deg_inv_sqrt.unsqueeze(1) * A * deg_inv_sqrt.unsqueeze(0)
    return A_hat


class EEGDataLoader:
    def __init__(self, root_path, subjects=23, sample_key='sample', label_col='PERCLOS', min_samples_per_class=50):
        self.root = root_path
        self.sample_key = sample_key
        self.label_col = label_col
        self.min_samples_per_class = min_samples_per_class

        space_mask = _space_mask_hw()
        self.idx_flat, self.coords_hw, self.hw = _nodes_from_mask(space_mask)
        self.E = len(self.idx_flat)

        self.subject_ids_initial = [f"subject_{i:02d}" for i in range(subjects)]
        self.subject_data, self.all_samples = {}, []
        self.excluded_subjects = []

        self._load()
        self._report_loading_status()

        self.space_mask_hw = space_mask
        self.mask_5 = np.stack([space_mask] * 5, 0).astype(bool)

    def _load(self):
        for sid in self.subject_ids_initial:
            data_dir = os.path.join(self.root, 'data_seed', sid)
            label_file = os.path.join(self.root, 'labels', f"{sid}.csv")

            if not os.path.isfile(label_file):
                self.excluded_subjects.append((sid, "Label file not found"))
                continue

            labels_df = pd.read_csv(label_file)

            if self.label_col not in labels_df.columns:
                print(f"[ERROR] Subject {sid}: Label file does not contain column '{self.label_col}'. Skipping.")
                self.excluded_subjects.append((sid, f"Column '{self.label_col}' Not Found"))
                continue

            labels_cont = labels_df[self.label_col].values.astype(float)

            labels_cls = (labels_cont >= 0.35).astype(np.int64)

            class_0_count = np.sum(labels_cls == 0)
            class_1_count = np.sum(labels_cls == 1)

            if class_0_count < self.min_samples_per_class or class_1_count < self.min_samples_per_class:
                reason = f"Sample imbalance (Alert: {class_0_count}, Drowsy: {class_1_count})"
                self.excluded_subjects.append((sid, reason))
                continue

            if not os.path.isdir(data_dir):
                self.excluded_subjects.append((sid, "Data directory not found"))
                continue

            files = sorted(
                glob.glob(os.path.join(data_dir, "sample_*.npz")),
                key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0])
            )

            if len(files) != len(labels_df):
                print(
                    f"[WARN] Subject {sid}: Mismatch between .npz files ({len(files)}) and labels ({len(labels_df)}). Skipping.")
                self.excluded_subjects.append((sid, "File/Label Mismatch"))
                continue

            xs = [np.load(f)[self.sample_key] for f in files]
            arr = np.stack(xs, axis=0).astype(np.float32)

            # Normalize per subject
            mean_hw = np.nanmean(arr, axis=(0, 1, 2), keepdims=True)
            std_hw = np.nanstd(arr, axis=(0, 1, 2), keepdims=True)
            std_hw = np.where((std_hw < 1e-6) | np.isnan(std_hw), 1.0, std_hw)
            arr = (arr - mean_hw) / std_hw

            self.subject_data[sid] = (arr, labels_cls, labels_cont)
            for i in range(arr.shape[0]):
                self.all_samples.append({'subject': sid, 'index': i})

    def _report_loading_status(self):
        print("\n" + "=" * 80)
        print(" EEGDataLoader Loading Status Report")
        print("=" * 80)
        print(f"  - Attempted to load {len(self.subject_ids_initial)} subjects.")
        print(f"  - Successfully loaded {len(self.subject_data)} subjects.")

        if self.excluded_subjects:
            print(
                f"  - Excluded {len(self.excluded_subjects)} subjects who did not meet the criteria (min samples per class: {self.min_samples_per_class}):")
            for sid, reason in self.excluded_subjects:
                print(f"    - {sid}: Excluded (Reason: {reason}).")
        else:
            print("  - All subjects met the inclusion criteria.")
        print("=" * 80 + "\n")

    def get_all(self):
        return self.all_samples, self.subject_data, self.space_mask_hw, self.mask_5

    def get_graph(self, device='cpu'):
        A_hat = _adj_from_coords(self.coords_hw, device=torch.device(device))
        return A_hat, self.E, self.idx_flat, self.hw


# EEGDataset remains unchanged
class EEGDataset(Dataset):
    def __init__(self, samples, subject_data, idx_flat, hw):
        self.samples = samples
        self.subject_data = subject_data
        self.idx_flat = idx_flat
        self.H, self.W = hw

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        info = self.samples[i]
        x, y_cls, y_reg = self.subject_data[info['subject']]
        x_hw = x[info['index']]  # This data is already normalized per subject

        # No normalization needed here
        xp_flat = x_hw.reshape(x_hw.shape[0], x_hw.shape[1], self.H * self.W)
        xp_e = xp_flat[:, :, self.idx_flat]

        return {
            "input_e": torch.from_numpy(xp_e).float(),
            "target": torch.tensor(y_cls[info['index']], dtype=torch.long),
            "perclos": torch.tensor(y_reg[info['index']], dtype=torch.float32),
        }
