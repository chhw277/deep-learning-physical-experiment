"""PyTorch DataLoader 用数据集（与 dataset 文件命名约定相同）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from polarized_csi.paths import list_pairs


class StokesPairDataset(Dataset):
    def __init__(
        self,
        compressed_dir: str | Path,
        stokes_dir: str | Path,
    ):
        self.pairs = list_pairs(compressed_dir, stokes_dir)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cp, sp = self.pairs[idx]
        # Load compressed: either .npy or image
        if cp.suffix.lower() == ".npy":
            g = np.load(cp).astype(np.float32)
        else:
            g = np.array(Image.open(cp).convert("L"), dtype=np.float32) / 255.0
        st = np.load(sp).astype(np.float32)  # 不归一化，直接用原始Stokes
        if g.ndim != 2:
            raise ValueError(f"bad gray image {cp}")
        if st.shape[:2] != g.shape[:2] or st.shape[-1] != 4:
            raise ValueError(f"bad stokes {sp} shape {st.shape} vs gray {g.shape}")
        x = torch.from_numpy(g).unsqueeze(0)
        y = torch.from_numpy(st).permute(2, 0, 1)
        return x, y
