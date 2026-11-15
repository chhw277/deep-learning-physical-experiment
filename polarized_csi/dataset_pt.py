"""PyTorch dataset using the shared intensity and label contract."""
import torch
from torch.utils.data import Dataset

from polarized_csi.io import load_measurement, load_stokes
from polarized_csi.paths import list_pairs


class StokesPairDataset(Dataset):
    def __init__(self, compressed_dir, stokes_dir, white_level=None):
        self.pairs = list_pairs(compressed_dir, stokes_dir)
        self.white_level = white_level

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        compressed, label = self.pairs[index]
        gray = load_measurement(compressed, self.white_level)
        stokes = load_stokes(label, gray.shape)
        return torch.from_numpy(gray).unsqueeze(0), torch.from_numpy(stokes).permute(2, 0, 1)
