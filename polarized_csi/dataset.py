"""Optional Keras dataset; shares preprocessing with PyTorch."""
import numpy as np
from tensorflow import keras

from polarized_csi.io import load_measurement, load_stokes

_load_gray_float = load_measurement
_load_stokes_npy = load_stokes


class StokesPairSequence(keras.utils.Sequence):
    def __init__(self, pairs, batch_size, shuffle=True, seed=42, white_level=None, **kwargs):
        super().__init__(**kwargs)
        if batch_size < 1 or not pairs:
            raise ValueError('Need non-empty data and batch_size >= 1')
        self.pairs = list(pairs)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.white_level = white_level
        self._rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.pairs) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self._rng.shuffle(self.pairs)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        batch = self.pairs[index * self.batch_size:(index + 1) * self.batch_size]
        xs, ys = [], []
        for compressed, label in batch:
            gray = load_measurement(compressed, self.white_level)
            xs.append(gray[..., np.newaxis])
            ys.append(load_stokes(label, gray.shape))
        return np.stack(xs), np.stack(ys)
