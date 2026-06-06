"""PyTorch 推理：单张压缩图 → stokes_pred.npy (H,W,4)，输出为物理尺度 Stokes。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from polarized_csi.model_pt import DL_M1


def load_compressed(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        g = np.load(path).astype(np.float32)
    else:
        g = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if g.ndim != 2:
        raise ValueError(f"expected 2D compressed image, got shape {g.shape} from {path}")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-npy", type=Path, default=Path("stokes_pred.npy"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load_compressed(args.input)
    x = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(device)

    model = DL_M1().to(device)
    model.load_state_dict(
        torch.load(str(args.weights), map_location=device, weights_only=True)
    )
    model.eval()
    with torch.no_grad():
        y = model(x)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    np.save(str(args.out_npy), y)
    print(f"Saved {args.out_npy} shape={y.shape}")


if __name__ == "__main__":
    main()
