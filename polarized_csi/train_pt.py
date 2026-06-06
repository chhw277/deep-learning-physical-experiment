"""
PyTorch 训练 DL-M1（报告 Step 12–13 超参）。

  py -3 -m polarized_csi.train_pt --train-compressed ... --train-stokes ... --out dl_m1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from polarized_csi.dataset_pt import StokesPairDataset
from polarized_csi.model_pt import DL_M1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-compressed", type=Path, required=True)
    ap.add_argument("--train-stokes", type=Path, required=True)
    ap.add_argument("--val-compressed", type=Path, default=None)
    ap.add_argument("--val-stokes", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=7)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=Path("dl_m1.pt"))
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = StokesPairDataset(args.train_compressed, args.train_stokes)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if args.val_compressed and args.val_stokes:
        val_ds = StokesPairDataset(args.val_compressed, args.val_stokes)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    model = DL_M1().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        msg = f"epoch {epoch + 1}/{args.epochs}  train_mse={train_loss:.6f}"
        if val_loader is not None:
            model.eval()
            vloss = 0.0
            vn = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    vloss += loss_fn(model(x), y).item() * x.size(0)
                    vn += x.size(0)
            val_mse = vloss / max(vn, 1)
            msg += f"  val_mse={val_mse:.6f}"
            if val_mse < best_val:
                best_val = val_mse
                torch.save(model.state_dict(), args.out)
        print(msg, flush=True)

    if val_loader is None:
        torch.save(model.state_dict(), args.out)
    elif best_val == float("inf"):
        torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
