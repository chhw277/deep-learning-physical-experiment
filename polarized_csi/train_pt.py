"""Train the linear DL-M1 baseline and save a run record."""
import argparse
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from polarized_csi.dataset_pt import StokesPairDataset
from polarized_csi.audit import audit_pairs
from polarized_csi.experiment import describe_pairs, save_record
from polarized_csi.model_pt import DL_M1
from polarized_csi.paths import check_disjoint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-compressed', type=Path, required=True)
    ap.add_argument('--train-stokes', type=Path, required=True)
    ap.add_argument('--val-compressed', type=Path)
    ap.add_argument('--val-stokes', type=Path)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=7)
    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--white-level', type=float)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--out', type=Path, default=Path('models/dl_m1.pt'))
    args = ap.parse_args()
    if bool(args.val_compressed) != bool(args.val_stokes):
        ap.error('Pass both validation directories, or neither')
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        ap.error('epochs/batch-size must be positive; num-workers must be non-negative')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        ap.error('learning-rate must be positive and finite')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = StokesPairDataset(args.train_compressed, args.train_stokes, args.white_level)
    validation = (StokesPairDataset(args.val_compressed, args.val_stokes, args.white_level)
                  if args.val_compressed else None)
    if validation is not None:
        check_disjoint(train.pairs, validation.pairs)
    physics = audit_pairs(train.pairs, args.white_level)
    if physics['invalid_pixels']:
        warnings.warn(f"Training labels violate raw Stokes constraints in {physics['invalid_fraction']:.2%} of pixels; see data audit. These results do not establish physical reconstruction quality.")
    for dataset in (train, validation):
        if dataset is not None:
            for index in range(len(dataset)):
                dataset[index]
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        generator=torch.Generator().manual_seed(args.seed))
    val_loader = (DataLoader(validation, batch_size=args.batch_size) if validation is not None else None)
    model = DL_M1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    history, best, selected_epoch = [], float('inf'), None
    record = {'framework': 'pytorch', 'python': sys.version.split()[0],
              'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
              'model': 'DL-M1 linear baseline, 796 parameters, no activation',
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'preprocessing': 'shared loader; normalized [0,1]; labels unchanged',
              'train': describe_pairs(train.pairs),
              'validation': describe_pairs(validation.pairs) if validation is not None else [],
              'history': history,
              'training_data_audit': physics,
              'selection': 'minimum validation MSE' if validation is not None else 'final epoch; no validation'}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss; checkpoint not updated')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite gradient; checkpoint not updated')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite weights; checkpoint not updated')
            total += loss.item() * x.size(0)
            count += x.size(0)
        row = {'epoch': epoch, 'train_mse': total / count}
        if val_loader is not None:
            model.eval()
            total = count = 0
            with torch.no_grad():
                for x, y in val_loader:
                    loss = loss_fn(model(x.to(device)), y.to(device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite validation loss; checkpoint not updated')
                    total += loss.item() * x.size(0)
                    count += x.size(0)
            row['val_mse'] = total / count
            if row['val_mse'] < best:
                best, selected_epoch = row['val_mse'], epoch
                torch.save(model.state_dict(), args.out)
        elif epoch == args.epochs:
            selected_epoch = epoch
            torch.save(model.state_dict(), args.out)
        history.append(row)
        record['selected_epoch'] = selected_epoch
        save_record(args.out, record)
        print(row)


if __name__ == '__main__':
    main()
