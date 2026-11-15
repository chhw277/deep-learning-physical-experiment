"""
可选 TensorFlow 训练入口；当前运行约定见 README。

用法:
  python -m polarized_csi.train \\
    --train-compressed path/to/train/compressed \\
    --train-stokes path/to/train/stokes \\
    --val-compressed path/to/val/compressed \\
    --val-stokes path/to/val/stokes \\
    --out path/to/weights.keras

数据约定见 polarized_csi.dataset.list_pairs。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow import keras
import math
import tensorflow as tf
from polarized_csi.experiment import describe_pairs, save_record
from polarized_csi.audit import audit_pairs
import warnings

from polarized_csi.dataset import StokesPairSequence
from polarized_csi.paths import list_pairs, check_disjoint
from polarized_csi.model import build_dl_m1, compile_for_stokes_regression


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-compressed", type=Path, required=True)
    p.add_argument("--train-stokes", type=Path, required=True)
    p.add_argument("--val-compressed", type=Path, default=None)
    p.add_argument("--val-stokes", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=Path("dl_m1_trained.keras"))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--white-level', type=float)
    args = p.parse_args()
    if bool(args.val_compressed) != bool(args.val_stokes):
        p.error('Pass both validation directories, or neither')
    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error('epochs, batch-size and learning-rate must be positive and finite')
    keras.utils.set_random_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    train_pairs = list_pairs(args.train_compressed, args.train_stokes)
    physics = audit_pairs(train_pairs, args.white_level)
    if physics['invalid_pixels']:
        warnings.warn(f"Labels violate raw Stokes constraints in {physics['invalid_fraction']:.2%} of pixels; verify provenance before interpreting results.")
    train_seq = StokesPairSequence(
        train_pairs, batch_size=args.batch_size, shuffle=True, seed=args.seed, white_level=args.white_level
    )

    val_seq = None
    if args.val_compressed and args.val_stokes:
        val_pairs = list_pairs(args.val_compressed, args.val_stokes)
        check_disjoint(train_pairs, val_pairs)
        val_seq = StokesPairSequence(
            val_pairs, batch_size=args.batch_size, shuffle=False, seed=None, white_level=args.white_level
        )

    model = build_dl_m1()
    compile_for_stokes_regression(model, learning_rate=args.learning_rate)

    callbacks = [keras.callbacks.TerminateOnNaN()]
    if val_seq is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(args.out),
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            )
        )

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if not all(math.isfinite(float(v)) for values in history.history.values() for v in values):
        raise FloatingPointError('Non-finite training history; run failed')
    if val_seq is None:
        model.save(str(args.out))
    save_record(args.out, {'framework': 'tensorflow', 'tensorflow': tf.__version__,
                          'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                          'train': describe_pairs(train_pairs),
                          'validation': describe_pairs(val_pairs) if val_seq is not None else [],
                          'history': history.history,
                          'training_data_audit': physics,
                          'selected_epoch': int(min(range(len(history.history['val_loss'])), key=lambda i: history.history['val_loss'][i]) + 1) if val_seq is not None else args.epochs,
                          'preprocessing': 'shared loader; normalized [0,1]; labels unchanged'})


if __name__ == "__main__":
    main()
