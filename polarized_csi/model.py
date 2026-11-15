"""现有 DL-M1 仿射基线：1×1 扩通道 + 7×7 空间卷积，796 参数。"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dl_m1(
    input_shape: tuple[int, int, int] | None = None,
) -> keras.Model:
    """
    DL-M1：输入 (H, W, 1)，输出 (H, W, 4) 对应 S0–S3。
    padding='same' 保持空间分辨率与输入一致。
    """
    if input_shape is None:
        inp = layers.Input(shape=(None, None, 1), name="compressed")
    else:
        h, w, c = input_shape
        if c != 1:
            raise ValueError("compressed input must have 1 channel")
        inp = layers.Input(shape=(h, w, 1), name="compressed")

    x = layers.Conv2D(
        4, (1, 1), padding="same", use_bias=True, name="conv1x1_expand"
    )(inp)
    x = layers.Conv2D(
        4, (7, 7), padding="same", use_bias=True, name="conv7x7_spatial"
    )(x)

    return keras.Model(inputs=inp, outputs=x, name="DL_M1")


def compile_for_stokes_regression(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> None:
    """线性基线：MSE + Adam；学习率需用验证集选择。"""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanSquaredError(name="mse")],
    )
