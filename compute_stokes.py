"""compute_stokes.py — 从 5 张偏振照片计算 Stokes 参数 (S0, S1, S2, S3) 并保存为 .npy

原理（基于旋转四分之一波片法）：
    LP 固定在 45°，QWP 分别转到 0°、22.5°、45°、67.5°、90°
    拍摄得到 5 张灰度强度图 I(0), I(22.5), I(45), I(67.5), I(90)

    由 Mueller 矩阵推导，Stokes 参数为：
        S0 = I(0) + I(90)
        S3 = I(0) - I(90)
        S1 = 2*[I(22.5) - I(67.5)] - sqrt(2)*S3
        S2 = 2*I(45) - S0

用法：
    （单个物品）
    py -3 compute_stokes.py --name ruler --raw-dir stokes_raw --out-dir stokes_npy

    （批量处理全部物品）
    py -3 compute_stokes.py --batch --raw-dir stokes_raw --out-dir stokes_npy

文件命名约定：
    stokes_raw/ 文件夹中，每个物品的 5 张照片命名为：
        {物品名}_0.png
        {物品名}_22.5.png
        {物品名}_45.png
        {物品名}_67.5.png
        {物品名}_90.png

    例如：ruler_0.png, ruler_22.5.png, ruler_45.png, ruler_67.5.png, ruler_90.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ANGLE_SUFFIXES = ["_0", "_22.5", "_45", "_67.5", "_90"]


def load_gray(path: Path) -> np.ndarray:
    """读取图片并转为灰度 float32 数组（0~255 范围）。"""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def find_image(directory: Path, stem: str) -> Path:
    """在目录中查找指定 stem 的图片文件（支持 png/jpg/tif 等格式）。"""
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到文件：{directory / stem}.*")


def compute_stokes(raw_dir: Path, name: str) -> np.ndarray:
    """从 5 张照片计算 Stokes 参数，返回 (H, W, 4) 数组。"""
    # 读取 5 张图并归一化到 [0, 1]
    I_0    = load_gray(find_image(raw_dir, f"{name}_0")) / 255.0
    I_22   = load_gray(find_image(raw_dir, f"{name}_22.5")) / 255.0
    I_45   = load_gray(find_image(raw_dir, f"{name}_45")) / 255.0
    I_67   = load_gray(find_image(raw_dir, f"{name}_67.5")) / 255.0
    I_90   = load_gray(find_image(raw_dir, f"{name}_90")) / 255.0

    # 计算 Stokes 参数
    S0 = I_0 + I_90
    S3 = I_0 - I_90
    S1 = 2.0 * (I_22 - I_67) - np.sqrt(2.0) * S3
    S2 = 2.0 * I_45 - S0

    # 堆叠为 (H, W, 4)
    stokes = np.stack([S0, S1, S2, S3], axis=-1)
    return stokes


def get_all_names(raw_dir: Path) -> list[str]:
    """自动扫描 raw_dir，找出所有物品名称。

    通过查找以 '_0' 结尾的文件来确定物品名。
    例如 ruler_0.png → 物品名为 'ruler'
    """
    names = set()
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        for p in raw_dir.glob(f"*_0{ext}"):
            # 文件名去掉 "_0" 后缀就是物品名
            item_name = p.stem[:-2]  # 去掉 "_0"
            names.add(item_name)
    return sorted(names)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 5 张偏振照片计算 Stokes 参数")
    ap.add_argument("--name", type=str, default=None,
                    help="单个物品名称（如 ruler）")
    ap.add_argument("--batch", action="store_true",
                    help="批量处理 raw-dir 中所有物品")
    ap.add_argument("--raw-dir", type=Path, required=True,
                    help="存放 5 张原始偏振照片的文件夹")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="输出 .npy 文件的文件夹")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        names = get_all_names(args.raw_dir)
        if not names:
            print(f"错误：在 {args.raw_dir} 中没有找到任何物品照片")
            print(f"请确保文件命名格式为：物品名_0.png, 物品名_22.5.png 等")
            return
        print(f"找到 {len(names)} 个物品：{names}")
    elif args.name:
        names = [args.name]
    else:
        print("错误：请指定 --name 或 --batch")
        return

    for name in names:
        print(f"正在处理：{name} ...", end=" ")
        try:
            stokes = compute_stokes(args.raw_dir, name)
            out_path = args.out_dir / f"{name}.npy"
            np.save(out_path, stokes)
            print(f"OK 保存到 {out_path}  形状={stokes.shape}")
        except FileNotFoundError as e:
            print(f"FAIL {e}")

    print("\n全部完成！")


if __name__ == "__main__":
    main()
