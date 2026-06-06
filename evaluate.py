"""评估 DL-M1 模型：对每组数据计算 PSNR 和 SSIM。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polarized_csi.model_pt import DL_M1

# 裁剪区域 (top, bottom, left, right)
CROP_REGION = (148, 678, 394, 873)

# 12组数据
GROUPS = [
    "banana_leaf", "beef", "bottle_bottom", "bottle_cap",
    "leaf", "normal_leaf", "paper", "photo",
    "plastic_ruler", "ruler", "tape", "water",
]

PIC_DIR = Path(r"D:\Pic")
MODEL_PATH = Path(__file__).resolve().parent / "dl_m1.pt"


def crop(img: np.ndarray) -> np.ndarray:
    top, bottom, left, right = CROP_REGION
    return img[top:bottom, left:right]


def load_gray(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return img


def compute_stokes(group: str) -> np.ndarray:
    """从5张偏振角度图计算Stokes参数，返回 (H,W,4) float32 /255归一化。"""
    exts = [".bmp", ".png", ".tif", ".tiff", ".jpg"]
    def find(suffix: str) -> Path:
        for ext in exts:
            p = PIC_DIR / f"{group}{suffix}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Cannot find {group}{suffix}")

    i0 = load_gray(find("_0")) / 255.0
    i225 = load_gray(find("_22.5")) / 255.0
    i45 = load_gray(find("_45")) / 255.0
    i675 = load_gray(find("_67.5")) / 255.0
    i90 = load_gray(find("_90")) / 255.0

    s0 = i0 + i90
    s3 = i0 - i90
    s1 = 2 * (i225 - i675) - np.sqrt(2) * s3
    s2 = 2 * i45 - s0

    stokes = np.stack([s0, s1, s2, s3], axis=-1).astype(np.float32)
    return crop(stokes)


def load_compressed(group: str) -> np.ndarray:
    """加载compressed图像并裁剪，返回 (H,W) float32 [0,1]。"""
    exts = [".bmp", ".png", ".tif", ".tiff", ".jpg"]
    for ext in exts:
        p = PIC_DIR / f"{group}{ext}"
        if p.exists():
            img = load_gray(p) / 255.0
            return crop(img)
    raise FileNotFoundError(f"Cannot find compressed image for {group}")


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    # 数据范围：stokes值范围取 target 的 max-min
    data_range = target.max() - target.min()
    if data_range < 1e-10:
        return 100.0
    return 10 * np.log10(data_range**2 / mse)


def ssim_channel(pred: np.ndarray, target: np.ndarray) -> float:
    data_range = target.max() - target.min()
    if data_range < 1e-10:
        return 1.0
    return float(ssim_fn(target, pred, data_range=data_range))

def main():
    device = torch.device("cpu")
    model = DL_M1().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    ch_names = ["S0", "S1", "S2", "S3"]
    print(f"{'Group':<16} {'PSNR_S0':>8} {'PSNR_S1':>8} {'PSNR_S2':>8} {'PSNR_S3':>8} {'SSIM_S0':>8} {'SSIM_S1':>8} {'SSIM_S2':>8} {'SSIM_S3':>8}")
    print("-" * 88)

    all_psnr = []
    all_ssim = []

    for group in GROUPS:
        comp = load_compressed(group)  # (H,W)
        stokes_gt = compute_stokes(group)  # (H,W,4)

        x = torch.from_numpy(comp).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        with torch.no_grad():
            pred = model(x).squeeze(0).numpy()  # (4,H,W)

        gt = stokes_gt.transpose(2, 0, 1)  # (4,H,W)

        psnrs = []
        ssims = []
        for ch in range(4):
            p = psnr(pred[ch], gt[ch])
            s = ssim_channel(pred[ch], gt[ch])
            psnrs.append(p)
            ssims.append(s)

        all_psnr.append(psnrs)
        all_ssim.append(ssims)

        print(f"{group:<16} {psnrs[0]:8.2f} {psnrs[1]:8.2f} {psnrs[2]:8.2f} {psnrs[3]:8.2f} {ssims[0]:8.4f} {ssims[1]:8.4f} {ssims[2]:8.4f} {ssims[3]:8.4f}")

    # 平均值
    avg_psnr = np.mean(all_psnr, axis=0)
    avg_ssim = np.mean(all_ssim, axis=0)
    print("-" * 88)
    print(f"{'AVERAGE':<16} {avg_psnr[0]:8.2f} {avg_psnr[1]:8.2f} {avg_psnr[2]:8.2f} {avg_psnr[3]:8.2f} {avg_ssim[0]:8.4f} {avg_ssim[1]:8.4f} {avg_ssim[2]:8.4f} {avg_ssim[3]:8.4f}")


if __name__ == "__main__":
    main()
