# 偏振压缩成像 — DL-M1 深度学习模型

基于单次压缩测量图像，利用两层 CNN（DL-M1）预测完整 Stokes 参数 (S0, S1, S2, S3)。

## 环境准备

```bash
pip install -r requirements.txt
```

依赖：PyTorch、NumPy、Pillow、scikit-image（详见 `requirements.txt`）。

## 目录结构

```
.
├── compute_stokes.py          # 从 5 张偏振照片计算 Stokes 真值
├── polarized_csi/
│   ├── model_pt.py            # DL-M1 模型定义（Conv2D 1×1 → Conv2D 7×7）
│   ├── train_pt.py            # 训练脚本
│   ├── infer_pt.py            # 推理脚本
│   ├── dataset_pt.py          # 数据加载器
│   ├── metrics.py             # PSNR / SSIM 评估
│   └── paths.py               # 文件配对工具
├── data/demo/                 # 示例数据（8 组样本 + 预训练权重）
│   ├── train/c/               # 压缩测量图 (.png)
│   ├── train/s/               # Stokes 真值 (.npy)
│   ├── dl_m1.pt               # demo 预训练权重
│   └── pred.npy               # demo 推理结果
├── dl_m1.pt                   # 完整训练权重
└── requirements.txt
```

## 快速体验（用自带 demo 数据）

### 1. 训练

```bash
python -m polarized_csi.train_pt \
    --train-compressed data/demo/train/c \
    --train-stokes     data/demo/train/s \
    --out demo_model.pt
```

### 2. 推理

```bash
python -m polarized_csi.infer_pt \
    --weights dl_m1.pt \
    --input   data/demo/train/c/sample00.png \
    --out-npy stokes_pred.npy
```

输出 `stokes_pred.npy`，shape 为 `(H, W, 4)`，4 个通道分别对应 S0、S1、S2、S3（**物理尺度**，与训练真值一致，不做 z-score 反变换）。

## 使用自己的数据

### 第一步：计算 Stokes 真值

每组物品需要 5 张偏振照片（QWP 角度 0°、22.5°、45°、67.5°、90°），放在以物品名命名的文件夹中。

```bash
# 批量处理所有物品
python compute_stokes.py --batch --raw-dir <原始照片目录> --out-dir <输出目录>

# 单个物品
python compute_stokes.py --name <物品名> --raw-dir <原始照片目录> --out-dir <输出目录>
```

### 第二步：训练

```bash
python -m polarized_csi.train_pt \
    --train-compressed <压缩测量图目录> \
    --train-stokes     <Stokes真值目录> \
    --out my_model.pt \
    --epochs 2000 --learning-rate 0.001 --batch-size 2
```

可选加验证集：`--val-compressed <目录> --val-stokes <目录>`

### 第三步：推理

```bash
python -m polarized_csi.infer_pt \
    --weights my_model.pt \
    --input   <压缩测量图> \
    --out-npy result.npy
```

### 第四步：评估

在 Python 中加载预测和真值进行对比：

```python
import numpy as np
from polarized_csi.metrics import mean_ssim, psnr_channels

pred = np.load("stokes_pred.npy")
true = np.load("stokes_true.npy")

psnr = psnr_channels(pred, true)
for i, name in enumerate(["S0", "S1", "S2", "S3"]):
    print(f"{name}: PSNR={psnr[i]:.2f} dB")
print(f"mean SSIM={mean_ssim(pred, true):.4f}")
```
