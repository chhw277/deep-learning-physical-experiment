# 偏振压缩成像实验

从一张压缩测量灰度图估计 S0、S1、S2、S3，研究偏振光学测量与计算重建的关系。当前提供可运行的软件基线，真实实验的精度和泛化能力仍待验证。

**第一次打开项目，先读 [当前进度](docs/STATUS.md)，再按 [实验路线](docs/EXPERIMENT.md) 做。** 不必先理解全部历史代码。

## 现在有什么

| 内容 | 当前状态 |
| --- | --- |
| 五角度照片 → Stokes 标签 | 有程序；理想模型代数可检查，实物角度和符号需标定 |
| DL-M1 重建模型 | 两层卷积，796 参数，无激活函数，属于仿射基线 |
| PyTorch 训练、推理、评价 | 主流程；每次运行保存来源及参数记录 |
| TensorFlow/Keras | 可选实现，共用预处理；旧 TF 权重需重新训练 |
| 8 对 32×40 示例 | 仅用于软件演示；原始 Stokes 物理约束违规率约 87.2%，来源待核实 |
| 实际光路、正式训练测试集、性能指标 | 尚无完整可追溯的验证材料 |

历史报告中的 29.5 dB、0.91、1629 元、0.02 秒/帧等数值**不作为当前已验证成绩**。原文保留在 [历史资料](docs/archive/README.md)，便于核对。

## 安装

建议 Python 3.12，在项目根目录运行：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 改用：.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

依赖分工：`requirements.txt` 是 PyTorch 主流程；`requirements-dev.txt` 增加测试工具。TensorFlow 请在另一个虚拟环境安装 `requirements-tf.txt`，两套权重不能互换。

## 先确认软件能跑

```bash
python -m pytest -q
python scripts/smoke_test.py --out-dir runs/first-smoke
```

第二条命令自动生成**明确标识的理想模拟数据**，执行五角度标签计算核对、数据审计、2 轮训练、最佳权重保存、推理和评价。所有产物保存在指定目录；目录已有内容时会要求换一个新目录，避免覆盖旧记录。

这项检查只证明软件链路工作。模拟数据采用固定偏振态、仅强度变化的受限样本族，不能证明从任意单帧唯一恢复任意四维偏振态，也不能代替实物实验。

## 看已有数据的问题

```bash
python -m polarized_csi.audit --compressed data/demo/train/c --stokes data/demo/train/s --out-json runs/demo-audit.json
```

检查文件配对、尺寸、数值与 `sqrt(S1²+S2²+S3²) <= S0`。加 `--strict-physics` 后，存在理想物理约束违规会返回失败；示例包预期无法通过此严格检查。噪声或通道尺度变换也会影响约束，需要先核对来源，不能据此认定真实实验失败。

## 用真实数据运行

先完成 [采集与验证步骤](docs/EXPERIMENT.md)，并按 [数据约定](docs/DATA.md) 准备灰度图和同名标签。

```bash
# 原始五角度照片 → 标签（示例路径需换成实际路径）
python compute_stokes.py --batch --raw-dir data/raw/session01 --out-dir data/private/session01-labels

# 成对数据检查
python -m polarized_csi.audit --compressed data/private/train/c --stokes data/private/train/s

# 训练：验证集必须与训练集按对象分开
python -m polarized_csi.train_pt --train-compressed data/private/train/c --train-stokes data/private/train/s --val-compressed data/private/val/c --val-stokes data/private/val/s --epochs 20 --learning-rate 0.001 --seed 42 --out models/session01/baseline.pt

# 对未参与训练和模型选择的测试对象推理
python -m polarized_csi.infer_pt --weights models/session01/baseline.pt --input data/private/test/c/object01.tif --out-npy results/session01/object01.npy

# 用同一对象、同一区域、同一尺度的真值评价
python evaluate.py --pred results/session01/object01.npy --true data/private/test/s/object01.npy --data-range 1.0 --out-json results/session01/object01-metrics.json
```

这里的路径是待准备的数据目录，不是仓库自带的已完成实验。无验证集时可以训练，但只能保存最后一轮模型，不能称为最佳泛化模型。默认学习率已从 0.1 调整为 0.001，仍需实验选择。

每个 `.pt`、预测 `.npy` 和新生成标签旁都会写 `.json` 记录；评价也可导出 JSON。`--data-range 1.0` 是示例固定范围，正式实验应提前定义各通道的评价尺度并保持一致。`target` 使用每张真值各通道的动态范围，跨实验比较时必须说明差异。

## 文件地图

```text
compute_stokes.py       五角度图像计算标签
polarized_csi/
  io.py                 统一强度读取与 Stokes 检查
  paths.py              严格配对与训练验证文件重叠检查
  audit.py              不依赖深度学习框架的数据审计
  model_pt.py           PyTorch 仿射基线
  train_pt.py           主训练入口及实验记录
  infer_pt.py           主推理入口及来源记录
  model.py / train.py / infer.py / dataset.py  可选 TensorFlow 实现
  metrics.py            原评价函数接口
  norm_stats.py         历史可选统计量，当前流程不调用
scripts/smoke_test.py    可复现的软件闭环检查
tests/                 数据、标签、指标、训练与模型保存回归测试
data/demo/             仅用于演示的数据
docs/                  当前状态、实验步骤、数据约定、历史资料
models/ / results/     运行产物，默认不提交
```

原本地包及未知来源的权重、预测数组保留在本地原目录，没有被当作新实验成果发布。使用本整理版时以此 Git 仓库目录为准。没有确认许可前，不添加开源许可证；公开仓库不等于已授予任意再分发许可。
