# 示例数据

`train/c` 有8张32×40灰度PNG，`train/s` 有8份同名32×40×4标签。文件原样保留，来源与生成过程待核实。

按未变换原始Stokes共同尺度解释，8929 / 10240像素违反理想偏振物理约束。因此仅作读写和训练流程演示，不能用于证明真实实验精度。没有擅自改写标签或把它们描述成已确认的实测/模拟数据。

可运行 `python -m polarized_csi.audit --compressed data/demo/train/c --stokes data/demo/train/s` 查看。软件端到端验证使用 `scripts/smoke_test.py` 另行生成并明确标识的理想模拟数据。
