# 验证记录

验证日期：2026-11-15。代码提交 `fe997f0` 的 [GitHub Actions 运行记录](https://github.com/chhw277/deep-learning-physical-experiment/actions/runs/43691694177) 已通过；以下是软件检查，不是实物重建成绩。

## 已执行的检查

| 检查 | 结果 | 覆盖范围 |
| --- | --- | --- |
| PyTorch / 共享逻辑 | 23 passed，11.80秒 | 位深、尺度、配对、五角度公式、指标、实际训练与验证选优、推理、同种子CPU重复性、异常输入 |
| TensorFlow / 共享逻辑 | 22 passed，2.78秒，5条依赖弃用警告 | 共享逻辑、避免重复归一化、实际单批训练与模型保存加载、验证最优权重不被覆盖 |
| PyTorch 软件闭环 | PASS | 理想模拟数据生成、公式核对、严格数据审计、2轮训练、保存、推理、评价 |
| 原示例审计 | 8对，10240像素，8929违规 | 共同尺度的原始Stokes约束；[审计JSON](validation/demo-audit.json) |
| 本地静态及保留性核对 | PASS | Python语法、现役文档链接、Git空白检查、归档原文及16个示例文件逐字节保持 |

共享核心有19项测试，分别在两个环境运行，不能把23+22解释为45项不同测试。两条CI任务没有跳过所指定的测试。TF验证选优回归以受控的两轮损失驱动真实ModelCheckpoint；单批训练另行执行，尚未把完整TF命令行训练闭环列为已验证。

## 验证环境和命令

GitHub托管Ubuntu 26.04.1，CPython 3.12.16，NumPy 2.5.4、Pillow 12.3.0、scikit-image 0.26.0、pytest 9.2.0；PyTorch 2.15.0+cpu；独立TF任务为TensorFlow 2.21.0、Keras 3.15.1。精确安装输出见上述CI日志，当前依赖文件为兼容范围，未锁定全部传递依赖。

```bash
python -m pytest -q tests/test_core.py tests/test_pytorch.py
python scripts/smoke_test.py --out-dir runs/ci-smoke
# 在另一个已安装TF依赖的环境
python -m pytest -q tests/test_core.py tests/test_tensorflow.py
```

本机macOS已完成静态检查及示例审计；框架安装遇到下载超时，未把本机两框架运行写成通过。上述运行结果来自干净的GitHub Linux CPU环境，尚未验证GPU、Windows或本机训练。

## 未消除的警告和实验边界

- Keras/NumPy交互产生5条`__array__ copy`弃用警告；本轮单批训练、序列化及选优检查均通过。
- Actions依赖报告Node 20、punycode及url.parse弃用提示；当前执行成功，后续升级Actions版本时重新验证。
- 冒烟样本为固定偏振态、仅强度变化的受限理想模拟族。2轮训练和评价只检查软件可运行，不能作为精度、泛化或单帧普遍可逆性的证据。
- 真实光路标定、数据来源核实、独立测试集以及速度/成本实测仍为pending，下一步按[实验路线](EXPERIMENT.md)执行。

## 整理和发布状态

代码、现役文档与项目规则：changed-and-verified；远端代码及自动测试：verified-current。实物实验：pending。部署服务、长期记忆维护：not-applicable。校级中期报告文件：out-of-scope，未发布。本地原包、旧权重和复核用环境保留，未执行删除清场。
