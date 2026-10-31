# 四成员预训练多 Batch 验证工具

`scripts/validate_nego_pretrain_vectorized.py` 对指定实验配置执行一个真实训练更新：复用正式 Dataset、DataLoader、模型、loss 和 YAML optimizer，检查场景/CAV 边界、输出、loss、全部可训练梯度及至少一个参数更新。它不会保存 checkpoint，也不改变源配置。报告仅在成功后以独占创建方式写入指定目录，不覆盖同名文件。

实验配置必须已经设置目标阶段、`seed: 303`、目标 `train_params.batch_size`、`nego.multi_batch.enabled: true` 和 `mode: vectorized`；工具不会修改这些值。传入的 `--batch-size` 必须与配置一致。指定一个 CUDA 可见设备并通过设备 UUID 核验它是预定的物理 GPU。示例命令中的配置必须是实验配置副本：

```bash
PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python scripts/validate_nego_pretrain_vectorized.py \
  --config /path/to/experiment/config.yaml --stage pretrain/m2 --batch-size 4 \
  --report-dir /path/to/experiment/validation_reports
```

分别使用 `pretrain/m1`、`pretrain/m2`、`pretrain/m3`、`pretrain/m4` 和相应配置进行独立验证；目标 batch size 分别为 4、4、8、4。不得自动更改参数或重试。输出记录实际 GPU UUID，但调用者仍须核对 UUID 与预定设备一致。日志耗时包括一次 forward、loss、backward、检查、更新前参数快照和 optimizer 更新；不包含 DataLoader 读取、模型构建及更新后的参数对比。

LiDAR 阶段报告 voxel 坐标形状及数量；m3 从实际 SECOND sparse 模块输入捕获并核对 sparse batch size。相机阶段报告图像与各标定张量的形状并验证 CAV、相机和矩阵维度。四阶段均记录配置 SHA-256、Git HEAD、seed、样本与 CAV 数、loss、输出形状、参数和梯度数量、梯度范数、更新的参数张量数、optimizer、学习率、GPU UUID、CUDA 显存峰值及耗时。Git HEAD 不证明工作树干净；复现实验时须单独检查 Git 状态。

本地测试只使用合成数据和 CPU，包含四种传感器输入选择、边界及配置拒绝、更新流程和异常路径；不证明真实 Dataset、相机编码、spconv 或 CUDA 正确。m2、m3、m4 真实 V2XSet GPU 验证为 `DEFERRED_TO_SILAS_GPU_VALIDATION`。用户提供的 m1 BS=4 已通过结果本次未复验。原有 m1 验证工具保持不变。

本地完整 `pytest -v`：127 passed，1 条第三方依赖弃用警告。新增脚本和测试的 Python 3.8 语法解析、`git diff --check` 均通过；实际运行环境是 Python 3.12，未运行 Python 3.8 解释器。本次未连接 Silas，未运行真实 GPU 测试。
