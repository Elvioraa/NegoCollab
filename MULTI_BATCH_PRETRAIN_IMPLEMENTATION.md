# 多 Batch 预训练实现：未完成，禁止作为正式实验版本

> 2026-10-31 Checkpoint 专项更新：下方原始失败记录作为历史保留。Checkpoint 加载模式和受限兼容声明已在本地修改，最终本地全量测试 **106 passed**；真实历史 checkpoint 和服务器环境仍未验证。四成员多 Batch 改造仍未完成，仍禁止作为正式实验版本。

日期：2026-10-31。工作分支 `feature/multibatch-pretrain`，基准 HEAD `e0f063e9ff9b87d622d9cab1f6136b8e21ac9bf1`。本轮未提交、未推送、未连接 Silas、未运行正式训练；原始 YAML 未修改。先前的 `MULTI_BATCH_FULL_AUDIT.md` 保留。

## 当前阻断

旧原生 checkpoint 加载兼容性测试失败。`tests/test_nego_pretrain_multibatch.py::test_original_native_checkpoint_remains_loadable_after_code_change` 用本地合成模型和 monkeypatch 模拟 checkpoint 写入时与当前代码的实现哈希不同；`opencood/tools/nego_state.py:136` 的严格保护拒绝加载：

```text
ValueError: Checkpoint implementation hash differs; audit/migrate explicitly: .../legacy.pth
1 failed
```

这不是实际服务器 checkpoint 的实测结果，而是对现有哈希契约的可复现测试。`code_hash()` 覆盖被本轮修改的训练和模型文件，因此变更后的代码不能直接宣称能够加载旧 checkpoint。安全兼容需要在不接受任意旧代码、损坏状态或错误协议的前提下，设计并审核明确的基线来源/代码完整性迁移方案；不能简单删除或忽略代码哈希检查。新模式 resume 还需要 checkpoint 中存在准确的 `global_step`，旧 checkpoint 无该字段时不能凭空推断。

按任务要求，出现测试失败后未提交或推送；以下代码均为**未完成、未经充分验证的工作区修改**，不应用于训练。

## 本轮修改及状态

- `opencood/tools/nego_protocol.py`：将显式 vectorized 开关的预训练阶段限制从 m1 扩展到 m1–m4；Alliance、Alignment、FT 仍拒绝多 batch。
- `opencood/data_utils/datasets/nego_m1_batch.py`：保留原 m1 collate；增加通用预训练包装，记录场景数量、CAV 前缀边界、场景索引、模态、CAV ID、object IDs 和样本索引，检查 LiDAR voxel 索引或相机图像/标定数量。原生路径未改。新增包装尚未与正式 Dataset/真实相机数据验证。
- `opencood/models/heter_encoders.py`：仅在新模式中让 SECOND 使用 `sum(record_len)`，检查空 voxel 和索引；保留 m1 已有显式 scatter CAV 总数功能。真实 spconv/尾部空 CAV 尚未验证。
- `opencood/models/sub_modules/pillar_vfe.py`：仅在新模式下以指定维度 squeeze，避免单 voxel 丢失 voxel 维度；原生分支仍使用原来的 squeeze。尚未完成真实完整网络回归。
- `opencood/tools/run_nego_doma.py`：尝试接入四成员 vectorized collate；新模式不再因为空 GT 自动跳过整个 batch，有背景 anchor 时保留原生分类项；无 anchor/深度监督时报错。新模式尝试记录 optimizer `global_step`、驱动 cosine scheduler、检查非有限梯度并写训练日志；原生分支保留原调用。数据加载时间、启用项完整性、日志与异常覆盖及恢复行为仍需补充审查/测试。
- `opencood/tools/nego_state.py`：`pack` 新增可选 `global_step` 用于新模式的 checkpoint 恢复；默认参数不改变原生 payload 形状，但旧 checkpoint 的代码哈希兼容问题尚未解决。
- `tests/test_nego_pretrain_multibatch.py`：增加阶段开关、场景/CAV 边界、相机标定数量和上述旧 checkpoint 兼容测试。使用合成/模拟输入，不等于真实训练验证。

没有修改 Alliance Negotiation、Alliance FT、m3/m4 Alignment/FT、Final Inference 或 `opencood/hypes_yaml/` 的文件。既有 m1 测试文件和服务器验证脚本保留；用户所述既有 m1 BS=4 GPU 结果本轮没有重复或独立核验。

## Loss、空 GT、scheduler 和日志：当前实现与未完成项

原生 `PointPillarLoss` 检测分类/回归/方向对各场景的 anchor 按正样本数量归一化，随后按场景数归约；`PointPillarPyramidLoss` 单车监督按 CAV 数归约。当前修改没有改写这些 loss 或权重。新模式对混合及全空 GT 的批次不做旧版的整批空 GT continue；有负 anchor 时交由原生 loss 计算背景分类，正 anchor 为零时对应回归/方向权重为零。当前无监督判断仅检查 anchor 存在及 depth 配置标志，**尚未完整证明所有 loss 权重、Pyramid 有效掩码和 depth 的边界条件**，不能宣称用户要求全部满足。

新模式每次 `optimizer.step()` 后增加 `global_step`，cosine scheduler 使用此计数；保存含 optimizer 的状态时写入 `resume.global_step`。默认原生路径使用原来的 scheduler 索引。连续运行、尾 batch、旧/新 checkpoint 恢复与学习率轨迹尚未做完整验证。

新模式建立单独的文件与流 handler、可配置 `log_every`（默认 10），尝试记录阶段、epoch、batch、CAV、voxel、loss 组成、学习率、耗时、样本速率、显存及 optimizer step，checkpoint 后写记录。当前 batch 计时从 DataLoader 返回之后开始，**不包含数据加载时间**；loss 组成、所有 checkpoint 场景、异常发生在 logger 初始化前的 traceback 覆盖和 screen 实时性尚未完成验证。不能把这些未验证功能视为最终完成。

## 测试和延迟项

- 首轮本地 `tests/`：**86 passed，1 warning**。该轮发生在新增旧 checkpoint 兼容测试之前，不能解释为最终工作区全部通过。
- 随后执行的 checkpoint 兼容专项：**1 failed**，具体原因见上。本轮按要求停止，不再提交/推送。
- Python 3.8 AST 检查曾在最后添加测试之前通过；最终文件未进行完整重新检查。实际 Python 3.8 解释器未测试。
- m2/m3/m4 的真实 V2XSet Dataset、DataLoader、相机编码、SECOND/spconv、完整模型及 loss、CUDA forward/backward/optimizer、BN 和显存测试均为 `DEFERRED_TO_SILAS_GPU_VALIDATION`。本地合成边界测试不能代替这些验证。
- m1/m2/m3/m4 全阶段真实 batch=4/4/8/4、空 GT 混合/全空、单 voxel、不同场景 CAV、checkpoint 兼容和 scheduler 恢复等验收未完成。

## 发布状态

分支 `feature/multibatch-pretrain` 停留在 `e0f063e9ff9b87d622d9cab1f6136b8e21ac9bf1`，所有本轮改动未提交；没有向 `personal-m1` 或官方 `origin` 推送。远程分支状态未查询。代码不得标记为正式可用。

## Checkpoint 兼容性专项更新

### 原实现及本次区分

原 `code_hash()` 覆盖 `CODE_FILES` 中训练入口、协议、模型与数据文件，加上 `opencood/models`、`opencood/loss`、`opencood/utils`、`opencood/data_utils` 下全部 `.py` 和 `opencood/hypes_yaml/yaml_utils.py`。原 `load_stage()` 比较完整配置哈希和完整实现哈希，并核对 `.sha256`、stage/dataset/protocol 和 checkpoint 自报 state schema；`initialize()` 再按父阶段映射复制权重，`strict_load()` 对实际模型做参数键、shape、dtype 和有限值检查。

现在保留原默认 `exact` 加载；`resume` 额外要求相同完整代码及配置哈希、目标模型类型和完整参数结构、协议、optimizer/scheduler/RNG、epoch 和实际 optimizer `global_step`。新训练过程计数并保存 `global_step`；对不含可靠计数的旧训练状态不猜测、不视为可恢复。原生 batch=1 的计算和 optimizer 调度路径不因计数而改变。

显式 `pretrained` 模式要求来源配置、目标配置、目标模型和协议哈希，验证模型类型、成员/阶段身份、模型配置以及完整 state_dict 的键、shape、dtype、有限值；跨版本还必须匹配经过审查的来源代码哈希与当前模型/数据实现签名声明。`strict_load` 仍调用 `model.load_state_dict(..., strict=True)`。目标训练超参数可不同，但来源完整配置仍须与 checkpoint 记录一致。

新的 checkpoint 元数据保留原始 `code_sha`、`config_sha`、protocol、state schema 和父来源，并记录 Git HEAD（可能为空或指向未提交工作树的基准）、完整实现哈希、模型实现签名、模型配置哈希、完整实验配置哈希、模型身份、epoch 和 global_step。完整实现哈希记录工作树实际代码内容；Git HEAD 本身不代表工作树干净。`.sha256` 用于检测意外损坏，不提供来源认证；`torch.load` 仍只应读取可信的本地 checkpoint。

父阶段默认仍走严格 `exact` 路径。需要经审查的跨版本前置权重时，可在实验配置副本显式设置 `nego.checkpoint.allow_reviewed_pretrained: true`；运行入口按父阶段配置独立构建期望模型并逐个严格核验。此过程可能产生额外模型构建开销和显存需求，尚未在真实环境测量；未知代码、未审查阶段、配置不一致或结构不匹配仍拒绝。

### 历史代码声明与测试边界

从本地 Git 的 `e0f063e9ff9b87d622d9cab1f6136b8e21ac9bf1` 树计算的 Linux 字节版原始完整 `code_sha` 为 `33ca2a8b10d5bd55847786734c0fbcf5d302e6679aec26513f5b102a6f49e7c3`；其源模型/数据文件的换行规范化签名为 `909f2af1da80ed8a9ba11f0eb597ce022dc91b053a06802360f304d954fb8ced`。声明只覆盖该来源的 `pretrain/m1` 与当前经审查的目标签名。审查中 `heter_encoders.PointPillar`、`SECOND` 和 `pillar_vfe` 的变化位于新模式条件分支；单 voxel 新模式行为有意与旧有 squeeze 边界不同，不能声称任意多 Batch 模型数值等价。其他历史版本、其他阶段、被修改的模型/数据代码、不同换行/额外文件导致无法识别的旧完整哈希均默认拒绝，不能因形状相同自动放行。

旧失败测试使用任意字符串假设任何代码哈希变化都兼容，这个假设不成立；已改为验证未知代码拒绝，并新增已审查声明、旧格式、模型配置、键/shape/dtype、源签名、错误模型类型、来源/文件完整性和 Resume 状态的正反测试。父阶段装配也有合成模型集成测试。测试使用模拟/合成 checkpoint，**并未核验 Silas 的实际旧 checkpoint**；Linux 源哈希只有在旧运行时扫描到的文件集合和字节完全对应时才能匹配。真实 checkpoint、真实模型及服务器加载仍待独立验证。

最终稳定工作树本地完整测试：`106 passed, 1 warning`；Python 3.8 **语法解析**和 `git diff --check` 通过，未在 Python 3.8 解释器运行。一次中间完整测试因测试执行时并行修改了被哈希覆盖的源码而失败，其完整 traceback 保存在 `F:\zzuStudy\exp_gpt\Baselines\nego_checkpoint_finaltest_1031b.log`；稳定工作树重跑结果保存在 `F:\zzuStudy\exp_gpt\Baselines\nego_checkpoint_finaltest_1031c.log`。本次没有连接服务器、启动正式训练、提交或推送。

## 2026-10-31 最终实现检查（本节覆盖上文较早的状态）

- 顶层 `seed` 是运行时唯一种子来源：Python、NumPy、PyTorch CPU/CUDA、DataLoader generator、worker 初始化和每个 epoch 的重建均由此派生；checkpoint 保存并恢复各 RNG 状态。既有 cuDNN 设置未新增或改变，日志及 provenance 记录 seed。配置中历史 `nego_protocol.seed` 不再约束顶层种子的值。恢复仅支持现有的 epoch 边界。
- 四成员预训练使用同一批次构造和场景/CAV 元数据；m1 沿用既有 PointPillar 路径，m2 为 EfficientNet/LSS，m3 的 SECOND 显式使用全部 CAV 数量，m4 为 ResNet101/LSS。Camera 图像与标定保留原生 Dataset 的合并顺序；本地合成测试不足以证明真实数据的相机几何和 spconv 行为。原始配置及非预训练阶段未改动。
- 检测 loss 按场景中正样本归一化再按场景归约；pyramid 单车监督按 CAV 归约。新路径保留空 GT 的有效背景分类；有正样本的回归/方向使用原生权重；全无有效监督时报错。原生路径维持原有空样本处理。
- 新模式 scheduler 以 DataLoader 批次数初始化，cosine 调度按实际 optimizer update 的 global_step 更新；last checkpoint 保存 optimizer、scheduler、RNG 和 step。日志默认每 10 batch、允许 YAML 配置，记录阶段、样本/CAV 数、loss 分量、学习率、加载及计算耗时、吞吐量、显存、step 和 seed。服务端实时 screen 行为尚未验证。
- checkpoint 仍执行严格代码、协议、来源、参数及文件完整性检查。跨版本 pretrained 仅允许既有经过审查的明确声明；其余版本和阶段不因形状相同而放行。未经修改的历史 m1 模型实际 checkpoint 未在本轮检验。
- 本地使用现有 `.local_deps` 与合成输入；m2/m3/m4 真正模型、真实 Dataset、深度监督、spconv 和 CUDA 的 forward/backward/optimizer、显存与不同图像尺寸均为 `DEFERRED_TO_SILAS_GPU_VALIDATION`。m1 BS=4 的真实 V2XSet 结果沿用用户提供的既有结果，本轮未复验。模拟模型的一次 batch 更新不是正式模型验证。
- Python 3.8 为语法解析，不代表已在 Python 3.8 解释器运行。具体最终测试和发布状态见本节后的最终记录。

### 最终测试与发布记录

- 全量本地测试：118 passed，1 条第三方 `timm` 弃用警告；完整输出位于 `F:\zzuStudy\exp_gpt\Baselines\nego_multibatch_final_1031c.log`。首次默认环境测试因缺少现有 `.local_deps` 的搜索路径和系统临时目录权限而失败；配置已有依赖路径及工作区测试目录后完整测试通过。未安装依赖。
- 新测试包含 CPU 随机数/生成器恢复、worker seed、随机采样与尾 batch、空 GT 背景和无效监督、四成员合成模型的一次 forward/backward/step、SECOND 实际 forward 的模拟 sparse 组件与尾部空 CAV 检查。没有执行真实 m2/m3/m4 模型的 GPU 测试；原生完整模型 CUDA 数值/梯度对比、真实 scheduler 学习率轨迹、服务器 screen 日志均未完成。
- Python 3.8 语法解析：9 个修改/新增 Python 文件通过；实际 Python 3.8 解释器未运行。`git diff --check` 和提交前暂存区检查通过。
- 分支 `feature/multibatch-pretrain`；本地提交及个人 GitHub 远程完整 SHA 均为 `c8ea6c82218dac5ee5ac2472552892479cf3f90b`。已普通推送至 `personal-m1`；官方 origin 未使用。两份本地审计报告未加入提交。
