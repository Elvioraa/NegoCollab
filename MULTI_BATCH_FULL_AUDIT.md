# NegoCollab 多阶段多 Batch 训练代码审查

审查日期：2026-10-31。范围为本地工作树和现有代码的静态审查；本轮未执行训练、推理、单元测试或服务器访问。代码位置以本工作树 HEAD 为准。以下“风险”不等于已经在真实数据上复现。

## 1. 当前代码状态

- 工作树：`F:\zzuStudy\exp_gpt\Baselines\NegoCollab-protocol-review`；分支 `feature/m1-vectorized-batch`；HEAD `e0f063e9ff9b87d622d9cab1f6136b8e21ac9bf1`。审查前 `git status --short` 无改动；Git 报告无法读取用户全局 ignore 文件的权限警告。
- `origin` 为官方 `https://github.com/scz023/NegoCollab.git`；`personal-m1` 为个人 `https://github.com/Elvioraa/NegoCollab.git`。本轮未访问远程，远程当前状态未核验。
- 既有 V2XSet 各训练阶段 YAML 当前是 batch size 1；用户本次表格中的 4/8 是未来实验目标，并不是当前生效配置，也不是官方固定值。未修改任何 YAML。
- 已有本地回归曾报告 77 项通过，但本轮没有重跑。m1 的真实 V2XSet、CUDA、spconv、完整模型 forward/backward 和 GPU 显存验证尚无本轮证据，不能标为已通过。

## 2. 已有 m1 功能与边界

`opencood/tools/nego_protocol.py:353-366` 读取 `nego.multi_batch.enabled` 与 `mode`，默认关闭；仅 `pretrain/m1` 可以选择 `enabled: true, mode: vectorized`。其他训练阶段或原生模式下 batch 大于 1 被运行时保护拒绝；验证 batch 也被限制为 1。这里是运行时安全限制，`validate_hyperparameters` 的正整数检查本身不把训练 batch 限定为特定数值。

`opencood/tools/run_nego_doma.py:39-47` 只在训练且开关开启时使用 `collate_m1_vectorized`，其余路径沿用原 collate。`opencood/data_utils/datasets/nego_m1_batch.py:5-41` 一次调用原训练 collate，核对样本 CAV 数、m1 模态、record_len、各模态数量、变换/标签维度和 voxel 索引，额外保存 `object_ids_by_sample`。它不是逐样本模型 forward，也不是梯度累积；默认路径不增加这些字段。

`opencood/models/heter_encoders.py:38-53` 在开启路径将 `sum(record_len)` 传给 `PointPillarScatter`；`opencood/models/sub_modules/point_pillar_scatter.py:42-54` 用该显式 CAV 总数保留中间或尾部无 voxel 的 CAV。原生未设置该字段时仍使用坐标最大值。**整批 voxel 为空**被 m1 collate 显式拒绝；不是已经验证能处理所有无 voxel 情况。`opencood/models/sub_modules/pillar_vfe.py` 的 `features.squeeze()` 对恰好一个 voxel 可能丢掉维度，需要单独测试。单独无 voxel CAV 的零特征语义也需真实数据验证。

`scripts/validate_nego_m1_vectorized.py` 包含服务器真实 Dataset、一次 forward/backward/optimizer 更新、GPU UUID 和峰值统计检查，并对 batch 1 比较输出、loss、梯度和参数更新；该脚本尚未在本轮运行，不能视为验证结果。其 batch 1 比较尚应覆盖输入结构、所有模型 buffer/BN 状态和 scheduler/RNG；其耗时从模型调用附近开始，不等于包含读取与预处理的端到端吞吐，GPU 利用率仅为单次快照。

## 3. 各阶段数据与模型审查

| 阶段 | 当前入口与限制 | 下一步需要核查/改动的位置 |
|---|---|---|
| m1 Pretrain | `intermediateheter`、`HomoCollab`、PointPillar、Pyramid loss；已有可选向量化路径，但真实数据未验证。 | `nego_m1_batch.collate_m1_vectorized`、`heter_encoders.PointPillar.forward`、`PointPillarScatter.forward`、`pillar_vfe.PillarVFE.forward`、`HomoCollab.forward`、`PyramidFusion.forward_collab` 和 loss；完善缺失 voxel、标签、完整梯度与 BN 验证。 |
| m2 Pretrain | 同样是 `HomoCollab`，但为 EfficientNet 相机；m1 专用 collate 和开关显式不适用。 | 普通异构 collate 的相机 `merge='cat'`、LSS `get_cam_feats`/`voxel_pooling` 的 CAV 维度、相机标定及图像与 `record_len` 的对应；对相机缺失规则继续严格校验。 |
| m3 Pretrain | `HomoCollab` + SECOND；没有已验证的多 batch 支持。 | `heter_encoders.SECOND.forward:69-79` 从 `voxel_coords[:,0].max()+1` 推断数量；尾部或全空 CAV 的 sparse batch 可能遗漏/失败。需审计 spconv sparse backbone 和 height compression 的显式 batch 数量及空输入行为。 |
| m4 Pretrain | `HomoCollab` + ResNet 相机；当前仍受单 batch 保护。 | 与 m2 相同的相机维度/标定和 per-CAV 索引，另外验证 m4 backbone、图像尺寸、BN 与显存；不可假设与 m2 相同。 |
| Alliance Negotiation | `intermediateheterpair` + `heter_nego_train` + `nego_loss`。配对 Dataset 是对同一场景内每个 CAV 为各模态构造输入，而不是简单地将实际混合模态列表按模态拆开；是否每个模态均完整对齐必须核验。 | `intermediate_heter_pair_fusion_dataset.__getitem__:480-507`、`collate_batch_train:709-875`；`heter_nego_train.forward:291-343,500-580` 的各模态特征列表、negotiator、公共特征、cycle 与标签对齐；变长 CAV、缺模态、不同场景顺序需逐项验证。 |
| Alliance FT | `intermediateheter` + `heter_nego_inf` + `nego_loss_ft`。联盟训练 ego 可因样本而不同。 | `heter_nego_inf.forward:127,154-169,192-223` 以 `agent_modality_list[0]` 为所有样本选 receiver、融合和检测头；`nego_loss_ft.forward:51` 用首个 ego 分类记录。必须使用每样本 ego、索引、label 和 loss 归约。 |
| m3 Alignment | `intermediateheterpair` + `heter_nego_train_align` + `nego_loss_align`；当前 `agent_modality_list[0]` 所在位置为注释，不能误报为执行代码。 | `heter_nego_train_align.forward:248-344` 按模态生成公共表示并 `stack/mean`，需证明同一索引为同一样本同一 CAV；检查不同 CAV 数、无 voxel SECOND 特征和变换对应。 |
| m3 FT | `intermediateheter` + `heter_nego_inf` + `nego_loss_ft`；与 Alliance FT 共用首 ego 风险，m3 的 SECOND 稀疏索引另需处理。 | 对 receiver、模态特征分组、`record_len_modality`、检测头和 GT 逐样本路由；不能假定某模态总在每个样本出现。 |
| m4 Alignment / FT | 路径分别同 m3 Alignment / FT，新增 m4 相机特征和标定。 | 检查 m4 相机输出与同一样本/CAV 的 LiDAR、联盟特征和空间变换配对；FT 需同样处理 per-ego 路由。 |
| Final Inference | `run_nego_doma.py:312-350` 使用 assembled checkpoint，Test 通过 `validate_dir=test_dir` 和 `evaluate` 逐帧运行；CAV1 使用 HomoCollab。 | `loader:44` 将验证/测试固定为 1，两个 collate 的 `collate_batch_test` 分别在 `intermediate_heter_fusion_dataset.py:738`、`intermediate_heter_pair_fusion_dataset.py:899` 限制长度并读取 `batch[0]`。保留逐帧 Test、GT、NMS、AP、CAV 前缀和结果 ledger。 |

普通异构 collate `intermediate_heter_fusion_dataset.py:611-688` 将场景成员展平，`record_len[b]` 是第 b 帧 CAV 数；前缀和提供场景区间，`record_len_modality[m][b]` 表示该场景 m 模态 CAV 数。`agent_modality_list` 属于所有 CAV 的拼接，不能将第 0 项作为整个 batch 的 ego。普通 collate 在 `:691-699` 仅保存 `object_ids[0]`；m1 新字段保存逐样本 object IDs，但其他阶段仍需独立审计。

配对 collate 在 `intermediate_heter_pair_fusion_dataset.py:796` 用前一循环留下的最后一个 `ego_dict` 的 targets 决定模态键，再在 `:810-825` 对其他样本按该键访问；目标键集合不一致会遗漏或报错，`:868-870` 暴露的是最后一个模态的 `object_bbx_mask`，而不是所有目标模态的显式汇总。需审计空样本和标签选择，不能用最后一个模态的 mask 代替整个 batch 的真实语义。

`heter_nego_inf.py:58-67` 按模态计数重组特征；`comm_in_pub.py:215-258` 对 `record_len` 逐样本变换，但 receiver 已由调用方根据首个 ego 选定，且用 `local_feature[b][0]` 取提示。因此不能单凭 comm 内部存在样本循环就判定混合 ego 安全。`heter_nego_train.py:353,360-371` 的 FT 分支还以全局 `cav_id` 索引按模态存储的特征，必须核查该分支是否实际被未来路径使用及其索引契约；当前正式 FT 使用 `heter_nego_inf`。

## 4. 配置限制与运行风险

- `nego_protocol.validate_hyperparameters:328-350` 校验正整数、有限正学习率、非负 weight decay、optimizer 类型及 scheduler 类型；不是固定本次实验值的训练协议。`:353-366` 的训练 batch=1 是**模型安全保护**，不是已解决的超参数问题。`eval_batch_size` 虽可作为合法配置解析，运行时仍拒绝非 1，且 `loader` 实际不读取该值；这与未来 YAML 自由驱动目标存在差距。
- `generate_nego_doma_configs.py:161-162` 将生成的默认 batch size 设置为 1、workers 设置为 0；生成器 `put()` 拒绝覆盖不同内容，而非安全地修改已有配置。生成默认值与训练运行时固定值须分开处理，不能直接对已有原始 YAML 重跑生成器。
- `run_nego_doma.py:208-223` 正常 batch 是一次 forward、一次 backward、一次 optimizer step；目前只检查 loss 是否有限，没有在 `optimizer.step()` 前全面检查所有梯度/参数是否有限，不能满足发生梯度异常即停止的要求。
- `run_nego_doma.py:211-212` 当**整个 batch** 的 `object_bbx_mask.sum()==0` 时直接跳过；有正 GT 的混合 batch 中空 GT 样本仍会进入 loss。batch size 1 的空 GT 会被跳过，分组变化可能改变参与训练的数据和背景损失。用户要求不得通过跳过样本规避问题，需先确认明确规则并设计新模式；默认原生路径如何兼容需单独定义。
- `run_nego_doma.py:185,221-223` scheduler 构造传入 `len(loader)`，但 TensorBoard 和 cosine `step_update` 使用 `epoch * len(train_data) + index`。batch 大于 1 时更新索引与每 epoch batch 数不一致；跳过全空 batch 也导致更新计数与循环索引不同，可能错误推进 cosine/warmup。应区分样本数、batch 数与真实 optimizer 更新数，明确 resume 和旧 batch=1 兼容策略后实现；不可隐式修改学习率或训练预算。
- `nego_state.py:116-135,165-201,222-233` 校验完整配置哈希、代码/协议哈希、父阶段身份、权重和训练状态；多 batch 实验要保留完整实际配置及来源，避免将旧配置 checkpoint 伪装成新配置。恢复时如何兼容代码版本与训练超参数变化应按现有严格约束显式设计。

## 5. 日志现状

`run_nego_doma.train:159-160` 创建 `stage_dir/train.log` 和 StreamHandler，`:254` 在验证频率命中时写 epoch、平均训练 loss、验证 metric 和 best。训练 batch 循环仅向 TensorBoard 写 `train/loss`（`:221`），没有逐 batch 的文本进度、loss、耗时、samples/s 或实时显存；也没有固定的 checkpoint 成功日志。日志可能因 root logger 已配置导致 `logging.basicConfig` 不接管而为空，需实际环境验证；仅凭 `-u`/`tee` 无法产生代码从未输出的逐 batch 文本。

建议未来增加可配置日志频率、明确独立 logger/handler 和 flush；输出阶段、epoch、batch/总数、实际场景与 CAV 数、loss、数据/计算耗时、验证与 checkpoint 成功/失败，并保留完整异常。日志不能改变梯度、顺序或训练参数。TensorBoard 的 global step 应与实际更新语义一致。默认不开启高频日志对已有性能的影响需评估。

## 6. 推荐统一实现架构（待授权）

采用 YAML 中默认关闭的阶段能力开关，**不是用 batch 大小猜测是否启用功能**。协议检查验证参数合法性和锁定的数据/评价协议；独立运行时能力检查按照已实现并通过验证的阶段与模式放行，其余阶段继续 fail closed。每个阶段的 batch size、学习率、epoch、optimizer、scheduler、workers 仍由该阶段 YAML 指定；不把 1/2/4/8 等具体实验值写成训练超参数白名单。逐帧评估若暂只支持 1，应明确作为评价路径兼容性限制，避免虚称 `eval_batch_size` 已生效。

共用一个 batch 元数据契约：`record_len[B]`、CAV 前缀和/场景索引、模态内前缀和、每场景 ego 模态与 CAV 身份、与输入/标签/坐标的双向映射；校验 `sum(record_len)`、单模态记录和实际张量条数。m1 直接复用现有 `collate_m1_vectorized` 与 scatter 分支，避免无理由重写。对普通和配对 collate 保留默认分支，新增开关控制的安全信息；不得变更数据、CAV 筛选、相机排除和协议锁。

向量化模型可先按模态编码，再根据明确索引/样本边界重组；FT 以每场景 ego 模态分组，调用相应 receiver、融合和检测头，并恢复样本顺序。必须处理某模态在某场景或整个 batch 缺席，确保缓存的 `local_feature`、`record_len_modality`、配对变换及梯度来自正确样本。Negotiation/Alignment 则以 `(sample_id, cav_id, modality)` 做配对并验证 stack 的身份与形状；不能直接把第一个样本的信息广播。对不满足完整配对的情况是报错还是有原生允许的掩码，应先查明协议并由用户确认，不可自动丢 CAV。

## 7. Loss、空 GT 与兼容性

`point_pillar_loss.py:43-92` 通常从目标 batch 维取得 B，按每样本正 anchor 数归一化，再对分类/回归/方向项除以 B；`point_pillar_pyramid_loss.py:46-103` 的 `_single` 占用监督使用单车标签的数量 `sum(record_len)` 作为归约分母。各场景 CAV 数不同，单车项天然按 CAV 而不是按场景等权，不能简单声称等同于平均多个独立场景 loss。`nego_loss.py`、`nego_loss_align.py` 的 cycle/对齐项有 sum 和 mean 混用（例如各文件 `:20-35,105-106`），需要逐项推导其样本、CAV、目标和模态维度及权重，给出可测的数学公式，禁止为追求吞吐随意改权。

推荐把有 GT 与无 GT 场景都视为有效输入，使用明确的负 anchor/空 GT 语义，并对 B 与 CAV 的分母作可复核定义；不过这会与现有 batch=1 的全空跳过行为不同，**不能声称与原生训练等价**。必须先由用户确定空 GT 的新模式规则、全空 batch 是否更新以及与 BN/随机数/scheduler 的关系；默认原生 batch=1 路径继续保持原行为。缺失数据、错误样本和非有限值应报错而非跳过。

开关关闭时比较完整输入、模型输出、逐项 loss、梯度、参数更新、优化器状态、BN/buffer、RNG、scheduler、checkpoint 内容及评价结果；固定随机状态并说明容差。开关打开、batch=1 也应与原生路径分别对照。多 batch 正常 BN 统计与随机数轨迹可不同，不应宣称训练轨迹与多个单 batch 完全一致。

## 8. 测试计划

1. CPU 和静态：参数合法性/默认关闭/非法组合；Python 3.8 语法；`record_len` 和模态前缀和；CAV 数不同、缺模态、同一场景配对、样本顺序变化、变长 voxel（含首/中/尾空 CAV 与全空）、坐标和 object IDs；m1 既有测试保留并扩充，检查样本替换不会污染其他样本。
2. 每阶段 batch=1：真实 Dataset/模型完整 forward、各 loss、backward 和 optimizer step 与原生路径对照；覆盖 train/eval、BN、冻结参数、随机数和 resume。Final Inference 保持逐帧，比较逐帧 GT、预测、CAV 前缀及 AP。
3. 每个阶段真实 batch=2、目标 batch 4 或 8（并测试不完整尾 batch）：V2XSet 真数据、正式 Dataset/collate/model/loss，核对记录的 scene/CAV ID、模态、ego、坐标、输出与标签；进行一次真正向量化 forward/backward/optimizer step，检查所有梯度有限、预期参数更新、冻结状态不变。不能用逐样本梯度累积冒充向量化。
4. Loss：独立数值 oracle 和梯度检查，覆盖不同样本 GT 数、CAV 数、空 GT 混合/全空、不同模态；逐项对齐 scene 与 CAV 分母，检查样本重排与 loss 加权。固定 eval 模式下比较单场景与 batch 输出；训练模式中单独解释 BN 差异。
5. 阶段依赖：对 Alliance、Alignment 和 FT 使用同一实验的合法父 checkpoint 做严格来源/哈希检查；预训练未完成时，使用明确标记的合成测试状态只能证明维度契约，不能宣称真实训练链验证成功。
6. 服务器资源：核对物理 GPU UUID、Python/PyTorch/CUDA/spconv，分别记录实际 batch、CAV/voxel 数、峰值 allocated/reserved、单步及含加载端到端吞吐、显存与采样利用率。OOM、非有限梯度、数据异常均停止并保留配置、进度和 traceback；不自动调整参数或重试。
7. 测试失败优先保持实验状态与安全保护；只有通过相关阶段全部验证后才逐阶段开放该阶段的开关。不要通过修改最终 Test 协议、额外过滤数据或错误使用 Validation 指标来获得通过结果。

## 9. 尚需确认

1. 新模式的空 GT 策略（全空是否更新、混合 batch 背景项与归约）及与默认 batch=1 原生跳过行为的关系。
2. 各 Negotiation/Alignment loss 项在 scene、CAV、模态和 GT 上的权重定义，以及变长配对是否完全由原生数据契约保证。
3. mixed-ego FT 按模态分组执行时，BN/Dropout、输出重排与单样本模型的等价边界；不同分组是否会改变统计，是否符合实验目标。
4. cosine/warmup 在跳过 batch、尾 batch、恢复训练时的真实 optimizer 更新计数和历史 checkpoint 兼容方案。
5. `eval_batch_size` 是继续明确固定 1 的评价兼容性约束，还是另行设计不改变逐帧 GT/AP 的批量评估；当前配置字段并未真正控制评估 DataLoader。
6. 全空 voxel、单 voxel、相机缺失或极端 CAV 场景在原生算子中的合法性与应有失败语义。
7. m1 服务器脚本的 `train.log`、GPU 利用率采样、完整 BN 状态和数据加载耗时验证如何增强；服务器真实结果目前不可用。

本审查没有修改任何源代码、原始配置或现有实验产物；未安装依赖、连接服务器、启动训练、提交或推送。后续模型核心改造须另行授权。
