# TTFS-LLM Development TODO

> 更新日期：2026-08-28
> 当前基线：`535a869 Add Qwen2.5 ANN-to-TTFS baseline`
> 当前阶段：MVP-0 最小闭环已通过；有限精度质量、正式扩展评测和事件后端尚未完成

## 当前判断

- 功能仿真完成度：较高，核心 Qwen 算子、TTFS 表示、mask 和 cache 已形成闭环。
- 研究完成度：约 30%，已有最小 PPL 基线，但仍缺少多数据集、下游任务、有限精度可用结果和充分消融。
- 系统完成度：不足 10%，当前仍是 decode-compute-encode 功能仿真器，没有事件原生能效证据。
- 当前最小闭环：官方 Qwen2.5-0.5B + WikiText-2 + TTFSFormer 转换已跑通。
- 下一优先级：定位 `P=4096` 的误差放大和范围漂移，建立真正可用的有限精度方案。

## 已完成的最高优先级：MVP-0 最小闭环

### 目标

先不扩展大数据集、下游任务或事件硬件，只证明以下最小链路可以一条命令稳定跑通：

```text
Qwen2.5-0.5B 官方权重
  -> WikiText-2 train 校准
  -> RangeSpec 工件
  -> Continuous TTFS 转换
  -> WikiText-2 validation PPL
  -> 一次有限精度 smoke run
  -> 一次带 TTFS KV cache 的最小生成
```

### 固定实验范围

- 模型：`Qwen/Qwen2.5-0.5B` Base，revision `060db6499f32faf8b98477b0a26969ef7d8b9987`。
- 数据集：`Salesforce/wikitext`，配置 `wikitext-2-raw-v1`。
- 校准数据：只使用 train split，先固定最多 100K 个 Qwen tokenizer token。
- 评测数据：只使用 validation split；test split 留到 MVP 通过后再启用。
- 上下文长度：固定 128。
- 文档处理：保留文档边界并插入 EOS，记录 packing 和 padding 规则。
- TTFS 模式：Continuous 为主验收；`P=4096` 只做有限精度机制 smoke test。
- 推理路径：先完成 no-cache PPL，再运行一个最小 cache generation 用例。
- LM Head：先使用 Hybrid Head；Full TTFS Head 作为附加开关，不阻塞 MVP。

### MVP-0 TODO

- [x] 在 `TTFS_LM` 增加 `datasets` 依赖并锁定兼容版本。
- [x] 实现 WikiText-2 下载、缓存和 revision 记录。
- [x] 实现空文档过滤、EOS 插入、tokenize、packing 和 padding。
- [x] 明确 train/validation/test 的使用边界，禁止 validation/test 参与校准。
- [x] 固定 100K 个校准 token、上下文 128、batch size 和 seed。
- [x] 保存实际校准样本数、token 数和数据指纹。
- [x] 实现标准 causal-LM NLL/PPL 计算，正确处理 shift 和 padding mask。
- [x] 先运行官方 Hugging Face Qwen ANN 的 validation PPL。
- [x] 运行本地 Qwen ANN 的 validation PPL，并与官方参考比较。
- [x] 用 train calibration subset 生成 `ranges.json`。
- [x] 检查 `ranges.json` 包含模型、revision、tokenizer、数据和配置元数据。
- [x] 使用严格范围模式运行 Continuous TTFS validation PPL。
- [x] 输出 ANN/TTFS 的 NLL、PPL、top-1 agreement、KL 和最大 logits 误差。
- [x] 输出逐层 overflow/clip 计数；Continuous 主结果不允许静默 clip。
- [x] 使用 `P=4096` 跑一次完整 validation smoke test。
- [x] 为 `P=4096` 输出 PPL、top-1、overflow 和 clip，不预设质量达标。
- [x] 用固定短提示运行至少 8 个 token 的 ANN greedy generation。
- [x] 用同一提示运行至少 8 个 token 的 TTFS cache generation。
- [x] 比较 cache/no-cache logits、生成 token 和 cache 长度。
- [x] 新增 `ttfs-lm mvp` 或等价的一键入口。
- [x] 增加 `configs/mvp_wikitext2.yaml`，禁止关键参数散落在命令行默认值中。
- [x] 保存 `ranges.json`、`metrics.json`、`run_manifest.json` 和运行日志。
- [x] 在 README 写出从安装到复现结果的完整命令。
- [x] 增加一个可选的官方权重 MVP 集成测试。
- [x] 将 MVP 实测结果填写回本节，不只保留任务描述。

### MVP-0 通过标准

- [x] 一条命令可以从已缓存的模型和数据运行到最终指标，退出码为 0。
- [x] 官方 Hugging Face ANN 与本地 ANN 的 FP32 logits 最大误差不超过 `1e-6`。
- [x] Continuous TTFS 全流程无 NaN/Inf、无 clip、无未记录 overflow。
- [x] Continuous TTFS 相对本地 ANN 的 validation PPL 退化不超过 `0.1%`。
- [x] Continuous TTFS teacher-forcing top-1 agreement 不低于 `99.9%`。
- [x] TTFS cache/no-cache logits 达到既定容差，最小生成可完成至少 8 个 token。
- [x] `P=4096` 能完整运行并生成有限数值与完整误差报告。
- [x] 所有结果均绑定模型、数据、tokenizer、配置和 git commit。

### MVP-0 实测结果（2026-08-28）

- 运行状态：`passed`，9/9 验收项通过；完整离线命令耗时 358.35 秒。
- 数据：train 校准 100,000 tokens；完整 validation 262,420 tokens、262,419 个预测目标。
- 官方 ANN / 本地 ANN：PPL 均为 `29.258942438`，logits max-abs `0`，top-1 agreement `100%`。
- Continuous TTFS：PPL `29.258942670`，相对变化 `7.91e-9`，logits max-abs `5.82e-5`，
  top-1 agreement `100%`，mean KL `2.28e-10`，clip/overflow `0`。
- Cache/generation：ANN 与 TTFS cache/no-cache max-abs 均为 `1.16e-4`，8 个新 token 完成，
  ANN/TTFS token agreement `100%`。
- `P=4096` smoke：完整运行且数值有限，但 PPL 为 `7.23e106`，top-1 agreement 仅 `0.1528%`；
  212,311,813,548 个编码元素中有 1,910,180,057 个越界后裁剪（约 `0.8997%`），涉及 142 类范围。
- 结论：Continuous 转换闭环成立；`P=4096` 当前仅证明机制可执行，质量完全不可用，是下一阶段首要问题。
- 产物：`TTFS_LM/artifacts/mvp_wikitext2/{ranges.json,metrics.json,run_manifest.json,run.log}`。

### 暂不纳入 MVP-0

- WikiText-103、C4、下游零样本任务和 LongBench。
- `P=1024` 的正式质量门槛、mixed precision、LoRA 或知识蒸馏。
- Qwen2.5-1.5B 及更大模型。
- 事件原生 kernel、硬件时延和能耗结论。
- 与 INT8/INT4 或其他脉冲语言模型的正式比较。

MVP-0 已通过；后续按下方阶段门推进，优先解决有限精度误差，不提前形成能效结论。

## 已完成

- [x] 固定官方 `Qwen/Qwen2.5-0.5B` Base 权重与 revision。
- [x] 本地 Qwen ANN 与官方 FP32 logits 完全一致。
- [x] 实现 RMSNorm、RoPE、GQA、SwiGLU 和因果注意力。
- [x] 实现 mask-aware min/max 校准。
- [x] 实现 Continuous TTFS 编解码并与 ANN 完全对齐。
- [x] 实现 TTFS KV cache、prefill/decode 和 cache reorder。
- [x] 实现 Hybrid LM Head 和 Full TTFS LM Head。
- [x] 提供 CLI、快速测试、官方权重集成测试和开发文档。
- [x] 创建独立可安装的 `TTFS_LM` 子项目。

## P0：工程基线加固

- [ ] 将官方完整权重测试加入 GPU CI，不再只依赖环境变量手动运行。
- [ ] 固定 Python、PyTorch、Transformers、CUDA、GPU 和模型文件哈希。
- [ ] 增加左 padding、右 padding、混合长度、全 padding 和单 token 测试。
- [ ] 增加 batch/unbatched 等价测试。
- [ ] 增加 prefill、逐 token decode 和 cache/no-cache 等价测试。
- [ ] 增加 cache crop、batch reorder 和 beam reorder 测试。
- [ ] 增加随机序列长度和随机 mask 的 property-based 测试。
- [ ] 明确 FP32、BF16、continuous 三种数值合同和容差。
- [ ] 解释并量化 BF16 融合核与本地实现的逐层误差。
- [ ] 输出逐层 logits、hidden state、Q/K/V、MLP 和 residual 误差报告。
- [ ] 将表示误差、算子误差和 kernel 实现误差分开统计。
- [ ] 保存实验命令、git commit、环境、配置和原始结果 JSON。

退出条件：官方完整模型上的正确性不变量全部通过，并可在新环境用一条命令复现。

## P1：正式校准数据管线

- [ ] 实现 Hugging Face Dataset 和本地 JSONL 流式加载器。
- [x] 建立 100K-token smoke 校准集。
- [ ] 建立 5M-token MVP 校准集。
- [ ] 建立 20M-token final 校准集。
- [ ] 覆盖中英文、代码、数学、百科、新闻和长文本。
- [ ] 固定文档打包、EOS、padding、截断和随机种子。
- [ ] 分离 calibration、in-domain validation 和 OOD validation。
- [ ] 检查并阻止校准数据与测试数据泄漏。
- [ ] 保存数据集 revision、样本 ID、tokenizer 和 token 序列哈希。
- [ ] 分离 prefill 和 decode 激活统计。
- [ ] 增加 `1-128`、`129-512`、`513-1024`、`1025-2048` 长度桶。
- [ ] 实现 streaming quantile 或 t-digest。
- [ ] 支持 min/max、分位数、对称区间和非对称区间。
- [ ] 支持 tensor、head、channel 和 group 范围粒度。
- [ ] 扩展 `RangeSpec`：phase、length bucket、granularity、模型哈希和数据哈希。
- [ ] 输出逐层 overflow、clip、饱和率和异常 token 报告。
- [ ] 支持校准中断恢复和分布式统计合并。

退出条件：不同校准 seed、长度和领域下，范围、overflow 和验证集 NLL 稳定。

## P2：有限时间精度

- [ ] 扫描 `P={256,512,1024,2048,4096}`。
- [ ] 找到误差首次显著放大的层和算子。
- [ ] 分析 RMSNorm square/variance/inv-rms 的敏感度。
- [ ] 分析 Q/K/V、attention logits 和 softmax 的敏感度。
- [ ] 分析 residual 和 SwiGLU product 的敏感度。
- [ ] 比较不同区间估计方法对 NLL 的影响。
- [ ] 搜索安全 margin，并以验证集 NLL 而非局部 MSE 选取。
- [ ] 实现逐层 mixed precision。
- [ ] 实现 Q/K/V、norm 和 residual 的细粒度范围。
- [ ] 实现 range-only 优化。
- [ ] 增加 layer reconstruction objective。
- [ ] 若纯转换失败，实现 STE rounding。
- [ ] 实现 LoRA 校准感知微调。
- [ ] 加入 teacher KL、hidden reconstruction 和 overflow loss。
- [ ] 对 teacher-forcing 和自由生成分别评估误差累积。
- [ ] 报告失败配置、溢出位置和最低可用精度，不只报告最佳点。

退出条件：`P=1024` 的 PPL 相对退化不超过 3%，下游任务平均下降不超过 1 个百分点；否则给出有证据的最低可用精度结论。

## P3：正式质量评测

- [ ] 评测 WikiText-103 token NLL 和 PPL。
- [ ] 评测一份更广分布语料的 NLL 和 PPL。
- [ ] 报告 teacher-forcing top-1/top-5 agreement。
- [ ] 报告 logits KL、hidden cosine 和 NRMSE。
- [ ] 接入并固定 `lm-evaluation-harness` revision。
- [ ] 锁定任务 prompt 模板和 few-shot 设置。
- [ ] 评测 LAMBADA。
- [ ] 评测 HellaSwag 和 PIQA。
- [ ] 评测 WinoGrande。
- [ ] 评测 ARC-Easy 和 ARC-Challenge。
- [ ] 评测 BoolQ。
- [ ] 建立 100 至 500 个固定生成提示。
- [ ] 报告贪心前缀存活长度、重复率和输出长度分布。
- [ ] 对采样生成固定 seed，并区分数值分叉与质量变化。
- [ ] 每个校准策略至少运行 3 个 seed。
- [ ] 报告均值、标准差、最差值和置信区间。
- [ ] 建立 FP32、BF16、INT8 和 INT4 基线。
- [ ] 保证所有基线使用相同 tokenizer、上下文、batch 和生成配置。

退出条件：连续与有限精度模式都有完整 PPL、下游任务和生成结果，并能支持论文主要质量论断。

## P4：长上下文与生成完整性

- [ ] 验证上下文长度 128、512、1024、2048 和 4096。
- [ ] 检查越界率和误差是否随 token 位置增长。
- [ ] 验证 RoPE position、cache position 和 position offset。
- [ ] 固定同一 cache 生命周期中的 `RangeSpec`。
- [ ] 禁止新旧区间编码的 K/V 在同一 cache 中混用。
- [ ] 实现 cache crop 和最大上下文回收。
- [ ] 支持动态 batch 结束和 batch reorder。
- [ ] 支持 beam reorder。
- [ ] 支持固定 seed sampling。
- [ ] 比较 FP cache、TTFS cache 和事件 cache。
- [ ] 统计 KV cache 字节/token。
- [ ] 基础长上下文门槛通过后接入 LongBench。

退出条件：所有长度下 cache/no-cache 一致，长上下文质量、范围漂移和存储成本有完整报告。

## P5：事件驱动后端与系统能效

- [ ] 定义 functional backend 和 event backend 的统一接口。
- [ ] 实现事件原生 Linear。
- [ ] 实现事件原生 QK 和 AV matmul。
- [ ] 实现事件原生 residual 和 Hadamard product。
- [ ] 实现事件原生 RMSNorm。
- [ ] 实现 valid-set masked softmax 的事件后端。
- [ ] 避免 GQA K/V 的物化复制。
- [ ] 实现事件形式的 KV cache 追加、读取和重排。
- [ ] 实现 Full TTFS LM Head，不排除大词表投影。
- [ ] 评估 embedding table 的 TTFS 存储方案。
- [ ] 统计每层标量数、脉冲数、零事件率和时间位宽。
- [ ] 统计事件路由、同步窗口和 encode/decode 次数。
- [ ] 测量 prefill tokens/s 和 time to first token。
- [ ] 测量 decode tokens/s、p50 和 p95 延迟。
- [ ] 测量 batch size 1 和 8。
- [ ] 核算权重、激活和 KV cache 的 SRAM/DRAM 访问。
- [ ] 建立包含 compute、memory、routing、control 和 sync 的能耗模型。
- [ ] 将 GPU 功耗明确标记为功能仿真开销。
- [ ] 建立可信周期模型或数据流模型。
- [ ] 在 FPGA 或神经形态平台完成最终测量。
- [ ] 统计事件原生计算量加权覆盖率。
- [ ] 覆盖率达到 80% 后再讨论端到端 SNN 能效。
- [ ] 在相同质量下与 FP16 和 INT8 比较。

退出条件：完整系统成本已包含 embedding、LM Head、KV cache、存储和控制，且结果达到明确证据等级。

## P6：消融、规模扩展与论文

- [ ] 消融区间方法和范围粒度。
- [ ] 消融长度桶与 prefill/decode 分离。
- [ ] 消融 margin 和 uniform/mixed precision。
- [ ] 消融 Hybrid/Full TTFS Head。
- [ ] 消融 FP/TTFS/event KV cache。
- [ ] 比较无训练、range-only 和 LoRA/KD。
- [ ] 扩展到 Qwen2.5-1.5B，验证规模趋势。
- [ ] 条件允许时扩展到更大模型。
- [ ] 与 SpikingBERT、SpikeGPT 和 SpikeLLM 做技术定位。
- [ ] 只在模型、数据和任务可比时进行定量比较。
- [ ] 绘制方法总览图。
- [ ] 绘制校准流程图和 prefill/decode 时序图。
- [ ] 制作主要质量结果表。
- [ ] 制作精度、粒度和范围消融表。
- [ ] 制作系统成本与事件覆盖表。
- [ ] 建立每个主要论断的 claim-evidence 映射。
- [ ] 明确报告有限精度失败案例。
- [ ] 明确区分功能仿真、周期估算和硬件实测。
- [ ] 完成 contribution、clarity、experiments、evaluation、soundness 五维自审。
- [ ] 发布配置、范围工件、原始结果、环境和复现命令。

退出条件：每个主要论文论断都有直接实验支持，重要审稿质疑均有结果或明确限制回应。

## 总体决策门

- [x] G-MVP：Qwen2.5-0.5B、WikiText-2、Continuous TTFS、PPL 和 cache generation 最小闭环。
- [x] G0：官方 ANN 与 continuous TTFS 数值正确性。
- [ ] G1：正式校准语料和 PPL 评测。
- [ ] G2：有限精度质量达到预设门槛。
- [ ] G3：长上下文、生成和 KV cache 稳健性。
- [ ] G4：规模趋势和完整消融。
- [ ] G5：事件后端覆盖率和完整系统成本。
- [ ] G6：论文论断与实验一一对应。

## 执行顺序

严格按 `MVP-0 -> P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6` 推进。MVP-0 未通过前不并行展开其他研究任务；P1、P2、P3 完成前，不投入大规模事件硬件优化，也不形成端到端节能或有限精度质量结论。
