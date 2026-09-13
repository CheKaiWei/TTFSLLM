# TTFS-LM MVP Reviewer 质疑与修正路线

> 审计对象：commit `3cb681c190c5b82ba0d74f9e03291f4794631a5e`
> 评审角色：熟悉语言模型、ANN-to-SNN、量化与系统评测的严格 reviewer
> 目标：提前暴露拒稿风险，并把每条质疑转换成可执行、可验收的修正任务。

方法细节与逐模块转换见 [`CONTINUOUS_TTFS_METHOD.md`](CONTINUOUS_TTFS_METHOD.md)。
MVP 与跨语料实验的完整证据、失败分析和 claim-evidence map 见
[`QWEN2_5_TTFS_EXPERIMENT_RESULTS.md`](QWEN2_5_TTFS_EXPERIMENT_RESULTS.md)。

## 1. Reviewer 总体判断

### 1.1 如果把它当工程 MVP

结论是 **通过**。当前实现已经完成一个边界清晰的最小闭环：

1. 官方 Qwen2.5-0.5B eager FP32 与本地 ANN logits 完全一致；
2. WikiText-2 train-only 校准、完整 validation PPL 和数据指纹可运行；
3. Continuous TTFS 在 262,419 个目标上 PPL 近似不变、top-1 100%、零越界；
4. 因果 mask、padding sideband、TTFS KV cache 和 greedy generation 形成闭环；
5. `P=4096` 的失败被完整报告，没有被包装成成功结果。

### 1.2 如果把它当论文主要方法

当前判断是 **Weak Reject / Reject**。主要原因不是代码不完整，而是论文核心命题还没有成立：

1. Continuous 结果主要来自 affine 编码的可逆性，方法增量可能被认为是“恒等包装”；
2. 真正引入时间离散性的 `P=4096` 质量完全崩溃；
3. 所有核心计算仍在解码后由 dense PyTorch 完成，无法支持 SNN 系统收益；
4. 只有一个小模型、一个数据集、一个短上下文和一个生成 prompt；
5. 没有量化基线、脉冲 Transformer 基线、硬件成本或事件覆盖率；
6. 当前运行证据没有被 commit `3cb681c` 干净绑定和发布。

换言之，这个 commit 是好的 **研究基础设施**，但还不是足以支撑论文主张的 **研究结果**。

## 2. 评分卡

| 维度 | 当前评分 | Reviewer 判断 |
| --- | ---: | --- |
| 工程完整性 | 8/10 | MVP 入口、配置、测试、失败产物和文档较完整 |
| 数值正确性 | 8/10 | Continuous 与官方 ANN 证据强，但作用域很窄 |
| 方法新颖性 | 3/10 | affine decode-compute-encode 的新意不足 |
| 有限精度有效性 | 1/10 | `P=4096` 严重失败 |
| 实验充分性 | 2/10 | 单模型、单数据集、短上下文、单 prompt |
| 系统/能效证据 | 0/10 | 无 event-native backend 和成本测量 |
| 可复现性 | 6/10 | 配置与指纹齐全，但产物未随 commit 发布且 run 非干净 commit |
| 论文可接受性 | 3/10 | 目前更像 feasibility baseline，而非完成的方法论文 |

## 3. 风险分级

| 级别 | 含义 |
| --- | --- |
| Blocker | 不解决就不能形成核心论文结论 |
| Major | 会显著降低可信度或导致拒稿 |
| Moderate | 不一定单独拒稿，但会形成集中质疑 |
| Minor | 工程或写作完整性问题 |

## 4. Blocker 级质疑

### R1. Continuous TTFS 是否只是恒等映射包装？

**Reviewer 质疑。** 你的模块定义是：

\[
S_f(t)=E(f(D(t))).
\]

当 `E` 与 `D` 可逆且没有时间量化时，`D(S_f(E(x)))=f(x)` 几乎由构造直接成立。PPL 不变是否只是一个预期的 sanity check，而不是方法贡献？

**严重度：Blocker。**

**当前证据。** Continuous PPL 相对变化 `7.91e-9`、top-1 100%、零 overflow。这证明实现没有破坏 Qwen 语义，但没有证明时间编码本身带来新的能力或收益。

**当前可以如何回答。**

> Continuous 结果的定位是转换正确性基线，而不是性能结果。它验证因果 mask、现代 Qwen 算子和 stateful KV cache 能统一接入 TTFS 表示合同，并为有限精度误差定位提供参考上界。

**必须如何修正。**

1. 在摘要和贡献中把 Continuous 降级为 correctness prerequisite；
2. 将主要研究问题改为有限精度、事件执行或状态存储，而不是“Continuous 与 ANN 等价”；
3. 报告每层 encode/decode 边界数和误差传播，不只给最终 PPL；
4. 增加边界消融：只编码 residual、只编码 K/V、只编码 block 输出、全部编码；
5. 证明某个非平凡设计在相同时间位宽下优于直接 uniform activation quantization。

**退出条件。** 至少有一个非恒等、可测量的核心结果，例如：在相同 bit/time budget 下优于 INT/naive TTFS 基线，或 event-native kernel 在相同质量下展示明确系统收益。

### R2. 为什么称为 SNN/TTFS，而所有计算仍是 dense ANN？

**Reviewer 质疑。** `TTFSQwenForCausalLM` 的 Linear、matmul、Softmax、RMSNorm、SiLU 和 residual 都先解码为实值，再调用 dense PyTorch。`SpikeTensor.time` 也是与激活同形状的稠密 tensor。模型没有事件列表、发放调度、膜动力学、事件路由或稀疏执行。为什么这不是 activation recoding simulator？

**严重度：Blocker。**

**当前证据。** 代码 docstring 已明确是 `decode-compute-encode` functional simulator。Hybrid LM Head 和 embedding 也保持 dense。

**当前可以如何回答。**

> 当前版本只验证表示和算子语义，不宣称 event-native、低功耗或低延迟。TTFS 指的是激活接口使用单时间码，而不是当前 GPU kernel 的执行方式。

**必须如何修正。**

1. 定义 functional backend 与 event backend 的统一接口；
2. 至少实现 Linear、QK、AV、residual 和 KV read/write 的 event-native 或可信周期模型；
3. 明确一个 time code 对应的发放方向、时间窗、同步屏障和无事件语义；
4. 统计每层 scalar、spike、zero-event、同步窗口和 encode/decode 次数；
5. 报告 compute-weighted event-native coverage；
6. 只有覆盖核心 block 计算量的 80% 以上，才讨论端到端 SNN 系统收益。

**退出条件。** 不再依赖逐算子完整 dense 恢复，或者论文标题与主张明确限定为“functional TTFS recoding analysis”。

### R3. `P=4096` 已经失败，方法的实际价值在哪里？

**Reviewer 质疑。** 4096 个时间间隔已经相当高，但 PPL 从 29.26 变成 `7.23e106`，top-1 只有 0.1528%，约 0.8997% 编码元素越界裁剪。既然有限时间实现完全不可用，Continuous 结果有什么实际意义？

**严重度：Blocker。**

**当前证据。** finite logits max-abs 470.97、mean KL 243.35，涉及 142 类 violating ranges。RMSNorm variance、residual、attention 与 MLP 后段出现级联漂移。

**必须如何修正。** 先定位，后优化，禁止直接堆训练：

1. 对每层输出记录 cosine、NRMSE、max-abs、SNR、overflow；
2. 执行 operator-bypass ablation：某一类算子保持 Continuous，其余使用 `P=4096`；
3. 执行 progressive quantization：从最后一层向前、从第一层向后逐层开启 finite；
4. 分离 representation error、operator error、range overflow 和 autoregressive accumulation；
5. 对 `P={256,512,1024,2048,4096,8192,16384}` 扫描，找到误差拐点；
6. 对 RMSNorm 考虑 fused/factorized range、log-domain inv-rms 或保留更高精度；
7. 对 Softmax 概率使用 simplex-preserving quantization或量化后重归一化；
8. 减少不必要的编码边界，比较 block-output-only 与 every-intermediate；
9. 再引入 per-head/per-channel range、mixed precision、range optimization 或 LoRA/KD。

**退出条件。** 至少一个明确 bit/time budget 在 held-out test 上达到预注册质量门槛；若做不到，论文应转为“有限 TTFS 在 LLM 中为何失败”的机制分析，而不是成功转换论文。

### R4. 没有系统收益证据，为什么 TTFS 比 INT8/INT4 更值得？

**Reviewer 质疑。** 当前 time code 仍以 FP32/FP64 dense tensor 保存，计算仍是 dense matmul。与成熟的 INT8/INT4 相比，TTFS 增加了范围工件、同步和编码边界，却没有展示质量、速度、存储或能耗优势。

**严重度：Blocker。**

**必须如何修正。**

1. 在相同模型、tokenizer、context、batch 和任务上加入 BF16、FP16、INT8、INT4；
2. 把 `P` 转换为明确的时间位宽与硬件表示成本；
3. 同时统计 weights、activations、KV cache、LM Head、embedding 和控制 sideband；
4. 报告 TTFT、decode tokens/s、p50/p95 latency、bytes/token 和能耗；
5. 将 GPU functional simulation 时间明确排除出神经形态收益结论；
6. 证明任何收益是在相同质量下取得，而不是用崩溃的 PPL 换成本。

**退出条件。** 在至少一个可信 backend 上，TTFS 在相同质量条件下优于一个浮点和一个量化基线的明确成本维度。

## 5. Major 级质疑

### R5. 校准集是否太小、太窄，且使用 validation 反馈调参？

**Reviewer 质疑。** 只有 WikiText-2 train 的前 100K token、context 128，decode 统计只有第一个 batch。开发过程中 5% margin 在 validation 越界后改成 100%，这意味着 validation 虽未直接进入 RangeSpec，也间接参与了超参数选择。

**严重度：Major。**

**当前可以如何回答。** validation activations 没有加入校准统计，因此没有直接数据泄漏；但 margin 选择确实使用了 validation 反馈，不能把 validation 结果当作最终无偏测试结果。

**必须如何修正。**

1. 固定三份数据：calibration、development、held-out test；
2. 只在 development 选择 margin、precision 和 range granularity；
3. test 只运行一次主要配置；
4. 校准扩展到 5M/20M token，覆盖中英文、代码、数学和长文本；
5. 分离 prefill/decode，增加长度桶和多个 seed；
6. 报告校准规模曲线与 range stability，而不是只报告最终范围。

**退出条件。** 主要结论来自未用于任何设计选择的 test/OOD split，并对校准 seed、规模和领域稳定。

### R6. WikiText-2 PPL 协议是否标准、是否可与文献比较？

**Reviewer 质疑。** 当前 `pack_texts` 对每个非空原始行执行 `strip()` 并追加 EOS，然后把行连接成 token 流。WikiText raw 的一行不一定等于一个文档；标题和段落会被频繁插入 EOS。PPL 29.26 可能不是标准 WikiText-2 protocol，不能直接和论文数值比较。

**严重度：Major。**

**必须如何修正。**

1. 实现并命名至少两种协议：`line_eos_mvp` 与 `standard_document_concat`；
2. 按空行或数据集文档结构恢复文档边界，而不是每个非空行都当文档；
3. 与 Hugging Face 官方模型在同一 token stream 上交叉验证 NLL；
4. 报告 tokenization、BOS/EOS、stride、overlap、padding 和目标计数；
5. 主要论文使用 WikiText test，不只用 validation；
6. 不将当前 PPL 与采用不同 tokenizer/protocol 的论文直接横向比较。

**退出条件。** 独立评测脚本可复现标准协议，且 ANN baseline 与可信参考实现一致。

### R7. RangeSpec 太粗，无法支撑长上下文和真实 decode

**Reviewer 质疑。** 当前每个语义角色只有一个 tensor-wise `[lower,upper]`。RangeSpec 不包含 phase、length bucket、head/channel granularity、precision、模型权重 hash 或 calibration artifact hash。字符串名称不足以保证 cache 生命周期安全。

**严重度：Major。**

**必须如何修正。**

扩展 RangeSpec：

```python
RangeSpec(
    name,
    lower,
    upper,
    phase,          # prefill/decode/shared
    length_bucket,
    granularity,    # tensor/head/channel/group
    precision,
    model_hash,
    tokenizer_hash,
    calibration_hash,
    version,
)
```

同时实现 streaming quantile/t-digest、分布式合并、cache range version 检查和越界 token 定位。

**退出条件。** 不同长度、phase、seed 和领域下的 overflow/NLL 稳定；cache 无法混用不兼容 range artifact。

### R8. 量化 Softmax 后概率不再归一，是否是设计缺陷？

**Reviewer 质疑。** `P=4096` 对每个概率独立 rounding。解码后的概率和不再严格为 1，随后直接用于 `probabilities @ value`。这会把概率量化误差转化为 context scale 漂移，并在 residual/Norm 中放大。

**严重度：Major。**

**必须如何修正。** 对比至少四种方案：

1. 独立 uniform time rounding；
2. rounding 后在有效集合重新归一化；
3. 保总和的 simplex quantization / largest-remainder 分配；
4. logits finite、Softmax 保持高精度；
5. event-native valid-set Softmax。

报告每行 probability-sum error、KL、context cosine 和最终 PPL。

**退出条件。** 选择依据是 held-out NLL 与系统可实现性，而不是单节点 MSE。

### R9. 编码边界是否过多，导致人为误差累积？

**Reviewer 质疑。** 每层 26 个边界意味着 24 层中反复量化数百次。RMSNorm 内部五次、attention 九次、MLP 五次。这样的边界密度是否由硬件要求决定，还是为了监控方便而引入？

**严重度：Major。**

**必须如何修正。**

1. 区分 observation point 与 physical emission point；
2. 实现边界策略：every-op、submodule-output、residual-only、cache-only；
3. 对每种策略报告时间码数量、误差、同步次数和后端覆盖；
4. 对可融合链路设计共享 range 或 fused kernel；
5. 只在有物理发放必要的位置调用 `emit/quantize`。

**退出条件。** 边界选择有消融与硬件理由，且有限精度质量不再被无必要的重复 rounding 主导。

### R10. Hybrid embedding/LM Head 隐藏了主要系统成本

**Reviewer 质疑。** token embedding 是 dense lookup，LM Head 是 `896 x 151936` 的 dense vocabulary projection。decode 时 LM Head 很可能是主要计算/访存之一。若它们不在 TTFS backend 中，能否称端到端 TTFS-LM？

**严重度：Major。**

**当前可以如何回答。** 当前明确称 Hybrid Head 和功能基线，不做端到端收益声明。

**必须如何修正。**

1. 分开报告 Hybrid 与 Full 配置；
2. 统计 embedding/LM Head 参数、bytes 和 latency 占比；
3. 实现完整词表 TTFS Linear 或诚实保留 mixed backend；
4. 若做候选词裁剪，必须加入召回率和质量评估；
5. tied weight 的存储形式只能保留一份，不能因两种 backend 隐式复制。

**退出条件。** 系统表包含全部输入、block、cache 和输出成本，不再把外围 dense 模块排除后称“端到端”。

### R11. Cache 与生成验证太弱

**Reviewer 质疑。** 官方模型只验证一个 8-token prompt、batch size 1、8 个 greedy token。没有左/右 padding、混合长度、长 prefill、长 decode、cache crop、beam reorder、动态 batch 结束或采样。

**严重度：Major。**

**必须如何修正。**

1. 官方模型覆盖 context 1/8/128/512/1024/2048；
2. 覆盖 batch 1/4/8、左/右 padding 和混合长度；
3. 比较 prefill + decode、纯逐 token、no-cache 三条路径；
4. 加入 cache reorder/crop/beam 与 position offset；
5. 建立 100-500 个固定 prompt，报告 greedy prefix survival；
6. 采样生成固定 seed，报告分叉位置、任务质量和重复率；
7. 检查同一 cache 生命周期的 RangeSpec hash。

**退出条件。** cache/no-cache 在所有支持长度和 batch 形态下达到预定义数值合同。

### R12. 当前实验产物没有真正绑定 commit `3cb681c`

**Reviewer 质疑。** `TTFS_LM/artifacts/` 被 `.gitignore` 排除。现有本地产物的 manifest 记录的是父提交 `535a869` 且 `dirty=true`，因为完整运行发生在提交 `3cb681c` 之前。论文中的结果如何由当前 commit 独立复现？

**严重度：Major。**

**必须如何修正。**

1. 在干净 clone 的 `3cb681c` 或后续 release commit 上重新运行；
2. 将 `metrics.json`、`run_manifest.json` 和 range 摘要纳入 release artifact；
3. 大型 `ranges.json` 可使用 Git LFS、release asset 或对象存储，但仓库需保存 SHA256；
4. manifest 记录 code tree hash、config hash、model file hashes、dataset/token hashes 和命令；
5. CI 验证 artifact hash 与声明结果；
6. 不以 README/TODO 手工抄写数值替代原始证据。

**退出条件。** 第三方从一个干净 commit 和公开工件可复现主要表格，manifest 显示 `dirty=false`。

## 6. Moderate 级质疑

### R13. 时间方向和物理语义没有定义

当前 `t=(x-a)/(b-a)` 使更大的值对应更大的 time code。很多 TTFS 系统使用更强激活更早发放。Reviewer 会要求说明：`t` 是实际发放时刻、反向时钟、归一化顺序码，还是仅数值坐标？

**修正。** 在方法中定义 `T_fire=T_start+t*DeltaT` 或反向映射，说明 sign、zero、无事件、同步窗口和硬件读出；若没有物理实现，统一称 normalized TTFS code。

### R14. “一个标量一个脉冲”不等于稀疏或节能

每个 active scalar 都有一个 time 值，脉冲数可能等于激活元素数；attention pair tensor 甚至达到 `O(L^2)`。没有 zero-event 语义时，TTFS 可能只是固定一脉冲的高密度事件流。

**修正。** 报告 spikes/token、spikes/layer、zero-event ratio、attention pair events 和 routing volume；与 dense bit operations 和 memory traffic 比较。

### R15. 只有 min/max，为什么 guard band 是 100%？

100% margin 保证了 Continuous validation 零越界，但把有限时间步分辨率浪费在低概率尾部。Reviewer 会怀疑这是为通过 Continuous 验收而设置的特殊参数。

**修正。** 在 development 上联合搜索 percentile、margin、granularity 和 precision；画 coverage-resolution-PPL 三维曲线；主要配置预注册后再跑 test。

### R16. 模型内置 `labels` loss 对 padding 不安全

`QwenForCausalLM.forward(labels=...)` 和 TTFS 对应路径直接对所有 shifted labels 做 cross entropy，没有使用 attention mask 或 `-100`。MVP 的独立 evaluator 是正确 mask-aware 的，但公开模型 API 在 padded batch 上会计算错误 loss。

**修正。** 由 attention mask 构造 `-100` labels，或要求调用者显式传入 masked labels并验证；增加左/右 padding loss 测试。论文应以独立 evaluator 为准，直到 API 修复。

### R17. 依赖版本与模型文件没有完全锁定

pyproject 只限制主要版本范围；manifest 有 package version，但没有 CUDA driver、cuDNN、GPU UUID、模型 safetensors hash 或 tokenizer 文件 hash。

**修正。** 提供 lockfile/container、`pip freeze`、驱动和 GPU 信息、checkpoint 文件 SHA256、tokenizer 文件 SHA256、dataset script/hash。

### R18. 单个模型规模不足以支持 LLM 泛化结论

0.5B 只能证明最小可执行性，无法说明误差随深度、宽度和词表规模如何变化。

**修正。** 至少扩展 1.5B 和一个不同架构/规模；报告 range width、finite error、PPL 和 memory 随规模趋势。若资源有限，论文标题和结论限定到 Qwen2.5-0.5B case study。

### R19. 缺少相关工作与强基线定位

当前代码证明了实现，但没有回答相对 TTFSFormer、SpikingBERT、SpikeGPT、SpikeLLM、ANN quantization 和 activation quantization 的区别。

**修正。** 完成系统文献表，至少比较表示、训练需求、模型规模、attention、cache、finite precision、硬件证据和公开代码；只有明确空白后才使用“首次”或“新颖”表述。

## 7. Reviewer 会追问的实验矩阵

### 7.1 正确性矩阵

| 维度 | 必须覆盖 |
| --- | --- |
| dtype | FP32、BF16、Continuous |
| sequence | 1、8、128、512、1024、2048 |
| padding | none、left、right、mixed length |
| execution | no-cache、prefill+decode、token-by-token |
| state | cache append、crop、reorder、beam reorder |
| module | norm、QKV、RoPE、softmax、AV、MLP、residual、head |

### 7.2 有限精度矩阵

| 维度 | 必须覆盖 |
| --- | --- |
| precision | 256、512、1024、2048、4096、8192、16384 |
| range | min/max、percentile、symmetric/asymmetric |
| granularity | tensor、head、channel/group |
| phase | shared、prefill、decode |
| boundary | every-op、module-output、residual-only、cache-only |
| optimization | no training、range-only、LoRA/KD |

### 7.3 质量矩阵

| 类别 | 建议最小集合 |
| --- | --- |
| LM corpus | WikiText test + 一份更广域语料 |
| reasoning/common sense | HellaSwag、PIQA、ARC-E/C |
| completion | LAMBADA |
| language | 中文/英文各一份固定语料 |
| generation | 100-500 prompts，greedy 与固定 seed sampling |
| long context | 至少到 2048，稳定后再到模型上限 |

### 7.4 系统矩阵

| Baseline | 质量 | latency | energy | memory |
| --- | --- | --- | --- | --- |
| FP32/BF16 | 必须 | 必须 | 必须 | 必须 |
| INT8 | 必须 | 必须 | 必须 | 必须 |
| INT4 | 建议 | 必须 | 建议 | 必须 |
| Continuous functional | 正确性参考 | 只报仿真开销 | 不作硬件结论 | 报 dense time tensor |
| Finite functional | 必须 | 只报仿真开销 | 不作硬件结论 | 报 time bitwidth |
| Event backend | 必须 | 必须 | 必须 | 必须 |

## 8. 修正路线图

### P0：证据卫生与评测协议，预计 2-4 天

优先级最高，因为它不依赖新算法，却决定现有结果是否可信。

1. 在干净 commit 上重跑并发布原始产物；
2. 修复 padded `labels` loss；
3. 增加标准 WikiText document protocol 和 test split；
4. 将 line-EOS MVP protocol 明确改名；
5. 保存完整环境、权重和 tokenizer hash；
6. 增加 official-model cache/padding integration tests。

**P0 交付物：** 可由第三方复现的 ANN/Continuous 表格，manifest `dirty=false`。

### P1：有限精度失效定位，预计 1 周

1. 层级误差 trace；
2. operator bypass；
3. progressive finite conversion；
4. precision sweep；
5. probability-sum error；
6. range overflow 与 quantization error 分离；
7. 输出首个误差爆炸层和首个失效算子。

**P1 交付物：** 一份可证伪的 failure mechanism report，而不是只列最终 PPL。

### P2：表示与范围优化，预计 2-4 周

1. prefill/decode 与长度桶；
2. percentile、per-head/per-channel；
3. mixed precision；
4. simplex-preserving probability；
5. 减少 emission boundaries；
6. RMSNorm 与 residual 的高精度/fused 设计；
7. 必要时 range-only 或 LoRA/KD。

**P2 交付物：** 至少一个 held-out setting 达到预注册有限精度门槛，或形成明确的负结论。

### P3：评价完整性，预计 2-3 周

1. test/OOD corpus；
2. lm-evaluation-harness 任务；
3. 多 prompt 生成；
4. 长上下文/cache；
5. 多 seed；
6. Qwen2.5-1.5B 规模趋势；
7. BF16/INT8/INT4 公平基线。

**P3 交付物：** 质量表、消融表、失败表和规模趋势图。

### P4：事件后端与系统证据，预计 1-3 个月

1. event backend contract；
2. event Linear/QK/AV/residual/cache；
3. Full LM Head 或明确 mixed backend；
4. cycle/traffic model；
5. FPGA/神经形态平台验证；
6. 相同质量下的 latency/energy/memory comparison。

**P4 交付物：** 能支持系统论断的完整成本表和 event-native coverage。

## 9. 建议的论文故事

### 9.1 当前不建议的故事

> 我们将 Qwen2.5 无损转换为高效 TTFS SNN，并保持语言质量。

这句话同时缺少有限精度、事件执行和能效证据，容易被直接否定。

### 9.2 近期可成立的故事

> 我们建立了现代自回归 Transformer 的 TTFS 转换正确性框架，系统解决 causal/padding mask、RMSNorm、RoPE、GQA、SwiGLU 与 stateful KV cache 的表示合同，并揭示统一有限时间量化在深层语言模型中的级联失效机制。

这个故事仍需要 P1 的失效定位和相关工作验证，但比“Continuous 等价即成功 SNN”更诚实，也更可能产生新知识。

### 9.3 长期目标故事

> 我们提出一种 phase-aware、range-adaptive、event-native 的 TTFS LLM 转换，在相同质量下改善 KV 存储与自回归 decode 的系统成本。

这个故事必须由 P2-P4 的质量和硬件结果支持。

## 10. 可以说与不能说

### 当前可以说

1. 实现了 Qwen2.5-0.5B 的 TTFS functional conversion baseline；
2. Continuous 模式在当前 WikiText-2 validation protocol 上近似保持 ANN 数值；
3. mask-aware attention 和 TTFS KV cache 的最小正确性合同通过；
4. 统一 `P=4096` 在当前范围策略下严重失败；
5. 产物记录了逐角色 overflow 和误差。

### 当前不能说

1. lossless SNN language model；
2. 有限精度 TTFS 保持语言质量；
3. event-driven 或 neuromorphic execution；
4. energy-efficient、low-latency、memory-efficient；
5. 首个 TTFS LLM；
6. 对 LLM 或长上下文普遍有效；
7. 优于 INT8/INT4 或已有脉冲语言模型。

## 11. Claim-Evidence 修正表

| 潜在 claim | 当前证据 | Reviewer 状态 | 修正动作 |
| --- | --- | --- | --- |
| 官方模型被正确复刻 | logits max-abs 0 | pass | 在干净 commit 复现 |
| Continuous 表示保持语义 | PPL/top-1/KL/overflow | pass with narrow scope | 扩展 test/OOD/length |
| causal mask 正确 | 实现与单元测试 | pass at functional level | official padding matrix |
| TTFS cache 正确 | 单 prompt cache/no-cache | needs revision | 长度、batch、padding、reorder |
| finite TTFS 可用 | `P=4096` 崩溃 | fail | P1/P2 |
| 方法具有非平凡新意 | 无相关工作和消融 | needs evidence | 文献矩阵、边界/模块消融 |
| 系统更高效 | 无系统测量 | fail | event backend、完整成本 |
| 结果可复现 | 产物非当前干净 commit | needs revision | P0 artifact release |

## 12. 五维自审清单

### Contribution

- `needs revision`：当前新知识主要是适配与失败观察，尚未形成可推广机制。
- `needs new experiment`：需要证明至少一个设计优于 naive affine/quantization baseline。

### Writing clarity

- `pass`：只要全文坚持 functional、finite、event-native 三层术语。
- `needs revision`：禁止用 “lossless” 概括有限精度或任意输入。

### Experimental strength

- `pass`：官方 ANN 与 Continuous 数值对照强。
- `needs new experiment`：有限精度、下游任务、长上下文、规模和 baseline 均不足。

### Evaluation completeness

- `needs revision`：PPL protocol 需标准化；validation 不能兼作最终 test。
- `needs new experiment`：任务、生成、cache、OOD、seed 和系统矩阵均缺失。

### Method design soundness

- `pass`：Continuous affine 表示合同在范围内自洽。
- `needs revision`：Softmax simplex、RangeSpec 版本、边界密度和 padded loss。
- `needs new experiment`：有限精度稳定性与事件执行可实现性。

## 13. 最优先的十项具体 TODO

1. 在 clean commit 上重跑并发布 metrics/manifest/range SHA256；
2. 修复模型 `labels` 路径的 padding loss；
3. 实现标准 WikiText document packing 与 test PPL；
4. 输出 24 层逐层误差和首个爆炸层；
5. 完成 operator-bypass 与 progressive finite ablation；
6. 修复或消融 finite Softmax 的 probability-sum error；
7. 比较 emission boundary 策略，减少无必要 rounding；
8. 分离 prefill/decode range，并扩展校准数据；
9. 加入 BF16、INT8、INT4 公平基线；
10. 在任何能效表述前实现 event backend 或可信周期模型。

这十项中，1-7 是下一轮最关键工作。它们完成后，才能判断项目应继续走“有效有限精度转换”路线，还是转向“TTFS LLM 失败机制与设计原则”路线。
