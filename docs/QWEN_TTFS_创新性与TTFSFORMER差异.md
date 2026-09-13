# Qwen2.5 TTFS 方案的创新性与 TTFSFormer 差异分析

> 分析基线：[`CONTINUOUS_TTFS_METHOD.md`](CONTINUOUS_TTFS_METHOD.md) 及当前 `TTFS_LM` 实现  
> 原版对照：ICML 2025 TTFSFormer 论文与本仓库视觉代码  
> 文献核查日期：2026-08-30  
> 目的：区分“工程上完成 Qwen 迁移”、“相对 TTFSFormer 的新增”和“足以支撑方法论文的研究创新”。

## 1. 结论先行

当前 Qwen2.5 TTFS 方案对原版 TTFSFormer 做了有意义的 **因果语言模型语义扩展**，但还没有形成足以独立支撑方法论文的核心算法创新。

目前最稳妥的定位是：

> **一个面向 Qwen2.5 自回归推理的 TTFS functional conversion 基线，它补全了 causal/padding mask、RMSNorm、GQA、RoPE、SwiGLU、KV cache 和生成评测的表示合同，并暴露了统一有限 TTFS 精度在深层 LLM 中的级联失效。**

当前贡献分为三层：

| 层级 | 当前内容 | 判断 |
| --- | --- | --- |
| 工程/基础设施 | typed `SpikeTensor`、mask-aware 校准、strict range telemetry、TTFS cache、PPL/KL/生成评测 | 已成立，完成度较高 |
| 相对 TTFSFormer 的方法扩展 | 因果 valid-set attention、GQA 和跨 token 时间码状态 | 代码级成立，但主要是语义扩展 |
| 绝对研究创新 | 有限精度稳定方法、事件后端、能效收益 | 尚未成立 |

如果现在直接投稿“我们把 Qwen 无损转换成 TTFS SNN”，reviewer 很容易给出 Weak Reject/Reject；如果将问题改为“为什么视觉上成立的 uniform TTFS 在自回归 LLM 中失效，并如何设计 causal-state-aware TTFS”，则有机会形成真正的研究故事。

## 2. 两个系统分别在解决什么

### 2.1 TTFSFormer 的问题设定

原版 TTFSFormer 面向 ViT/EVA 图像分类，主要问题是：

1. 如何扩展 TTFS 的实值表示范围，包括负值；
2. 如何用广义输入/输出核表示 GELU、SiLU、Softmax 和 LayerNorm；
3. 如何给出 Hadamard 与 MatMul 的 TTFS 构造；
4. 如何保持图像分类 Top-1/Top-5。

它的主要学术 claim 是“Transformer 的复杂非线性可被广义 TTFS 神经元表示”。对应论文见 [TTFSFormer, ICML 2025](https://proceedings.mlr.press/v267/zhao25n.html)。

### 2.2 Qwen TTFS 的问题设定

Qwen 方案面向 decoder-only causal LLM，除了单次 Transformer forward，还必须处理：

1. causal mask 与 padding mask 的组合；
2. query heads 与 KV heads 数量不同的 GQA；
3. 基于 position id 的 RoPE；
4. 不含去均值的 RMSNorm；
5. gate/up/product 路径的 SwiGLU；
6. prefill 与逐 token decode 之间持久化的 KV cache；
7. next-token PPL、cache 等价性和自回归生成。

因此，Qwen 方案的真正增量不是“又实现了一个 Transformer”，而是将 TTFS 表示从 **stateless encoder** 扩展到 **stateful autoregressive decoder**。

## 3. TTFSFormer 与 Qwen TTFS 的完整差异

| 维度 | 原版 TTFSFormer | 当前 Qwen TTFS | 创新性判断 |
| --- | --- | --- | --- |
| 任务 | ImageNet 图像分类 | next-token prediction 与 greedy generation | 新任务迁移，不是单独方法创新 |
| 主体架构 | ViT/EVA encoder | Qwen2.5 decoder-only Transformer | 应用域扩展 |
| Norm | LayerNorm | 五阶段 RMSNorm | 算子适配，本身不新 |
| Attention | 全双向 MHA | causal/padding GQA | 相对原版的明确新增 |
| Mask | ViT 无 mask；EVA 路径不适合 causal 语义 | bool valid-set Softmax，全无效行输出 0 | 语义修正；研究新颖性待证 |
| Position | 学习位置编码；EVA 可选 RoPE | Q/K RoPE + causal positions | RoPE 原版 EVA 已有；因果状态适配是增量 |
| MLP | GELU/GLU/SwiGLU | Qwen SwiGLU | SwiGLU 不能声称首次 |
| 时间数据类型 | 裸 `Tensor` + 外部 range tuple | `SpikeTensor(time, range_name, dtype)` | 类型合同与可审计性增强 |
| 范围注册 | `SpikeParamDict` | 带 metadata 的 `RangeRegistry` | 工程/可复现贡献 |
| 校准统计 | 平均 batch extrema | active-only streaming global extrema + guard band | 修正原版统计合同，但 min/max 算法不新 |
| 越界行为 | Continuous 可静默外推；finite clip | strict failure + 逐 role telemetry | 工程和实验方法增强 |
| Hadamard | log-domain 两层构造 | decode 后直接密集乘法 | Qwen 方案相对退化 |
| MatMul | log-domain Triton kernel | decode 后 `torch.matmul` | Qwen 方案相对退化 |
| Softmax | 无 causal mask 的 LSE 分解 | valid-set masked Softmax | 因果语义新增，但仍是 dense functional op |
| 模型状态 | 没有跨样本或跨时步状态 | time-coded K/V + positions + valid sideband | 相对原版最重要的新增 |
| Cache 更新 | 不存在 | append/reorder/range-identity check | 工程成立；研究 claim 需长上下文证据 |
| 评测 | Top-1/Top-5 | PPL、KL、top-1、cache、generation、overflow | 实验合同增强 |
| Continuous | 近似保持视觉精度 | PPL 近似不变 | 正确性 sanity check，不是核心创新 |
| Finite | 视觉上 P=1024/4096 可用 | Qwen `P=4096` 完全失败 | 当前是失效证据，不是性能贡献 |
| Noise | Gaussian time noise | 未实现 | 未继承 |
| Event backend | 无真实硬件，但有专用代数 kernel | 无，主要算子完全 dense | 不能声称系统收益 |

## 4. 哪些内容明确不是新创新

下列内容应从论文的“方法创新”中删除或降级为实现细节。

### 4.1 Affine TTFS 编码

Qwen 方案的

\[
t=\frac{x-a}{b-a}, \qquad \hat{x}=a+t(b-a)
\]

与原版 [`value_to_ttfs()`](../ttfsformer/spike/param.py#L9) 和 `ttfs_to_value()` 数学同构。这是直接继承，不是新编码理论。

### 4.2 Decode-compute-encode wrapper

Qwen `_linear()` 与原版 [`SpikingWrapped`](../ttfsformer/spike/conversion.py#L9) 都是：

\[
S_f(t)=E(f(D(t))).
\]

Continuous PPL 不变主要证明该包装没有破坏 Qwen 语义，不证明新的 SNN 计算能力。

### 4.3 RoPE、SwiGLU、Softmax 和 residual 的“支持”

原版 EVA 路径已包含 RoPE 和 SwiGLU；Softmax、Hadamard 和 residual add 也已有构造。Qwen 的价值是将它们放入因果生成数据流，而不是首次实现这些算子。

### 4.4 RMSNorm 分解

将 RMSNorm 拆成 square、mean、`rsqrt`、multiply 和 weight 是正确的算子适配，但它没有提出新神经元动力学，也没有解决 finite precision 稳定性，因此暂不构成方法创新。

### 4.5 “首个 spiking LLM/TTFS LLM”

这类声称已不成立：

- [SpikeLLM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/510e7d39fce008a3e31de54b8f5be9ac-Abstract-Conference.html) 已将 spiking mechanism 扩展到 7B--70B LLM，并研究了 KV cache 中的显著通道。
- [Distribution-Aware Multi-Granularity Phase Coding, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/81858886d6a6f22c2e555e9d8f5172d9-Abstract-Conference.html) 已对 LLaMA 实现 ANN-to-SNN conversion，并专门优化非均匀激活的离散编码。
- [Kirin, arXiv 2026](https://arxiv.org/abs/2602.08817) 已明确提出面向 LLM 的 TTFS/整数混合转换、silence threshold 和有限时窗实验。
- [BiSpikCLM, arXiv 2026](https://arxiv.org/abs/2605.13859) 已提出因果 spiking language model 和 softmax-free spiking attention。
- [Otters++, arXiv 2026](https://arxiv.org/abs/2606.13016) 已将 TTFS Transformer 与实测器件衰减、噪声和系统能耗结合。

因此，当前 Qwen 项目不应使用“首个”类宽泛标题。

## 5. 当前仍然成立的贡献

### 5.1 因果 LLM 的 TTFS 正确性合同

当前代码完整定义了以下不变式：

1. 只有 valid query-key pair 参与 attention logits 和 Softmax；
2. padding 和全无效行不会引入 NaN；
3. GQA 中 K/V 先按原 KV heads 编码和缓存，使用前再 `repeat_kv`；
4. RoPE 后的 K 与对应 `position_ids` 同步进入 cache；
5. 旧 cache 的 time code 只 append，不在每个 decode step 重复量化；
6. cache reorder 同时重排 time、position 和 valid sideband。

这是相对 TTFSFormer 最明确的实质性增量。但在更广文献中，causal spiking attention 和 spiking KV 已经存在，所以当前更适合称为 **TTFS-specific correctness contract**，而不是独立核心算法。

**当前证据：** mask 单元测试、cache/no-cache 对照、cache reorder 测试和单个 8-token generation smoke case。

**当前限制：** 缺少长上下文、批处理、padding 组合、beam search、cache 持久化与跨 RangeRegistry 兼容性实验。

### 5.2 可审计的时间码类型系统

`SpikeTensor.range_name` 将解码区间身份与 time tensor 绑定，`RangeRegistry` 保存模型、数据和配置元数据，`TTFSCodec` 则对每个 role 记录越界。

这一设计解决了原版“裸 tensor + 由调用者传 range tuple”容易误解码的软件合同问题。它是有价值的 **研究基础设施贡献**，但仅凭类型安全不足以支撑算法论文。

### 5.3 对 uniform finite TTFS 失效的可观测性

当前 `P=4096` 实验并未成功，但它提供了一个重要观察：

| 设置 | PPL | Logits max-abs | Top-1 | Clip |
| --- | ---: | ---: | ---: | ---: |
| ANN | `29.258942438` | 0 | 100% reference | N/A |
| Continuous | `29.258942670` | `5.82e-5` | 100% | 0 |
| `P=4096` | `7.23e106` | `470.97` | `0.1528%` | 1,910,180,057 |

这说明“在 ViT 上可用的统一时间网格可以直接搬到 LLM”是错误假设。但目前只能声称 **发现了失效现象**，还不能声称 **揭示了失效机制**，因为量化、范围裁剪、边界数量和算子敏感性尚未通过消融分离。

### 5.4 面向 LLM 的评测与证据合同

当前工流不只比较最终 accuracy，还比较 PPL、NLL、KL、logits、cache/no-cache、generation 和 role-level overflow。这比原版只报图像分类准确率更适合自回归误差研究。

这仍然更接近 **evaluation contribution**，而非新 TTFS 算法。

## 6. 当前最有希望的研究创新方向

### 6.1 方向 A：Causal-State-Aware TTFS Coding

这是最符合当前代码积累、也最能与 TTFSFormer 拉开差异的方向。

核心问题是：视觉 TTFSFormer 每个样本只有一条有限深度轨迹，而自回归 LLM 的 K/V 在不同时刻生成后长期共存。如果 decode 阶段使用动态 range，新旧 cache 将不再共享相同时间语义；如果始终使用全局 range，又会严重牺牲有限时间分辨率。

可研究的方法包括：

1. prefill/decode 分离的 RangeSpec；
2. per-layer/per-head/per-channel 时间窗；
3. 带 codec epoch 的分段 cache；
4. 无需解码全部旧 K/V 的跨区间 attention；
5. 根据 cache age、head sensitivity 或当前 query 分配时间精度。

要成为论文创新，必须证明它在相同 time/bit/cache budget 下优于 global uniform TTFS 和普通 KV quantization。

### 6.2 方向 B：Uniform TTFS 在自回归 LLM 中的失效机制

这是当前最快可以做成的研究贡献。可将每个边界的误差写成：

\[
e_{l+1}\approx J_l e_l + q_l + c_l + r_l,
\]

其中 \(q_l\) 是时间舍入，\(c_l\) 是 range clipping，\(r_l\) 是算子近似，\(J_l\) 是后续算子的局部 Jacobian。

必须通过以下消融将“现象”升级为“机制”：

1. round-only 与 clip-only 分离；
2. progressive layer quantization；
3. RMSNorm/Softmax/MatMul/SwiGLU operator bypass；
4. every-boundary 与 block-output-only 比较；
5. prefill 与 decode 误差轨迹分离；
6. 多模型、多长度和多领域验证。

如果能得出可迁移的结论，例如“RMSNorm 的二次量化与 residual 长尾是首个稳定失效点”，即使最终有限 TTFS 仍然不成功，也可以形成有价值的 negative-result/mechanistic paper。

### 6.3 方向 C：敏感度加权的时间精度分配

当前所有 role 共享统一 `P`，但不同边界的 range width、分布和下游敏感度不同。一个更合理的目标是：

\[
\min_{\{P_i,E_i,D_i\}}
\sum_i s_i\,\mathbb{E}\left[
\left(x_i-D_i(Q_{P_i}(E_i(x_i)))\right)^2
\right]
\quad
\text{s.t.}\quad
\sum_i C_i(P_i)\le B,
\]

其中 \(s_i\) 是 role 对最终 NLL 的敏感度，\(B\) 是总时间/存储/事件预算。

但单纯的“分布感知非均匀编码”已被 ICLR 2026 Phase Coding 工作覆盖。要有新意，需要把优化限定为 **causal phase + stateful cache + end-to-end NLL sensitivity**，而不是再做一个 activation histogram quantizer。

### 6.4 方向 D：Simplex-Preserving Temporal Softmax

当前对 Softmax 概率逐元素量化，会破坏：

\[
\sum_j p_j=1.
\]

可研究在有限时码下同时保持：

1. causal invalid key 严格为 0；
2. valid 概率和为 1；
3. 排序或比值关系尽可能保持；
4. 不引入高成本全局归一化。

这一点相对 TTFSFormer 的无 mask Softmax 是明确扩展。但需要与 softmax-free spiking attention 和普通概率量化方法比较。

### 6.5 方向 E：Event-Native Autoregressive Backend

如果目标是高能效 SNN 论文，最终必须解决：

1. QK/AV/Linear 如何消费脉冲事件而不全量恢复密集实值；
2. KV cache 如何以 time/event 形式存储并读取；
3. 每个 token 的发放/接收窗、同步屏障和无脉冲语义；
4. prefill 并行性与 decode 低延迟之间如何取舍。

这是最强、也最难的长期创新方向。在此之前，应坚持使用 `functional TTFS simulator` 而非 `event-driven SNN` 的名称。

## 7. 最推荐的论文故事

### 7.1 当前不成立的故事

> 我们首次将 Qwen2.5 无损转换为高能效 TTFS SNN。

不成立原因：

1. “首次”被 SpikeLLM、PhaseSNN 和 Kirin 等工作压缩；
2. Continuous 结果主要是 affine 可逆性；
3. finite `P=4096` 已崩溃；
4. 当前没有 event-native execution 或能耗证据。

### 7.2 近期可成立的故事

> **Uniform temporal coding does not transfer from vision Transformers to autoregressive LLMs. We build a causal-state TTFS diagnostic framework and identify how normalization, residual accumulation, masked probability coding, and persistent KV state jointly produce conversion failure.**

中文表述：

> 我们建立自回归 Transformer 的 causal-state TTFS 评测框架，系统分离时间量化、范围裁剪、算子敏感性与跨 token 状态累积，并揭示 uniform TTFS 从视觉 Transformer 迁移到 LLM 时的级联失效机制。

这个故事不要求先取得能效收益，但必须完成多模型、多长度和严格消融。

### 7.3 中期方法故事

> **Causal-State TTFS uses phase-specific ranges, sensitivity-aware temporal precision, and versioned time-coded KV memory to preserve autoregressive language quality under a fixed temporal budget.**

这个故事需要至少一个成功的有限精度结果，并在相同质量和预算下优于：

1. global uniform TTFS；
2. block-output-only TTFS；
3. uniform INT activation/KV quantization；
4. distribution-aware phase coding 的可比较实现。

## 8. 要把候选创新变成可投贡献，最少需要哪些证据

### 8.1 失效机制证据

- 输出 24 层的 NRMSE、cosine、SNR、overflow 和 NLL sensitivity。
- 确定首个系统性发散的 role 与 layer。
- 完成 round/clip、operator、boundary-density 和 cache 消融。
- 在 Qwen 与至少一个 LLaMA/OPT 架构上复现机制。

### 8.2 有限精度方法证据

- 扫描 `P={256,512,1024,2048,4096,8192,16384}`。
- 在 held-out test 而非开发 validation 上报告 PPL 和下游任务。
- 在相同 bit/time budget 下比较 INT8/INT4 与非均匀编码。
- 分别报告 prefill、decode 和 cache 使用下的质量。

### 8.3 Stateful cache 证据

- 上下文长度至少覆盖 128/512/2K/8K。
- 覆盖 batch padding、不同 prompt 长度、beam reorder 和长生成。
- 报告 K/V 与最终 logits 的误差随 cache age 的变化。
- 比较 time-coded cache 与标准 KV quantization 的 bytes/token 和质量。

### 8.4 系统证据

- 明确 time code 位宽、时窗、脉冲数、静默值和同步成本。
- 统计 event-native coverage，不得把 dense fallback 算作 AC。
- 报告 TTFT、tokens/s、latency/token、bytes/token 和能耗。
- 与 FP16/BF16、INT8/INT4 和一个 spiking LLM 基线在等质量下比较。

## 9. Claim-Evidence 映射

| 潜在 claim | 当前证据 | 状态 |
| --- | --- | --- |
| Qwen2.5 的 Continuous TTFS 功能基线已跑通 | PPL/KL/top-1/零越界 | supported within current protocol |
| 因果/padding attention 在 TTFS 表示下语义正确 | valid-set 实现与单元测试 | supported at functional level |
| TTFS cache 支持最小自回归生成 | cache/no-cache + 8-token smoke | supported only for MVP scope |
| typed range 合同比原版更可审计 | 代码数据类型与 telemetry | supported as engineering contribution |
| Qwen 方案首次实现 spiking/TTFS LLM | 2025--2026 文献已有相关方法 | contradicted |
| RMSNorm、RoPE 或 SwiGLU 构成独立方法创新 | 主要是 dense 算子重编码 | contradicted |
| `P=4096` 的 finite TTFS 保持语言质量 | PPL 崩溃 | contradicted |
| 已揭示 uniform TTFS 的级联失效机制 | 只有端到端失败和 telemetry | needs controlled ablations |
| causal-state-aware coding 优于 uniform coding | 方法尚未实现 | needs method and evidence |
| 当前实现具有能效或延迟优势 | 无 event backend/完整成本实验 | unsupported |

## 10. 最终研究定位

当前 Qwen TTFS 方案相对 TTFSFormer 的主要价值是：

1. 将 TTFS 表示从单次视觉 encoder 扩展到因果、带状态的 decoder-only LLM；
2. 建立 mask、position、GQA 和 KV cache 的完整 functional contract；
3. 将 range 身份、越界和评测从隐式行为变成可审计工件；
4. 提供 uniform finite TTFS 在 Qwen 上严重失败的端到端证据。

这些内容足以构成一个高质量的研究基线和技术报告，但尚不是完成的方法贡献。

最值得继续投入的主线不是重复证明 Continuous 可逆，而是：

> **先用可控消融揭示 uniform TTFS 在自回归 LLM 中的失效机制，再提出同时感知 prefill/decode phase、下游敏感度和持久 KV state 的有限时间编码。**

如果这条主线成立，它与 TTFSFormer 的区别就不再是“ViT 换成 Qwen”，而是从 **stateless operator representation** 进一步推进到 **stateful causal temporal computation**。

## 11. 文档自审

### Contribution

- `pass`：已区分工程贡献、相对 TTFSFormer 的新增和绝对方法创新。
- `needs new experiment`：失效机制与 causal-state-aware coding 都尚未被实验证明。

### Writing clarity

- `pass`：全文区分 Continuous、finite 和 event-native 三层合同。
- `pass`：没有用“首个”、“无损 SNN”或“高能效”概括当前结果。

### Experimental strength

- `pass`：Continuous 正确性与 finite 失败都有明确数值。
- `needs new experiment`：只有单模型、单主数据集、短上下文和极少生成证据。

### Evaluation completeness

- `needs revision`：需要标准 PPL protocol 和真正 held-out test。
- `needs new experiment`：需要量化、spiking LLM、长上下文与系统基线。

### Method soundness

- `pass`：Continuous causal-state 功能合同在当前作用域内成立。
- `fail`：统一 finite precision 方法当前不可用。
- `needs new experiment`：尚未证明事件执行可实现性和系统净收益。
