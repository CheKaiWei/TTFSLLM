# 0913 Idea：从 TTFSFormer 到 Causal-State Differential TTFS

> 日期：2026-09-13  
> 分析对象：[`CONTINUOUS_TTFS_METHOD.md`](CONTINUOUS_TTFS_METHOD.md)、[`TTFSFORMER_拆包解包与研究价值分析.md`](TTFSFORMER_拆包解包与研究价值分析.md)、当前 `TTFS_LM` Qwen2.5 MVP，以及原版 TTFSFormer 代码。  
> 目标：回应两个核心 reviewer 质疑：
>
> 1. 当前方案与 TTFSFormer 过于相似；
> 2. 当前方案与 ANN activation quantization 过于接近，无法体现 TTFS 的独特价值。

## 1. 最重要的结论

当前 Qwen 方案已经完成了一个有价值的因果 LLM TTFS functional baseline，但还没有完成新的 TTFS 算法。它目前主要解决的是：

```text
如何把 Qwen 的 dense hidden state
放进一个带 range、mask、position 和 KV cache 的时间码接口
```

它还没有解决：

```text
如何让脉冲到达时间本身驱动计算，
并在不恢复密集实值的情况下获得延迟、事件数或能耗收益。
```

因此，最值得发展的主线不是再增加一个 `SpikingRMSNorm` 或 `SpikingSwiGLU`，而是：

> **Causal-State Differential TTFS（CSD-TTFS）：用持久的、带位置和有效性状态的时间码表示 Transformer 状态增量，并让 residual、attention 和 KV cache 在时间域中直接更新。**

它的关键变化是把当前的：

$$
\widetilde h_{l+1}
=E_{l+1}\left(F_l\left(D_l(\widetilde h_l)\right)\right)
$$

改成：

$$
\Delta h_l=F_l(h_l),
\qquad
h_{l+1}=h_l+\Delta h_l,
\qquad
\widetilde{\Delta h_l}=E_{\Delta,l}(\Delta h_l).
$$

硬件或 event-native backend 保存的是当前状态 $h_l$ 与稀疏时间事件 $\widetilde{\Delta h_l}$，而不是每层都重新编码完整 hidden activation。

这条主线同时解决三个问题：

1. **差异性**：计算对象从 ANN activation quantization 变成 temporal residual events；
2. **TTFS 特性**：时间用于事件到达、调度和早停，而不只是离散数值桶；
3. **LLM 特性**：将 residual state、causal attention 和 persistent KV cache 放入同一套状态方程。

## 2. 当前方案与 TTFSFormer 的根本问题

### 2.1 TTFSFormer 的定理是“可表示性”，不是“高效实现性”

原版 Theorem 4.1 通过构造输入核 $\eta$，使：

$$
V_i(T_{\text{emit}})=
\sum_j w_{ij}f_{ij}(x_j).
$$

Theorem 4.3 再通过输出核 $\psi$ 表示：

$$
x=h(V(T_{\text{emit}})).
$$

这可以证明函数被某个广义 TTFS 神经元表示，但核心代价被隐藏在：

- $f'(x)$；
- $h^{-1}(x)$；
- $\psi(s)$；
- 指数、对数、逆函数或 LUT。

也就是说，定理证明的是：

$$
\text{任意满足条件的函数}
\rightarrow
\text{某个可构造的时间核}
$$

而不是：

$$
\text{该时间核}
\rightarrow
\text{低面积、低延迟、低能耗的物理电路}.
$$

这使 reviewer 可以提出一个实质性批评：TTFSFormer 把复杂度从 ANN 算子搬到了“广义神经元核”中，而没有给出受约束的核族、近似误差和硬件成本定理。

### 2.2 原版实现的核心路径仍是 decode-compute-encode

原版 `SpikingWrapped` 执行：

$$
S_f(t)=E(f(D(t))).
$$

代码见 [`ttfsformer/spike/conversion.py`](../ttfsformer/spike/conversion.py#L9)。

当前 Qwen `_linear()`、RMSNorm、Softmax、SwiGLU 和 residual 仍沿用同一语义，只是边界数更多。主要运算依然是：

$$
\widetilde x
\rightarrow D(\widetilde x)
\rightarrow \text{dense PyTorch operator}
\rightarrow E(\cdot).
$$

因此当前 `SpikeTensor` 更接近一个带类型的数值包装器，而不是事件流。

### 2.3 Continuous 无损高度依赖可逆编码，研究辨识度低

当：

$$
D(E(x))=x
$$

且不进行量化、裁剪和噪声时，Continuous TTFS 近似保持 ANN 输出是预期结果。它可以证明实现合同没有破坏 ANN 语义，但不能证明：

- TTFS 比普通量化更好；
- TTFS 的时序真的参与计算；
- TTFS 具有事件稀疏性；
- TTFS 在硬件上具有能效优势。

因此 Continuous 应该是 correctness baseline，而不是论文的主要 novelty。

### 2.4 原版能耗模型没有覆盖广义核的真实成本

原版用：

$$
\frac{E_{\mathrm{SNN}}}{E_{\mathrm{ANN}}}
=
\frac{OP_{\mathrm{SNN}}E_{\mathrm{AC}}}
{OP_{\mathrm{ANN}}E_{\mathrm{MAC}}}
$$

并把 SNN 操作按统一 AC 代价估计。该模型没有明确计入：

- log/exp/rsqrt/LUT；
- 时间比较、排序和同步；
- membrane state；
- event routing 和 memory movement；
- KV cache 读写；
- 非线性核的面积、带宽和控制逻辑。

因此，“20%--30% ANN 能耗”只能是理想化算术估算，不能作为当前 Qwen 方案的继承性证据。

### 2.5 原版缺少自回归状态问题

TTFSFormer 的视觉推理主要是：

$$
x\rightarrow F(x)\rightarrow y
$$

Qwen 的 decode 是：

$$
\left(h_s,\mathcal C_{s-1}\right)
\rightarrow
\left(h_{s+1},\mathcal C_s\right),
$$

其中：

$$
\mathcal C_s=
\{(K_1,V_1,p_1,m_1),\ldots,(K_s,V_s,p_s,m_s)\}.
$$

原版没有回答以下问题：

1. 不同 token 的时间码如何共存于同一个 cache；
2. range 改变时是否必须重编码整个历史 cache；
3. 位置、有效性与时间码如何保持一致；
4. cache 读取能否由事件到达直接驱动；
5. decode latency 能否随事件提前结束。

这正是 Qwen 方案最有机会产生新研究问题的地方。

### 2.6 原版 mask、校准与时间语义并不完整

原版 EVA 的 attention mask 路径将无效 score 填成 time code '1'，而不是在 score 域填入 $-\infty$ 或在 Softmax 分母中排除。若该值直接解码为最大 score，它会把无效 key 推向最有利的位置，不能等价于标准 causal/padding mask。

原版默认的 MinMaxMonitor 还保存 batch 极值的平均值：

$$
a=\frac1B\sum_{b=1}^{B}\min(X_b),
\qquad
b=\frac1B\sum_{b=1}^{B}\max(X_b),
$$

而不是覆盖校准集的全局 active extrema。这会把范围选择、裁剪和时间分辨率混在一起，且没有逐 role overflow telemetry。

此外，论文的物理 TTFS 关系通常写成：

$$
t_{\mathrm{phys}}=T_{\mathrm{ref}}-\tau x,
$$

即值越大，发放越早；Qwen 当前 codec 使用：

$$
t_{\mathrm{code}}=\frac{x-a}{b-a},
$$

即值越大，code 越大。二者可以通过反向仿射映射关联，但必须明确 time code 与物理 spike timestamp 的方向。否则“取最大 code”与“取最早 spike”不是同一个操作。

这些问题说明，原版还缺少一份同时约束 **mask、range、时间方向和事件生命周期** 的协议。

### 2.7 Qwen MVP 暴露了原版未观测的 failure regime

当前 Qwen 结果显示：

| 模式 | 结果 |
| --- | --- |
| Continuous | PPL 约 '29.258942670'，与 ANN 近似一致 |
| P=4096 | PPL 约 '7.23e106'，Top-1 约 '0.1528%' |

这不是简单的“LLM 更大”问题。Qwen 同时引入了：

$$
\text{many boundaries}
\;+\;
\text{residual accumulation}
\;+\;
\text{persistent KV state}
\;+\;
\text{causal masked probability}.
$$

如果不做逐层消融，不能判断失败来自时间量化、范围裁剪、RMSNorm、Softmax、KV cache 还是它们的组合。这个未被 TTFSFormer 覆盖的 failure regime 本身可以成为研究对象。

## 3. 当前 Qwen MVP 的真实增量

### 3.1 因果 valid-set attention

当前 Qwen 使用：

$$
M_{b,q,k}
=v_{b,k}\mathbf 1[p_{b,k}\le p_{b,q}],
$$

并在有效 key 上计算：

$$
P_{b,h,q,k}
=
\frac{\exp(L_{b,h,q,k})}
{\sum_{j:M_{b,q,j}=1}\exp(L_{b,h,q,j})}.
$$

这相对视觉 TTFSFormer 是必要的语义扩展，但 mask 公式本身不是新的 attention 算法。它只有在以下情况下才可能成为研究贡献：

- mask 直接控制事件调度，而不只是 `masked_fill`；
- 因果约束可以减少未来事件的生成和路由；
- valid-set Softmax 能在时间域保持归一性。

### 3.2 GQA-aware time-coded cache

Qwen 的：

$$
H_q=14,\qquad H_{kv}=2,\qquad g=7.
$$

当前 cache 在 repeat 前保存：

$$
\widetilde K\in
\mathbb R^{B\times H_{kv}\times S\times d},
\qquad
\widetilde V\in
\mathbb R^{B\times H_{kv}\times S\times d}.
$$

使用前才执行：

$$
K'_h=K_{\lfloor h/g\rfloor},
\qquad
V'_h=V_{\lfloor h/g\rfloor}.
$$

这减少了重复存储，但仍是功能层优化。真正的研究问题是：能否让同一个 KV event 被多个 query heads 共享处理，而不是 repeat 后再恢复成 dense K/V。

### 3.3 Range identity 与 telemetry

当前 `SpikeTensor` 保存：

$$
(\text{time},\text{range\_name},\text{value\_dtype}).
$$

这比原版“裸 tensor + 外部 tuple”更安全，可以表达：

$$
\operatorname{decode}(t,r_i)\ne
\operatorname{decode}(t,r_j).
$$

这是明确的工程贡献，但还不是神经计算创新。它可以成为新方法的基础：将 `range_name` 升级为包含 phase、head、cache segment 和 codec epoch 的时间协议。

## 4. CSD-TTFS：最值得投入的主 idea

### 4.1 设计目标

CSD-TTFS 的目标不是把完整 activation 变成一个更细的量化桶，而是：

1. 用时间事件表示 residual update；
2. 用膜电位或本地状态保存 common-mode baseline；
3. 用 position/valid/epoch sideband 管理 causal KV state；
4. 让时间决定事件到达顺序、早停和计算调度；
5. 将每个核限制在可实现的 temporal primitive 中。

### 4.2 差分时间码

原始 Qwen block 为：

$$
n_l=\operatorname{RMSNorm}(h_l),
$$

$$
a_l=\operatorname{Attention}(n_l),
\qquad
h'_l=h_l+a_l,
$$

$$
m_l=\operatorname{SwiGLU}(\operatorname{RMSNorm}(h'_l)),
\qquad
h_{l+1}=h'_l+m_l.
$$

CSD-TTFS 不把两个完整 hidden state 都重新编码，而是编码更新量：

$$
\Delta^a_l=a_l,
\qquad
\Delta^m_l=m_l.
$$

状态更新为：

$$
h'_l=h_l+D_{\Delta^a,l}(\widetilde{\Delta^a_l}),
$$

$$
h_{l+1}=h'_l+D_{\Delta^m,l}(\widetilde{\Delta^m_l}).
$$

若硬件的 neuron state 保存 $h_l$，则 residual add 不需要：

$$
D(\widetilde h_l)+D(\widetilde a_l)
\rightarrow
E(h_l+a_l).
$$

它只需要将 attention event 作为对状态的增量写入：

$$
u_l\leftarrow u_l+\operatorname{EventAcc}(\widetilde{\Delta^a_l}).
$$

### 4.3 为什么差分编码不是普通量化

普通 activation quantization 量化的是：

$$
Q(h_l).
$$

CSD-TTFS 量化的是：

$$
Q_\mathrm{time}(\Delta h_l),
$$

并让事件通过状态更新改变下一层的计算时刻。两者的计算路径不同：

| 机制 | 普通量化 | CSD-TTFS |
| --- | --- | --- |
| 编码对象 | 完整 activation | 相对当前状态的 update |
| 主要收益 | 存储位宽 | 事件数、调度和 residual state 更新 |
| 误差基准 | activation 重构误差 | state trajectory 误差 |
| 计算方式 | 量化后仍 dense MAC | 事件到达驱动 state accumulation |
| 零值含义 | 一个量化桶 | 无 update、无需路由或计算 |
| latency | 固定 kernel 时间 | 可由事件完成时间决定 |

### 4.4 Signed differential TTFS

Qwen activation 和 residual update 都包含正负值。建议使用零参考时间 $t_0$，定义：

$$
\Delta h_i
=
\begin{cases}
\Delta h_i^+-c^+,&\Delta h_i\ge 0,\\
-(\Delta h_i^- -c^-),&\Delta h_i<0.
\end{cases}
$$

分别用正、负 event channel 编码：

$$
t_i^+=E^+(\Delta h_i^+),
\qquad
t_i^-=E^-(\Delta h_i^-).
$$

当 $|\Delta h_i|\le \epsilon_l$ 时，不发放 event：

$$
\widetilde{\Delta h_i}=\varnothing.
$$

这使静默状态具有明确物理意义：该神经元对 residual state 没有可观测更新，而不是把数值恰好塞到某个量化桶里。

## 5. Stateful temporal KV memory

### 5.1 Cache 的时间协议

每个 cache segment 使用固定 codec：

$$
R_{l,h,\phi,e}=[a_{l,h,\phi,e},b_{l,h,\phi,e}],
$$

其中：

- $l$：layer；
- $h$：KV head；
- $\phi\in\{\text{prefill},\text{decode}\}$：阶段；
- $e$：codec epoch 或 cache segment。

K/V 编码为：

$$
\widetilde K_{l,s}
=E_{R_{l,h,\phi,e}}(R(p_s)W_kh_s),
$$

$$
\widetilde V_{l,s}
=E_{R_{l,h,\phi,e}}(W_vh_s).

$$

cache 保存：

$$
\mathcal C_{l,s}
=
\left(
\widetilde K_{l,1:s},
\widetilde V_{l,1:s},
p_{1:s},m_{1:s},e
\right).
$$

新增 token 只能 append 到同一个 epoch：

$$
U(\mathcal C_{l,s-1},s)
=\operatorname{append}(\mathcal C_{l,s-1},
\widetilde K_{l,s},\widetilde V_{l,s},p_s,m_s).
$$

如果 range 必须变化，不要静默覆盖旧 cache，而是显式执行：

$$
\mathcal C^{e}_{l,s}
\rightarrow
\operatorname{Bridge}_{e\rightarrow e+1}(\mathcal C^{e}_{l,s})
$$

或把 cache 切成多个 segment，分别携带 codec metadata。

### 5.2 Cache 中最值得研究的 invariants

需要证明或测试：

1. **append invariant**：旧 token 的 time code 不因新 token 加入而改变；
2. **position invariant**：同一个 $p_s$ 在 full/prefill/decode 中生成相同 RoPE 语义；
3. **mask invariant**：无效 token 不参与 Softmax 分母；
4. **GQA invariant**：repeat KV 前后 attention 结果一致；
5. **epoch invariant**：不同 codec epoch 的 time code 不可直接拼接；
6. **reorder invariant**：beam reorder 同时重排 time、position 和 validity；
7. **event invariant**：静默 event 不需要被路由，也不会改变状态。

这些 invariants 是 TTFS-specific state contract；原版 TTFSFormer 没有相应对象。

## 6. 让 TTFS 真正区别于量化的第二个 idea：时间驱动 attention

### 6.1 当前 attention 仍是量化后的 dense attention

当前路径为：

$$
\widetilde Q,\widetilde K
\rightarrow D(\widetilde Q),D(\widetilde K)
\rightarrow QK^\top
\rightarrow E(L).
$$

如果只改变 $P$、range 或 histogram，它在 reviewer 看来仍然是 activation quantization。

### 6.2 Temporal-Arrival Attention

建议将 attention 重新定义为事件到达过程：

$$
S_{qk}=\operatorname{TemporalScore}
\left(t_q,t_k,p_q,p_k\right).
$$

只有满足：

$$
p_k\le p_q,qquad m_k=1
$$

的 key event 才进入 accumulator。

可采用一个受约束的时间核：

$$
\kappa_\theta(t_q,t_k)
=
\sum_{r=1}^{R}
\alpha_r
\exp\left(-\frac{|t_q-t_k|}{\tau_r}\right).
$$

attention score 由核响应累积：

$$
S_{qk}
=
\sum_{d=1}^{d_h}
\operatorname{sign}(q_{qkd})
\kappa_\theta(t^q_{qd},t^k_{kd}).
$$

这不是简单把数值量化后再 matmul，而是让 spike timestamp 直接成为相似度计算的输入。

### 6.3 Margin-certified early stop

在 decode 阶段，事件可以按到达时间排序。处理到时间 (	au) 时，维护当前候选输出的上下界：

$$
L_v(\tau)
\le z_v
\le U_v(\tau).
$$

如果当前最佳 token $v^*$ 满足：

$$
L_{v^*}(\tau)
>
\max_{v\ne v^*}U_v(\tau),
$$

则可以安全停止后续事件处理：

$$
\operatorname{argmax}_v z_v=v^*.
$$

这给出 TTFS 特有的动态 latency：不是固定处理所有 quantized activation，而是当时间证据足以确定决策时提前结束。

这条方向的论文 claim 应该是：

> 在保持 token decision 一致率的条件下，基于时间 margin 的 attention event processing 减少平均事件数和 decode latency。

不能只报告 PPL 不变；必须报告 early-stop ratio、平均事件数和 p50/p95 latency。

## 7. 让广义神经元不再隐藏复杂度：受约束 temporal kernel

### 7.1 TTFSFormer 的问题

如果允许任意 $\eta=f'$ 和任意 $\psi$，几乎任何函数都可以被写成一个“新神经元”。这会产生一个可证伪的 reviewer 质疑：创新只是把函数写进 kernel。

### 7.2 受约束 kernel family

将所有神经元核限制为有限个可实现基函数：

$$
\eta_\theta(s)
=
\sum_{r=1}^{R}
a_r\exp(-s/\tau_r)\mathbf 1[s\ge0],
$$

$$
\psi_\phi(s)
=
\sum_{r=1}^{R}
b_r\exp(-s/\rho_r)\mathbf 1[s\ge0].
$$

对于目标函数 $f$，不再声称精确表示，而是给出：

$$
\sup_{x\in\mathcal X}
|f(x)-\hat f_{R}(x)|
\le \varepsilon_R,
$$

并报告：

$$
\text{kernel cost}
=O(R),
\qquad
\text{state cost}=O(R),
\qquad
\text{latency}=O(T_R).
$$

这样可以把“任意函数可表示”改成“在明确硬件预算下，函数可近似到什么误差”。

### 7.3 适用于 Qwen 的算子

优先处理：

1. SiLU：近似为少量指数基函数；
2. RMSNorm 的 `rsqrt`：分段单调基函数或 reciprocal approximation；
3. Softmax：用 log-domain accumulator + bounded exponential kernel；
4. Hadamard：受限的 temporal product primitive；
5. QK/AV：共享 temporal product primitive 和 event accumulation。

这条方向比再次声称“Softmax 可以被 TTFS 表示”更强，因为它同时给出：

- 表示误差；
- 核数量；
- 时间延迟；
- 状态面积；
- 能耗上界或测量方法。

## 8. 时间精度分配：不能只是 distribution-aware quantization

已有工作已经研究分布感知、多粒度 phase coding，因此简单地把 global `P` 改成 per-layer histogram 不足以形成新意。

真正有差异的目标应是联合优化 **状态敏感度、事件数量和 decode latency**：

$$
\min_{\{P_i,\tau_i,\epsilon_i\}}
\mathbb E[\Delta\mathrm{NLL}]
+\lambda\mathbb E[\mathrm{events}]
+\mu\mathbb E[\mathrm{latency}]
+\nu\mathbb E[\mathrm{cache\ bytes}],
$$

约束：

$$
\sum_i C_i(P_i,\tau_i,\epsilon_i)
\le B,
$$

其中 role $i$ 还要区分：

$$
i=\left(layer,head,phase,cache\ age,operator\right).
$$

敏感度可以定义为：

$$
s_i
=
\left\|
\frac{\partial \mathcal L_{mathrm{NLL}}}
{\partial x_i}
\right\|,
$$

或用有限差分估计 token decision margin 对该 role 的影响。

与普通 quantization 的区别在于：$P_i$ 不仅控制重构误差，还控制 event release、early stop 和 cache retention。

## 9. 哪一个 idea 最值得现在做

### 推荐主线：CSD-TTFS

将以下四个组件合成一个最小可发表方法：

1. **Differential residual coding**：编码 attention/MLP update，而非完整 hidden；
2. **Stateful codec epoch**：prefill/decode/cache segment 使用明确时间协议；
3. **Temporal-arrival attention**：因果 mask 直接约束 event scheduler；
4. **Margin-certified early stop**：根据时间证据提前停止 attention 或 token decision。

推荐的核心算法接口：

$$
(h_s,\mathcal C_s)
\xrightarrow{\mathrm{CSD\text{-}TTFS}}
(\widetilde{\Delta h_s},\mathcal C_{s+1}).
$$

推荐的核心理论目标：

$$
\Pr[\hat y_s\ne y_s]
\le
\Pr[\mathrm{margin}_s\le \varepsilon_s]
$$

以及：

$$
\mathbb E[\mathrm{events}_{\mathrm{CSD}}]
<
\mathbb E[\mathrm{events}_{\mathrm{uniform}}]
$$

在相同或更低的 NLL 和 token agreement 下成立。

### 次优主线：TTFS 失效机制论文

如果 event-native backend 暂时做不出来，仍可把当前项目做成一篇有价值的机制论文：

> **Why Uniform TTFS Conversion Fails in Autoregressive LLMs: Error Propagation Across Normalization, Residual State, Attention and KV Cache**

核心不是报告 `P=4096` 失败，而是证明：

$$
e_{l,s}
\approx
J_{l,s}e_{l,s-1}
+q_{l,s}
+c_{l,s}
+r_{l,s},
$$

并通过 operator bypass、progressive quantization 和 cache-age sweep 确定哪一项主导失效。

这条路线的价值来自新知识，而不是硬件 claim：

1. 视觉 TTFS 的低脉冲优势为何不能直接迁移到 LLM；
2. 哪些算子组合造成不可逆误差放大；
3. 为什么 cache state 让 range/precision 问题不同于单次 vision forward；
4. 哪些设计原则可用于未来的 TTFS LLM。

## 10. 必须明确放弃的 claim

当前或近期不应再写：

1. “首个 spiking LLM”；
2. “首个 TTFS LLM”；
3. “Qwen 的 RMSNorm/RoPE/SwiGLU 本身是新方法”；
4. “Continuous lossless 等于真实 SNN 无损”；
5. “统一 `P` 的时间码就是 TTFS 优势”；
6. “dense decode-compute-encode 实现具有能效收益”；
7. “只用 MAC/AC 单价即可证明端到端能耗下降”。

相关工作已经覆盖大规模 spiking LLM、KV cache spiking、分布感知 phase coding、TTFS/整数混合以及硬件协同 TTFS，例如：

- [SpikeLLM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/510e7d39fce008a3e31de54b8f5be9ac-Abstract-Conference.html)
- [Distribution-Aware Multi-Granularity Phase Coding, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/81858886d6a6f22c2e555e9d8f5172d9-Abstract-Conference.html)
- [Kirin, arXiv:2602.08817](https://arxiv.org/abs/2602.08817)
- [Otters++, arXiv:2606.13016](https://arxiv.org/abs/2606.13016)

## 11. 最小实验矩阵

### 11.1 先证明不是普通量化

比较以下四条路径：

| 路径 | 编码对象 | 是否 decode dense | 是否使用时间调度 |
| --- | --- | ---: | ---: |
| FP16 ANN | 无 | 是 | 否 |
| INT8/INT4 activation quantization | 完整 activation | 是 | 否 |
| Uniform TTFS | 完整 activation time code | 是 | 否 |
| CSD-TTFS | residual event + state | 否或局部 | 是 |

必须在相同模型、数据、质量门槛和存储预算下比较：

- PPL/NLL；
- token agreement；
- event count；
- active/silent ratio；
- KV bytes/token；
- prefill latency；
- decode p50/p95 latency；
- dense fallback percentage。

### 11.2 逐项消融

1. full activation code vs differential code；
2. global range vs phase-aware range；
3. no epoch vs epoch-checked cache；
4. numeric causal mask vs event scheduler mask；
5. fixed latency vs margin early stop；
6. arbitrary log/exp kernel vs constrained temporal basis；
7. every-boundary emit vs residual-only emit；
8. dense matmul vs event temporal score。

### 11.3 规模与状态

至少覆盖：

- Qwen2.5-0.5B、1.5B 或 3B；
- 一个 LLaMA/OPT 对照；
- context length 128/512/2K/8K；
- prefill 与 decode 分开；
- batch padding、不同 prompt 长度、cache reorder；
- WikiText-2、C4、LAMBADA 或标准 zero-shot task。

## 12. Claim-Evidence 计划

| 未来 claim | 必需证据 | 当前状态 |
| --- | --- | --- |
| CSD-TTFS 的 residual event 比完整 activation 更稀疏 | event count、silent ratio、相同质量对照 | 未实现 |
| cache epoch 防止跨阶段时间码语义冲突 | epoch mismatch 测试、长 decode | 仅有 range identity 基础 |
| event scheduler 正确实现 causal mask | 与 dense attention 逐 token 对照 | 当前是 dense mask |
| temporal score 而非 quantized dot product | 不解码的 event kernel | 未实现 |
| margin early stop 保持 token decision | certified bound、agreement、latency | 未实现 |
| constrained kernel 具有可控成本 | approximation error + kernel count + cycle/energy | 未实现 |
| CSD-TTFS 优于 INT8/INT4 | equal-quality/equal-budget Pareto 曲线 | 未实现 |

## 13. 实施路线

### P0：失效机制与基线卫生

1. 将当前 `P=4096` 失败拆为 round、clip、operator 和 cache 四类误差；
2. 输出每层、每个 role、每个 decode step 的误差和 overflow；
3. 增加 INT8/INT4 与 uniform TTFS 公平基线；
4. 固定标准 PPL、held-out test 和多上下文评测；
5. 将 Continuous 重新定义为 correctness prerequisite。

### P1：差分表示原型

1. 在 Qwen 中先只对 attention residual 使用 differential code；
2. 保持 MLP 和 LayerNorm 为 dense functional reference；
3. 验证 `h_l + delta_l` 与 full activation encoding 的误差、事件数和范围宽度；
4. 加入正负 event channel 和 silence threshold；
5. 比较 residual-only、block-output-only 和 every-boundary 三种边界策略。

### P2：Stateful cache 协议

1. 将 `range_name` 扩展为 `codec_epoch + phase + head + segment`；
2. 实现 prefill/decode 分离的 RangeSpec；
3. 禁止跨 epoch 直接拼接 cache；
4. 对长上下文记录 cache age、range width、event count 和 logits drift；
5. 证明旧 K/V 不因新 token 加入而重编码。

### P3：Event-native attention

1. 先实现不解码的 temporal score reference kernel；
2. 将 causal mask 移入事件 scheduler；
3. 实现 margin-certified early stop；
4. 用 dense reference 验证 score、概率和 token decision；
5. 再考虑 Triton/FPGA/ASIC 的物理映射。

### P4：受约束神经元与系统证据

1. 用有限指数基函数替代任意 log/exp kernel；
2. 给出 kernel approximation error；
3. 统计真实 memory movement、同步、事件路由和 cache 访问；
4. 在 cycle-level simulator 或硬件原型上测量；
5. 只有在这些证据完成后，才恢复 energy/latency claim。

## 14. 最终建议

当前最值得押注的不是“Qwen 版 TTFSFormer”，而是：

> **CSD-TTFS：面向自回归 Transformer 的差分、带状态、事件调度驱动的 TTFS。**

其相对 TTFSFormer 的真正区别应当是：

$$
\text{单次 stateless activation recoding}
\rightarrow
\text{stateful causal residual event computation}
$$

其相对 ANN quantization 的真正区别应当是：

$$
\text{固定数值桶}
\rightarrow
\text{由时间到达驱动的状态更新、动态延迟和事件稀疏}
$$

如果暂时无法实现 event-native backend，应诚实地把论文定位为：

> **Uniform TTFS conversion 的 LLM 失效机制与 Causal-State 表示基线。**

这仍然可以是一项有价值的工作，但不能继续把 decode-compute-encode 的 Continuous 结果包装成已完成的高能效 SNN。

## 15. 五维自审

### Contribution

- `pass`：提出了区别于简单量化的明确主线：差分事件、持久状态和时间调度。
- `needs new experiment`：CSD-TTFS 尚未实现，当前只能称为研究计划。

### Writing clarity

- `pass`：将 TTFSFormer 的表示定理、软件模拟和硬件实现分开。
- `pass`：明确列出不能继续使用的宽泛 claim。

### Experimental strength

- `pass`：已有 Continuous 基线和 `P=4096` 失败结果。
- `needs new experiment`：尚无 event count、latency、cache bytes 或不解码 kernel 证据。

### Evaluation completeness

- `needs revision`：需要 INT8/INT4、spiking LLM、长上下文和 held-out test。
- `needs new experiment`：需要 Pareto 曲线、消融和真实状态生命周期实验。

### Method soundness

- `pass`：问题定义和差分状态方程自洽。
- `needs new experiment`：时间核近似界、early-stop 安全界和 event-native 执行仍待证明。
