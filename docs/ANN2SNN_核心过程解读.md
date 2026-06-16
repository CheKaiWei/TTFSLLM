# ANN2SNN 核心过程解读：以 ViT Block 的 Attention 子层为例

这份文档的目标不是把整篇论文逐段翻译，而是把这份仓库里最核心的 `ANN2SNN` 过程真正讲清楚：

1. 论文里的转换流程，在代码里分别落到了哪里。
2. 选一个最关键的例子，把 ANN 是怎样一步一步变成 SNN 的讲透。
3. 说明转换后“数据结构”到底长什么样，避免把它误解成事件列表。

---

## 1. 先给结论：这个仓库里的 ANN2SNN，本质上在做什么

这份代码实现的是论文 **TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer** 的转换流程。它不是重新训练一个 SNN，而是：

1. 先跑一遍原始 ANN。
2. 在每个关键位置记录激活值范围 `[(a, b)]`。
3. 用这个范围把实数激活值线性映射成 TTFS 编码。
4. 用 `Spiking*` 模块替换原来的 ANN 模块。
5. 在 SNN 前向中，张量始终以“TTFS 编码张量”的形式流动；只有在每个算子内部，才会临时解码回实数、做运算、再编码回 TTFS。

对应代码：

- 监控 ANN 并导出区间：`test/eval_origin.py`
- TTFS 编码和参数字典：`ttfsformer/spike/param.py`
- ANN 模块转 SNN 模块：`ttfsformer/spike/conversion.py`
- SNN 版 ViT：`ttfsformer/model/spiking_vit.py`
- LayerNorm / Softmax / 加法 / 乘法等 SNN 运算：`ttfsformer/spike/module.py`、`ttfsformer/spike/ops.py`

---

## 2. 论文流程如何映射到代码

论文附录里的 Algorithm 1 可以概括成三步。

### 第一步：监控原始 ANN 的每层输出范围

论文说要先监控每层输出范围 `[a_n, b_n]`，再根据这个范围设置 TTFS 的表示尺度。

代码对应：

- `test/eval_origin.py` 先加载原始模型，然后调用 `set_monitor(model, lambda: MinMaxMonitor())`。
- `set_monitor` 会把 `LayerNorm` 和 `Softmax` 替换成可监控版本，并把所有 `BaseMonitor` 替换成真正的监控器。
- 跑完数据后，用 `get_monitor_data(model)` 导出一个 JSON。

关键点：

- 普通层用 `monitor_0 / monitor_1 / ... / monitor_o` 标记输入、输出和中间点。
- `LayerNorm`、`Softmax` 因为内部本身有中间步骤，所以还会额外记录内部范围。

### 第二步：把数值范围变成 TTFS 编码区间

论文里使用的是绝对时间变量，比如 `T_ref`、`T_emit`、`T_end`、`tau`。代码没有显式保存这些绝对时刻，而是把它们折叠成了一个更直接的线性编码：

```python
value_to_ttfs(x, (a, b)) = (x - a) / (b - a)
ttfs_to_value(t, (a, b)) = t * (b - a) + a
```

这意味着：

- 当张量在代码里叫 `t` 时，它不是“真实物理时钟上的绝对发放时刻”。
- 它更准确地说是“归一化后的 TTFS 码”或“归一化后的发放先后量”。
- 论文里的时间尺度，已经被吸收到区间 `(a, b)` 和 `emit()` 的离散化过程里了。

### 第三步：把 ANN 层替换成 SNN 层

`ann_to_snn()` 做的就是这个工作：

- `Linear / Conv / GELU / SiLU / ReLU` -> `SpikingWrapped`
- `LayerNorm` -> `SpikingLayerNorm`
- `Softmax` -> `SpikingSoftmax`

`SpikingWrapped` 的思想非常直接：

1. 先把 TTFS 张量解码回实数。
2. 调用原始 ANN 层。
3. 再把输出编码回 TTFS。
4. 最后调用 `config.emit()` 模拟精度量化或噪声。

所以这个仓库的“lossless”核心并不是说完全不做实数计算，而是说：

- 权重直接继承原 ANN。
- 每个 SNN 层都严格围绕同一组区间做“解码 -> 运算 -> 编码”。
- 当 `precision=None` 且 `noise=None` 时，理论上能非常贴近原 ANN 的数值流。

---

## 3. 为什么我选 Attention 子层当“最核心例子”

如果只讲 `Linear -> GELU -> Linear`，虽然容易，但不够体现这篇论文的核心价值。

这篇论文真正难的地方是把 Transformer 中最麻烦的几类操作都接住了：

1. `LayerNorm`
2. `QK^T` / `AV` 这样的乘法与矩阵乘
3. `Softmax`
4. `Residual Add`

而一个 `ViT Block` 里的前半段：

`LayerNorm -> Attention(qkv -> qk^T -> softmax -> av -> proj) -> residual add`

正好把这四类操作全部串起来，所以这是最值得讲透的例子。

---

## 4. 核心例子：一个 ViT Block 的 Attention 是怎样从 ANN 变成 SNN 的

下面用 `blocks.0` 的前半段举例。

### 4.1 原始 ANN 路径

在原始 `Block.forward()` 里，前半段逻辑是：

```python
x = monitor_0(x)
tmp = norm1(x)
tmp = monitor_1(tmp)
tmp = attn(tmp)
tmp = monitor_2(tmp)
x = x + tmp
x = monitor_3(x)
```

其中 `attn(tmp)` 在原始 ANN 中又会展开成：

```python
qkv = qkv(tmp)
q, k, v = reshape(qkv)
attn = (q @ k.transpose(-2, -1)) * scale
attn = softmax(attn)
x = attn @ v
x = proj(x)
```

### 4.2 转换前先得到了什么数据

运行 `test/eval_origin.py` 后，会生成一个监控 JSON。对于 `blocks.0`，它的结构大致会像这样：

```json
{
  "blocks.0.monitor_0": [a0, b0],
  "blocks.0.monitor_1": [a1, b1],
  "blocks.0.monitor_2": [a2, b2],
  "blocks.0.monitor_3": [a3, b3],

  "blocks.0.norm1.monitor_1": [n1_a, n1_b],
  "blocks.0.norm1.monitor_2": [n2_a, n2_b],

  "blocks.0.attn.monitor_0": [q_in_a, q_in_b],
  "blocks.0.attn.monitor_1": [qkv_a, qkv_b],
  "blocks.0.attn.monitor_2": [score_a, score_b],
  "blocks.0.attn.monitor_3": [prob_a, prob_b],
  "blocks.0.attn.monitor_4": [ctx_a, ctx_b],
  "blocks.0.attn.monitor_o": [proj_a, proj_b],

  "blocks.0.attn.softmax.monitor_1": [lse_a, lse_b]
}
```

这里最重要的不是具体数值，而是它表达了：

- 每一个中间结果，都有自己独立的数值范围。
- SNN 不是拿一套全局统一缩放去硬套所有层，而是按监控点精确转换。

### 4.3 转换后模型的结构是什么样

原来的：

```text
Block
  norm1: LayerNorm
  attn: Attention
```

会变成：

```text
SpikingBlock
  norm1: SpikingLayerNorm
  attn: SpikingAttention
    qkv: SpikingWrapped(Linear)
    softmax: SpikingSoftmax
    proj: SpikingWrapped(Linear)
```

也就是说，模型结构还是 PyTorch 模块树，但内部算子已经换成了 SNN 版本。

### 4.4 运行时张量到底长什么样

以 ViT-S/16 为例，输入一张 `224 x 224` 图像时：

- patch 数是 `14 x 14 = 196`
- 再加 1 个 `cls token`
- token 总数 `N = 197`

如果 embedding 维度是 `C = 384`，head 数是 `6`，那么：

- block 输入张量形状：`[B, 197, 384]`
- 每个 head 维度：`64`

最关键的一点：

- 在 SNN 前向过程中，这些张量**仍然是普通 `torch.Tensor`**
- 只是它们的数值语义不再是“激活值”，而是“归一化 TTFS 码”
- 所以它们通常落在 `[0, 1]`

换句话说，这份代码不是把脉冲保存成：

```text
[(neuron_id, spike_time), ...]
```

这样的事件列表，而是保存成和 ANN 同形状的稠密张量：

```text
Tensor[B, N, C]
```

只是张量里的每个浮点数代表“该神经元的 TTFS 编码位置”。

### 4.5 第 0 步：先把 block 输入编码成 TTFS 张量

假设 ANN 中某一层输入范围是：

```text
blocks.0.monitor_0 = (-2.0, 3.0)
```

如果某个元素的 ANN 值是 `x = 1.0`，那么它会被编码成：

```text
t = (1.0 - (-2.0)) / (3.0 - (-2.0)) = 0.6
```

所以：

- ANN 看的是 `1.0`
- SNN 张量里保存的是 `0.6`

整个 block 输入张量 `x` 经过编码后，形状不变，仍然是：

```text
[B, 197, 384]
```

只是语义从“实值激活”变成了“TTFS 码”。

### 4.6 第 1 步：`norm1` 从 ANN 变成 `SpikingLayerNorm`

这是 Attention 前最重要的一步。

`SpikingLayerNorm.forward()` 的过程可以直接按代码理解：

#### 1. 先解码

```python
x = ttfs_to_value(t, self.in_range)
```

也就是把 `[0, 1]` 的 TTFS 码还原回原始实值区间 `blocks.0.monitor_0`。

此时：

- 输入形状还是 `[B, 197, 384]`
- 语义重新变回 ANN 值

#### 2. 做减均值

```python
mean = x.mean(self.dim, keepdim=True)
x.sub_(mean)
```

这一步对应论文 LayerNorm 分解里的“先减均值”。

然后立刻把中间结果编码到 `blocks.0.norm1.monitor_1`：

```python
t = value_to_ttfs(x, self.x_range)
t = emit(t)
```

所以这时得到的是：

- 形状：`[B, 197, 384]`
- 语义：减均值后的 TTFS 码

#### 3. 算逆标准差

代码会再次解码刚才的中间结果，然后计算：

```python
inv_std = (x.var(...) + eps) ** (-1 / 2)
```

得到的 `inv_std` 形状是：

```text
[B, 197, 1]
```

因为它是对最后一个归一化维度做方差统计。

随后它会被编码到 `blocks.0.norm1.monitor_2`。

#### 4. 用 Hadamard 乘法完成缩放和仿射变换

代码最后调用：

```python
out = ops.hadamard(
    t, inv_std,
    self.x_range,
    self.inv_std_range,
    self.out_range,
    self.config,
    self.weight,
    self.bias
)
```

这一步等价于：

```text
out = (x - mean) * inv_std * gamma + beta
```

只是实现方式是：

1. TTFS 张量先解码
2. 在 `hadamard()` 里用论文的乘法思路计算逐元素乘法
3. 再乘 `gamma`、加 `beta`
4. 最后编码到 `blocks.0.monitor_1`

于是 `norm1` 的输出就是：

- 形状：`[B, 197, 384]`
- 语义：LayerNorm 输出的 TTFS 码
- 区间：`blocks.0.monitor_1`

### 4.7 第 2 步：`qkv` 线性层变成 `SpikingWrapped(Linear)`

`SpikingAttention` 中先执行：

```python
qkv = self.qkv(x)
```

这里的 `self.qkv` 是 `SpikingWrapped(nn.Linear)`。

它的内部过程非常简单：

1. 用 `attn.monitor_0` 解码输入。
2. 调用原始 `nn.Linear`。
3. 用 `attn.monitor_1` 编码输出。
4. 调用 `emit()` 做量化或加噪。

因此：

- 输入形状：`[B, 197, 384]`
- 输出形状：`[B, 197, 1152]`
- 输出语义：`qkv` 的 TTFS 码
- 输出区间：`blocks.0.attn.monitor_1`

### 4.8 第 3 步：reshape 出 `q / k / v`

随后执行：

```python
qkv = torch.reshape(qkv, (B, N, 3, num_heads, C // num_heads)).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
```

这一步只改视图，不改数值语义。

于是：

- `q.shape = [B, 6, 197, 64]`
- `k.shape = [B, 6, 197, 64]`
- `v.shape = [B, 6, 197, 64]`

而且三者都仍然处在同一个范围：

```text
blocks.0.attn.monitor_1
```

### 4.9 第 4 步：`q @ k^T` 变成 `ops.matmul`

原始 ANN 是：

```python
attn = q @ k.transpose(-2, -1)
attn *= scale
```

转换后是：

```python
attn = ops.matmul(q, k.transpose(-2, -1), ..., scale=self.scale)
```

`ops.matmul()` 做了四件事：

1. 把 `q` 和 `k` 从 TTFS 码解码回实数。
2. 调用 `spiking_matmul_core()` 做矩阵乘。
3. 乘上 attention 的缩放系数 `scale`。
4. 把结果编码到 `blocks.0.attn.monitor_2`。

这里的输出张量形状变成：

```text
[B, 6, 197, 197]
```

语义上它是 attention score 的 TTFS 码。

这一步非常关键，因为它说明：

- 论文里最难的“乘法/矩阵乘”并没有被跳过。
- 仓库里确实为它写了专门的 SNN 运算路径。

### 4.10 第 5 步：`softmax` 变成 `SpikingSoftmax`

原始 ANN 的 `softmax(attn)`，在 SNN 里变成：

```python
attn = self.softmax(attn)
```

`SpikingSoftmax` 的具体步骤是：

1. 先按 `attn.monitor_2` 解码 score。
2. 做 `x -= max(x)`，保证数值稳定。
3. 计算 `lse = log(sum(exp(x)))`。
4. 把这个 `lse` 编码到 `attn.softmax.monitor_1`，并经过 `emit()`。
5. 再解码回 `lse`。
6. 执行 `x -= lse`，然后 `exp()`。

这里有个非常重要的实现细节：

- softmax 的最终输出本身就是概率
- 概率天然落在 `[0, 1]`
- 所以下一步 `attn @ v` 在代码里直接把 attention 权重的输入范围写成 `(0, 1)`

也就是说，softmax 输出之后：

- 张量形状：`[B, 6, 197, 197]`
- 张量语义：attention 概率
- 数值范围：近似 `[0, 1]`

### 4.11 第 6 步：`attn @ v` 再次走 `ops.matmul`

接下来：

```python
x = ops.matmul(
    attn,
    v,
    (0, 1),
    self.param.at("monitor_1"),
    self.param.at("monitor_4"),
    self.config,
)
```

这意味着：

- `attn` 按 `(0, 1)` 解码
- `v` 按 `attn.monitor_1` 解码
- 输出编码到 `attn.monitor_4`

输出张量形状：

```text
[B, 6, 197, 64]
```

这对应每个 head 聚合后的上下文向量。

### 4.12 第 7 步：合并 heads，再过输出投影 `proj`

然后：

```python
x = x.transpose(1, 2).reshape(B, N, -1)
x = self.proj(x)
```

所以：

- 先从 `[B, 6, 197, 64]`
- 回到 `[B, 197, 384]`
- 再经过 `SpikingWrapped(proj)`
- 输出编码到 `attn.monitor_o`

此时 attention 子层输出已经完成。

### 4.13 第 8 步：残差加法 `x + tmp` 变成 `ops.add`

在 `SpikingBlock.forward()` 里：

```python
x = ops.add(
    x, tmp,
    self.param.at("monitor_0"),
    self.param.at("monitor_2"),
    self.param.at("monitor_3"),
    self.config
)
```

这一步的意义是：

1. 原始 block 输入 `x` 按 `blocks.0.monitor_0` 解码
2. attention 输出 `tmp` 按 `blocks.0.monitor_2` 解码
3. 两者做实数加法
4. 再编码到 `blocks.0.monitor_3`

因此残差后的输出：

- 形状还是 `[B, 197, 384]`
- 但范围已经切换到新的 `blocks.0.monitor_3`

这一步很关键，因为它说明在这个仓库里：

- 残差连接也不是直接在“时间值”上乱加
- 而是严格先回到同一实值语义，再重新编码

---

## 5. 把上面过程压缩成一句“总公式”

对于这个仓库的大多数层，都可以用下面这句来理解：

```text
TTFS张量
  -> 按该层输入区间解码成实值
  -> 用原ANN权重/公式完成这一层计算
  -> 按该层输出区间重新编码成TTFS张量
  -> 用 emit() 加入精度限制或噪声
```

只有 `LayerNorm`、`Softmax`、`matmul`、`hadamard` 这类比较复杂的层，会拆出更多中间区间。

---

## 6. 转换后的“数据结构”到底是什么

这是最容易误解的地方，我单独说清楚。

### 6.1 不是事件列表，而是稠密张量

转换后运行时传递的仍然是 PyTorch 稠密张量，例如：

```text
Tensor[B, 197, 384]
Tensor[B, 6, 197, 197]
Tensor[B, 6, 197, 64]
```

区别只在于：

- ANN 中这些张量表示激活值
- SNN 中这些张量表示 TTFS 编码值

### 6.2 参数字典是一个“监控点 -> 区间”的映射

`SpikeParamDict` 的底层就是：

```python
dict[str, tuple[float, float]]
```

例如：

```json
{
  "monitor_0": [-2.12, 2.64],
  "patch_embed.monitor_0": [-2.12, 2.64],
  "patch_embed.monitor_1": [-6.83, 5.41],
  "blocks.0.monitor_0": [-4.91, 5.27],
  "blocks.0.norm1.monitor_1": [-2.65, 2.73],
  "blocks.0.attn.monitor_2": [-14.20, 13.88]
}
```

`child("blocks.0")` 会把这个大字典裁成当前 block 的局部视图。

### 6.3 模型权重没有重训，而是直接继承

例如：

- `SpikingWrapped(inner=ann_linear, ...)` 里面直接持有原始 `nn.Linear`
- `SpikingLayerNorm` 直接复用原 `LayerNorm` 的 `weight` 和 `bias`
- `SpikingVisionTransformer` 直接复用原模型里的 `cls_token`、`pos_embed`

也就是说，转换时最核心的新信息不是“新训练出来的权重”，而是：

1. 原始 ANN 权重
2. 每个监控点的范围 `(a, b)`
3. 可选的发放精度配置 `SpikeConfig`

### 6.4 `SpikeConfig` 决定“理想 TTFS”要不要量化

`SpikeConfig.emit()` 有三种模式：

1. `precision is not None`
   把 TTFS 码量化到固定精度，再裁到 `[0, 1]`
2. `noise is not None`
   给 TTFS 码加噪声，再裁到 `[0, 1]`
3. 都不设
   直接返回

所以这个仓库既能做“近似理想 TTFS”模拟，也能做“有限时间精度”模拟。

---

## 7. 一个最小数值例子，帮助建立直觉

假设某层监控到输入范围是：

```text
(a, b) = (-2, 3)
```

某个神经元的 ANN 激活值是：

```text
x = 1
```

编码后：

```text
t = (1 - (-2)) / (3 - (-2)) = 0.6
```

如果设置 `precision = 16`，那么 `emit()` 会把它量化成：

```text
round(0.6 * 16) / 16 = 10 / 16 = 0.625
```

再解码回来：

```text
x' = 0.625 * 5 - 2 = 1.125
```

这就说明：

- 不开量化时，SNN 张量可以非常贴近原 ANN 数值
- 开量化后，误差主要来自 TTFS 发放精度
- 论文 Figure 4 研究的就是这种精度退化下的准确率变化

---

## 8. 最后用一句话总结这个仓库的 ANN2SNN

这份实现的核心思想可以概括成：

**先用 ANN 跑出每个关键节点的值域，再把每个节点都变成“以 TTFS 码为接口、以原 ANN 运算为内核”的 SNN 模块，从而让 Transformer 的 LayerNorm、Softmax、Attention、Residual 都能在统一的 TTFS 表示下逐层接起来。**

如果后面你要继续深挖，我建议下一步就看两个地方：

1. `SpikingLayerNorm`，因为它最能体现论文怎么把复杂非线性拆开。
2. `ops.matmul` + `spiking_matmul_core`，因为它对应 Transformer 注意力里最关键的乘法结构。
