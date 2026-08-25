# TTFSFormer 论文结果复现记录

本文记录对论文 *TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer*
（ICML 2025）官方代码的端到端复现。记录区分“论文报告值”和“本机实测值”，不以模型卡
指标代替本仓库代码的实际输出。

## 1. 复现目标

论文的主要可复现实验分为三组：

1. 表 1：ImageNet-1K 上 6 个 ANN 与精确 TTFS-SNN 的 Top-1 准确率。
2. 图 4：ViT-S/16、ViT-B/16 在不同 TTFS 时间精度下的准确率。
3. 表 2：ViT-S/B/L 的 SNN 操作数与能耗估算。

表 1 中需要核对的论文原值如下。

| 模型 | 论文 ANN Top-1 | 论文 SNN Top-1 |
| --- | ---: | ---: |
| ViT-S/16 | 81.38% | 81.40% |
| ViT-B/16 | 85.10% | 85.07% |
| ViT-L/16 | 85.83% | 85.78% |
| EVA-G | 88.88% | 88.90% |
| EVA02-S | 85.72% | 85.68% |
| EVA02-L | 90.05% | 90.03% |

论文表 2 报告 ViT-S/B/L 的 `OPSNN` 分别为 5.42、19.2、65.7 GOP，估算能耗分别为
4.9、17、59 mJ，相对同架构 ANN 能耗分别为 22.3%、21.0%、20.7%。

## 2. 代码与硬件

- 官方仓库发布提交：`369551dc1382084b4eb62b7c6fe62a2c7a2129b5`
- 当前 Git HEAD：`0953b1c`；该已提交版本相对发布提交只增加了文档和论文 PDF。
- 本次未提交工作树另含复现辅助改动：连续数据分片、可配置 worker/输入输出路径、结构化指标、
  shard 合并器，以及不改变数学定义的监控归约性能优化。原始默认单 shard、批大小和模型计算路径
  保持不变；所有辅助改动均在本文给出验证。
- GPU：8 x NVIDIA GeForce RTX 5090，每张显存 32607 MiB。
- 驱动：580.126.20。
- 系统内存：1.0 TiB。
- 当前 Python：3.13.11。
- PyTorch：2.11.0+cu130。
- torchvision：0.26.0+cu130。
- timm：1.0.15。
- safetensors：0.7.0。

README 声明代码在 Python 3.12 下测试、最低支持 3.10，但项目没有锁定依赖版本。本机没有现成
Python 3.12 环境，因此先使用 Python 3.13.11；这是与作者环境的已知差异。`timm==1.0.15`
取自代码发布同期，并已验证 6 个模型名均被本仓库实现正确注册。

安装命令：

```bash
python -m pip install -e . 'timm==1.0.15'
```

为诊断 EVA02-S 的偏差，另建了 Python 3.12.13、PyTorch 2.7.1、torchvision 0.22.1、
`timm==1.0.15` 环境。普通 PyPI 的 PyTorch 2.7.1 是 `+cu126` 构建，只包含到 `sm_90`；
在 RTX 5090（`sm_120`）上执行首个 CUDA matmul 即报 CUDA kernel image 不可用错误。因此诊断
环境必须改装官方 `+cu128` wheel。这个组合接近代码发布时期，但作者
没有公开其 PyTorch/CUDA 小版本，故只能作为版本敏感性实验，不能称为作者原始环境。

## 3. 数据集核验

实际使用路径：

```text
/data/xuep/datasets/ImageNet1K/val
```

核验结果：1000 个类别目录、50000 张图像、占用 6.4 GiB。目录名采用 ImageNet synset，
例如 `n01440764`，满足 `torchvision.datasets.ImageFolder` 按类别名排序建立标签的要求。

核验命令：

```bash
find /data/xuep/datasets/ImageNet1K/val -mindepth 1 -maxdepth 1 -type d | wc -l
find /data/xuep/datasets/ImageNet1K/val -type f | wc -l
```

另一个路径 `/data/dataset/ImageNet1K/val` 虽然存在，但当前账号没有读取权限，未使用。

## 4. 权重准备

README 指定 Hugging Face `timm` 组织下的 6 个权重仓库。当前机器无法直连
`huggingface.co:443`，因此使用 `hf-mirror.com` 下载相同仓库文件：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  hf download timm/vit_small_patch16_224.augreg_in21k_ft_in1k \
  config.json model.safetensors \
  --local-dir weight/vit_small_patch16_224.augreg_in21k_ft_in1k
```

已完成权重的文件校验：

```text
模型        config.json                                                       model.safetensors
ViT-S/16    c5d0037088681d8fe1a92a05f8ec6be7002d992ed213aed6ab97b55d90b8756b 79c03c635cdfd798a364a9d8c4e5c0b7255b975ea2c9616046d4f77ab01435aa
ViT-B/16    b2d2d7d483d964889a9bdf7fcad9da6f3dd7fd3ffdf05def4807a29980ce6a16 32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b
ViT-L/16    fa20ba5d70fa0a0e085ea88daeb51c4777b376a819d965a2378645b3f014d552 109390825a5bada2864f1b74445a63d29c51ef079c349644bc98996c452f6cf1
EVA02-S     7e389763d4aca14c544401d6a64607425c0fb5c5eaf5093035a287faead36038 518a27f41de1d81eda6a6102fcca2bb0c4697720ef13911ad77e2cc374df99b4
EVA02-L     3fd38abc37443bc6e8a5edaf91e5e1f16a5d3dacd0d8c29eaadf172539d6c80b 1b54182f1f92613b96bc9e325f3795bde38bd1708233bb77a07d5c52eed2d638
EVA-G       9dd6fa95f37173b8dcbdddab17e24589ab73f7872dc694eab62c78b590aa4fe4 0786c919b43f33fa52671abd9b9a00d707fa9435d309bd1274b8b91fe25befcd
```

该配置规定 `224 x 224` 输入、bicubic 插值、`crop_pct=0.9`、中心裁剪、均值和标准差均为
`[0.5, 0.5, 0.5]`。实际预处理由 `imagenet_loader()` 读取各权重目录的 `config.json`
生成，不手工覆盖。

## 5. ANN 校准与基线评测

`eval_origin.py` 一次遍历完整 ImageNet-1K 验证集，同时做两件事：

1. 计算 ANN Top-1/Top-5。
2. 收集每个转换点的数值区间并写入 `data/<architecture>/monitor-100.json`。

`eval_origin.py` 显式调用 `set_monitor(model, lambda: MinMaxMonitor())`，因此 ViT 和 EVA 都是：
先取每个批次的最小/最大值，再对全部批次的极值求平均。EVA 类中虽然另有一个会安装
`RangeMonitor` 的 `monitor()` 方法，但该评测入口没有调用它。虽然输出文件统一叫
`monitor-100.json`，原脚本也没有只处理 100 个样本或 100 个批次。

为缩短大模型的墙钟时间，本次给评测入口增加了可选的连续数据分片参数。默认值仍是单进程，
不改变作者命令。50000 张图按 4 shard 切分后，每个 shard 恰好包含 12500 张、250 个完整批次；
最终准确率按样本数加权，监控区间按 batch 数加权。以 EVA02-S 为例，4 shard 合并区间与单进程
完整校准的 724 个边界中有 633 个逐位相同，最大绝对差只有 `2.84e-14`。

```bash
CUDA_VISIBLE_DEVICES=0 python test/eval_origin.py ... --num_shards 4 --shard_index 0
CUDA_VISIBLE_DEVICES=1 python test/eval_origin.py ... --num_shards 4 --shard_index 1
CUDA_VISIBLE_DEVICES=2 python test/eval_origin.py ... --num_shards 4 --shard_index 2
CUDA_VISIBLE_DEVICES=3 python test/eval_origin.py ... --num_shards 4 --shard_index 3

python test/merge_origin_shards.py \
  --monitor_inputs data/<ARCH>/monitor-100.shard-{0,1,2,3}-of-4.json \
  --metrics_inputs data/<ARCH>/origin-metrics.shard-{0,1,2,3}-of-4.json \
  --monitor_output data/<ARCH>/monitor-100.json \
  --metrics_output data/<ARCH>/origin-metrics.json \
  --monitor_mode mean
```

ViT-S/16 实际命令：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONWARNINGS=ignore \
  python test/eval_origin.py \
  --path /data/xuep/datasets/ImageNet1K/val \
  --weight weight/vit_small_patch16_224.augreg_in21k_ft_in1k \
  --batch_size 50 \
  > log/vit_small_origin.out 2>&1
```

## 6. 精确 TTFS-SNN 评测

精确实验不传 `--precision` 和 `--noise`。`SpikeConfig.emit()` 此时不量化也不加噪声，模型按
校准区间执行 TTFS 编码、算子计算和重新编码。输入先由 `monitor_0` 区间编码；最终输出仍为与
类别 logit 单调对应的编码值，因此直接取 Top-k，无需再解码。

命令模板：

```bash
CUDA_VISIBLE_DEVICES=<GPU> PYTHONWARNINGS=ignore \
  python test/eval_ttfs.py \
  --path /data/xuep/datasets/ImageNet1K/val \
  --weight weight/<WEIGHT_DIR> \
  --batch_size <BATCH_SIZE> \
  > log/<MODEL>_ttfs_precise.out 2>&1
```

六个模型的实际执行清单如下；全部保持 `batch_size=50`，分片数大于 1 时 ANN 区间均用
`--monitor_mode mean` 合并，SNN 指标均用 `merge_metrics.py` 按样本数合并。

| 模型 | 权重目录 | ANN shard | 精确 SNN shard | 最终数据目录 |
| --- | --- | ---: | ---: | --- |
| ViT-S/16 | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 1 | 1 | `data/vit_small_patch16_224/` |
| ViT-B/16 | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 1 | 1 | `data/vit_base_patch16_224/` |
| ViT-L/16 | `vit_large_patch16_224.augreg_in21k_ft_in1k` | 4 | 4 | `data/vit_large_patch16_224/` |
| EVA-G | `eva_giant_patch14_224.clip_ft_in1k` | 5 | 5 | `data/eva_giant_patch14_224/` |
| EVA02-S | `eva02_small_patch14_336.mim_in22k_ft_in1k` | 4 | 1 | `data/eva02_small_patch14_336/` |
| EVA02-L | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | 4 | 4 | `data/eva02_large_patch14_448/` |

以 5 shard 的 EVA-G 为例，实际并行命令为：

```bash
for shard in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$shard PYTHONWARNINGS=ignore \
    python test/eval_ttfs.py \
    --path /data/xuep/datasets/ImageNet1K/val \
    --weight weight/eva_giant_patch14_224.clip_ft_in1k \
    --batch_size 50 --num_workers 2 \
    --num_shards 5 --shard_index $shard --no_print_model \
    > log/eva_giant_ttfs_precise_shard${shard}.out 2>&1 &
done
wait

python test/merge_metrics.py \
  --inputs data/eva_giant_patch14_224/ttfs-precise-metrics.shard-{0,1,2,3,4}-of-5.json \
  --output data/eva_giant_patch14_224/ttfs-precise-metrics.json
```

## 7. 离散时间精度实验

代码中的 `--precision p` 执行：

```python
round(t * p) / p
```

并将结果裁剪到 `[0, 1]`。为复现图 4，需要至少对 ViT-S/16 和 ViT-B/16 执行论文图中给出的
`p = 4096, 1024, 512, 384`，并以不传 `--precision` 的精确结果作为 `Precise` 点。

```bash
python test/eval_ttfs.py ... --precision 1024
```

## 8. 能耗估算

论文使用 45 nm 参考能耗 `EAC=0.9 pJ`、`EMAC=4.6 pJ`，并按下式估算：

```text
ESNN / EANN = (OPSNN * EAC) / (OPANN * EMAC)
```

附录表 3 给出 TTFS 算子的操作数：矩阵乘 `MN(3K+2)`、Softmax `6N`、LayerNorm
`9N+1`、GELU/SiLU `MN`。本仓库没有随附逐层操作数统计脚本，因此表 2 属于解析复算，不是 GPU
功耗实测。本次增加的可执行复算命令为：

```bash
python test/reproduce_energy.py
```

由 `OPSNN * 0.9 pJ` 可直接得到：ViT-S 为 `5.42 * 0.9 = 4.878 mJ`，ViT-B 为
`19.2 * 0.9 = 17.280 mJ`，ViT-L 为 `65.7 * 0.9 = 59.130 mJ`，按论文精度舍入后就是
表 2 的 4.9、17、59 mJ。仓库和论文均未给出逐模型 `OPANN` 原始统计结果；只能结合论文能耗比
反推约 4.755、17.888、62.098 GOP。反推值用于一致性检查，不能声称为独立测量。

## 9. 实测结果

所有准确率均来自完整 50000 张验证图，不使用中途 tqdm 数值。

| 模型 | 本机 ANN Top-1 | 本机 ANN Top-5 | 本机精确 SNN Top-1 | 本机精确 SNN Top-5 | 与论文 SNN 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-S/16 | 81.406% | 96.146% | 81.406% | 96.146% | +0.006 pp |
| ViT-B/16 | 85.098% | 97.526% | 85.096% | 97.526% | +0.026 pp |
| ViT-L/16 | 85.844% | 97.822% | 85.844% | 97.822% | +0.064 pp |
| EVA-G | 88.878% | 98.680% | 88.898% | 98.684% | -0.002 pp |
| EVA02-S | 85.716% | 97.622% | 85.356% | 97.484% | -0.324 pp |
| EVA02-L | 90.052% | 99.054% | 90.038% | 99.056% | +0.008 pp |

三种 ViT 的精确 SNN 相对各自 ANN 的 Top-1 变化分别为 `0.000`、`-0.002`、`0.000` 个百分点。
它们与论文 SNN 报告值的绝对差均小于 0.1 个百分点。

EVA02-L 的四个 shard 均为 12500 张、250 个批次，分片 Top-1 分别为 91.320%、89.504%、
89.984%、89.344%；按样本数合并得到 90.038%。它相对本机 ANN 只下降 `0.014` 个百分点，
与论文精确 SNN 的差值为 `+0.008` 个百分点，同样满足 0.1 个百分点的复现判据。

EVA-G 的五个 shard 均为 10000 张、200 个批次，分片 SNN Top-1 分别为 90.580%、89.080%、
88.170%、88.880%、87.780%；合并得到 88.898%。它相对本机 ANN 提升 `0.020` 个百分点，
与论文精确 SNN 的差值为 `-0.002` 个百分点。

时间精度实验的当前实测值：

| 模型 | Precise | p=4096 | p=1024 | p=512 | p=384 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-S/16 | 81.406% | 81.272% | 79.880% | 73.912% | 66.228% |
| ViT-B/16 | 85.096% | 85.124% | 84.076% | 76.934% | 79.754% |

论文图 4 的 ViT-B 曲线在 `p=512` 约为 80%、`p=384` 约为 78%，随时间精度降低而下降。
本机的两个点却局部非单调。为排除并发、缓存或输出文件标错，已在同一张 GPU 上分别顺序重跑；
两次仍逐样本汇总为完全相同的 `76.934%` 和 `79.754%`，生成的 JSON 中 `precision` 字段也分别为
512 和 384。PyTorch 2.7.1 诊断环境又得到 76.640% 和 79.672%，仍然非单调。因此这两个点是
跨两个软件栈可重复但没有复现论文曲线的结果。

EVA02-S 的正确区间结果相对同次 ANN 下降 `0.360` 个百分点，超过论文所述小于 0.1 个百分点；
它也是当前环境下第一个没有复现论文“无损”结论的模型。为排除分片误差，已用单进程完整校准区间
复核；它与正确的 batch 均值 shard 合并区间最大绝对差仅 `2.84e-14`。

版本诊断使用 Python 3.12.13、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Triton 3.3.1
和相同的 `timm==1.0.15`。两套环境的 EVA02-S ANN 指标完全相同；2.7 与 2.11 校准区间的
724 个边界最大绝对差 `1.24e-5`、平均绝对差 `4.03e-7`。2.7 使用自身区间时 SNN 仍为
85.356%/97.484%，与 2.11 完全相同；2.7 使用 2.11 区间时为 85.358%/97.484%，只改变一张
图像的 Top-1 判定。因此该偏差不能由这两个 PyTorch 版本或微小校准差异解释。

EVA02-S 曾用错误的全局 `min/max` shard 合并方式得到 `85.388%`。该值对应被明显放宽的区间，
不是论文算法的有效复现结果；指标已明确改名为
`ttfs-precise-metrics.invalid-global-range.json`，仅用于说明问题定位过程。标准
`ttfs-precise-metrics.json` 和最终表格均采用正确的 `MinMaxMonitor` batch 均值区间重跑结果。

## 10. 判定标准与已知风险

- 论文声称精确 TTFS 转换的准确率损失小于 0.1 个百分点；先检查同一次复现中的
  `SNN Top-1 - ANN Top-1`，再检查与论文表 1 的绝对差异。
- 模型卡准确率不能作为成功证据，必须由仓库脚本在本地 50000 张验证图上实际得到。
- 作者未提供依赖锁文件、随机种子、原始监控 JSON 或各实验日志，逐层区间可能随数据读取实现、
  CUDA/PyTorch 数值细节和批大小变化。
- ViT 和 EVA 都对每批极值求平均，因此必须保留作者脚本默认 `batch_size=50` 才能得到同定义的
  区间。显存不足时不能无说明地修改批大小；可以像本次一样按完整 batch 边界切 shard，再按
  shard 的 batch 数加权合并。
- 日志中的 tqdm 中间准确率受按类别顺序读取影响，只有最终汇总行可用于结论。

最终判定：六个表 1 模型中，ViT-S/B/L、EVA-G、EVA02-L 共五个模型同时满足本机
`|SNN-ANN| < 0.1 pp` 且与论文 SNN 报告值相差小于 0.1 pp；EVA02-S 不满足，且在两套软件栈、
单进程/分片校准及交叉区间诊断下均稳定复现该偏差。图 4 的 ViT-S 走势基本复现，ViT-B 的
`p=512/384` 局部关系未复现。表 2 的能耗乘法可精确复算，但仓库缺少原始逐层统计器，因此
`OPSNN` 只能核对论文给定值，不能独立重建为逐层测量结果。

## 11. 输出与最终验证

- ANN 最终值：`data/<architecture>/origin-metrics.json`。
- 精确 SNN 最终值：`data/<architecture>/ttfs-precise-metrics.json`。
- 时间精度结果：`data/vit_{small,base}_patch16_224/ttfs-precision-<p>-metrics.json`。
- 完整 stdout/stderr：`log/*.out`；分片日志保留 shard 编号。
- `data/`、`log/` 和 `weight/` 受 `.gitignore` 管理，不提交数据集派生文件或大权重；本文表格是其
  可审计摘要。

最终验证命令：

```bash
python test/test_monitor.py -v
python test/test_dataset.py -v
python -m compileall -q test ttfsformer
python test/reproduce_energy.py
git diff --check
```
