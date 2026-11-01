# TTFS LLM

把 **TTFSFormer（ICML 2025，视觉 Transformer 的无损 ANN→TTFS 转换）** 迁移到**因果语言模型**上：
让每个激活标量只用一个首脉冲时间码表示，并回答一个上游没回答的问题——

> **在固定的时间分辨率 / 位宽 / 存储预算下，能保住多少模型质量？**

上游问的是"能不能无损"（一个恒等式），这里问的是"预算下能保住多少"（一个工程问题）。

---

## 仓库内容

| 目录 | 是什么 | 来源 |
|---|---|---|
| `ttfsformer/` | 视觉 SNN 算子、转换器、ViT/EVA 的脉冲实现 | ICML'25 官方代码，原样保留 |
| `test/` | 视觉复现入口：`eval_origin.py`（ANN 取参数）、`eval_ttfs.py`（SNN 评测） | 同上 |
| `TTFS_LM/` | **本项目新增**：Qwen2.5 的 ANN→TTFS 功能转换、校准、因果 KV cache、跨语料与前推理评测 | 本项目 |
| `assets/` | 论文 PDF、汇报幻灯片 | — |
| `tools/` | 幻灯片生成脚本 | 本项目 |

文档不在仓库里，见下文「文档在哪」。

## 三十秒结论

| 合同 | 状态 | 证据 |
|---|---|---|
| **① 表示合同**：每个激活 → `[0,1]` 归一化时间码，可逆 | ✅ 成立 | Continuous 模式 PPL 29.258942670，与官方 ANN 29.258942438 等价 |
| **② 有限精度合同**：`t = round(P·t)/P` | ❌ 失败 | `P=4096` 时 PPL 崩到 `7.23e106`，top-1 掉到 0.15% |
| **③ 事件执行合同**：不恢复稠密实值、按稀疏事件执行 | ⬜ 不存在 | 当前所有算子都是 decode → 稠密 PyTorch → encode |

**因此本仓库当前的定位是：研究基础设施 + 失效证据，而不是一个已成立的方法。**
它证明了"Qwen 的语义能塞进时间码"，同时也证明了"简单均匀有限精度在 24 层上会级联崩溃"。

## 关键数字

模型：`Qwen/Qwen2.5-0.5B` Base（24 层 / hidden 896 / 14 Q heads / 2 KV heads / GQA），
校准：WikiText-2 train 前 100K token，631 个 RangeSpec，guard band γ=1.0，评测：WikiText-2 validation 全量。

| 设置 | PPL | Logits max-abs | Top-1 一致率 | 裁剪元素 |
|---|---:|---:|---:|---:|
| 官方 HF ANN | 29.258942438 | 0 | 100% | — |
| 本地 ANN（本仓库实现） | 29.258942438 | 0 | 100% | — |
| Continuous TTFS | 29.258942670 | 5.82e-5 | 100% | 0 |
| Finite TTFS `P=4096` | 7.23e106 | 470.97 | 0.1528% | 1.91e9 / 2.12e11（0.90%） |

跨语料（WikiText-103 test / PTB test / LAMBADA OpenAI test / C4 val 262K）：
strict 零裁剪合同 **0/4 通过**，但每个语料仅 2–13 个元素越界；显式裁剪后 4/4 的 PPL 相对变化 < 2.2e-5。
**表示本身是稳的，崩的是均匀有限精度。**

## 安装

```bash
# 视觉部分（原版）
python -m pip install -e .

# 语言部分
python -m pip install -e ./TTFS_LM
```

Python ≥ 3.10。视觉部分按原版要求；语言部分实测环境为 PyTorch 2.11 / Transformers 5.6 / Safetensors 0.7 /
huggingface-hub 1.13。需要镜像时在启动 Python 前 `export HF_ENDPOINT=https://hf-mirror.com`。

权重不入库：视觉权重按下方表格从 Hugging Face 取；Qwen 权重走 Hugging Face 缓存。

## 快速开始

### 1. 视觉基线（验证上游能复现）

```bash
python test/eval_origin.py     # 跑 ANN，并导出 SNN 需要的逐层区间参数
python test/eval_ttfs.py       # 在 SNN 上评测
```

权重（放到 `weight/<name>/`）：

| 模型 | Checkpoint |
|---|---|
| ViT-S/16 | [timm/vit_small_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k) |
| ViT-B/16 | [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k) |
| ViT-L/16 | [timm/vit_large_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k) |
| EVA02-S | [timm/eva02_small_patch14_336.mim_in22k_ft_in1k](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k) |
| EVA02-L | [timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k) |
| EVA-G | [timm/eva_giant_patch14_224.clip_ft_in1k](https://huggingface.co/timm/eva_giant_patch14_224.clip_ft_in1k) |

### 2. 语言基线

```bash
# 单元测试
python -m unittest discover -s TTFS_LM/tests -v

# WikiText-2 MVP：官方 ANN → 本地 ANN → train-only 校准 → Continuous → P=4096 → 8-token 生成
ttfs-lm mvp --config TTFS_LM/configs/mvp_wikitext2.yaml --local-files-only

# 跨语料 sweep（复用同一份 WikiText-2 ranges，评测语料绝不参与校准）
ttfs-lm dataset-sweep --config TTFS_LM/configs/dataset_sweep.yaml --local-files-only

# 1.5B 上的 GPQA-Diamond / 推理评测
ttfs-lm gpqa-diamond \
  --model-id Qwen/Qwen2.5-1.5B \
  --revision b4b7d0ff85668c78171cba0bd02714ee6ebf6e69 \
  --precision 4096 --allow-clipping
```

产物落在 `TTFS_LM/artifacts/`（已 gitignore）：`ranges.json`、`metrics.json`、`run_manifest.json`、`run.log`。
每个 pipeline 都有显式 acceptance gate，失败时非零退出，不静默放宽区间。

当前机器上保留的四组产物（本文所有数字的 provenance，不入库）：

| 目录 | 内容 |
|---|---|
| `artifacts/mvp_wikitext2/` | 0.5B WikiText-2 MVP：校准 ranges、PPL、logits 对齐、cache 检查 |
| `artifacts/dataset_sweep/` | 0.5B 跨语料 sweep（WikiText-103 / PTB / LAMBADA / C4） |
| `artifacts/dataset_sweep_qwen1.5b/` | 1.5B sweep 首轮 |
| `artifacts/dataset_sweep_qwen1.5b_rerun_20260914/` | 1.5B sweep 重跑（数值尚未通过 provenance 验收） |

细节（契约、逐模块转换、accepted tolerance、数据集 provenance）见 [`TTFS_LM/README.md`](TTFS_LM/README.md)、
[`TTFS_LM/DATASETS.md`](TTFS_LM/DATASETS.md)、[`TTFS_LM/DATASET_SWEEP.md`](TTFS_LM/DATASET_SWEEP.md)。

## 分支与版本策略

只有 **`master`** 一条长期分支，且它始终可跑通上面四条命令。

| 分支 | 用途 | 寿命 |
|---|---|---|
| `master` | 唯一长期分支，已验证的代码 | 永久 |
| `feat/<topic>` | 新功能 / 新算子 | 合并即删 |
| `exp/<topic>` | 实验性跑批（配置 + 脚本），结果不保证通过 acceptance | 结论成型后合并或删除 |

当前历史里的 `exp/dataset-sweep`、`exp/qwen-large-bench` 均已并入 `master` 并删除；
1.5B 评测（`ttfs_lm/reasoning_eval.py`、`configs/dataset_sweep_qwen1.5b.yaml`）的**代码**在 `master`，
但**数值尚未通过 provenance 验收**，引用时须注明。

## 文档在哪

**文档不在本仓库。** 原 `docs/` 目录已于 2026-11-01 迁出并归档到本地 Obsidian 知识库：

```
Knowledge/Research/LLM/TTFSLM/
├── README.md            总索引
├── TTFSLM.md            方法主文档（55K，逐模块转换分析）
├── Qwen2.5 TTFS 方案的创新性.md   创新性分级与上游差异
├── 源码文档/             原 docs/ 的完整归档（复现、审计、能耗、实验记录）
├── 202610思考/           事实与证据边界
└── 202611设计/           新方案 CLASP、实验矩阵与预注册放弃判据
```

历史版本仍可从 git 取回：

```bash
git log --diff-filter=D --name-only -- docs/    # 列出曾被删除的文档
```

仓库里只保留 **会跟着 commit 走** 的东西：代码、`TODO.md`、以及各子目录里描述契约与命令的 README。

## 不要过度解读

- 当前 `TTFS_LM` 是 **功能仿真器**：`decode → 稠密 PyTorch → encode`，没有事件驱动执行。
- 因此**没有任何**延迟、能耗、存储或稀疏性收益的证据。幻灯片里的能耗是**理论路径推导**，不是实测。
- `P=4096` 的结果只说明"有限编码能跑完且如实上报"，不是质量声明。
- 视觉上的有限精度几乎不掉点，语言上崩 100 个数量级——这个差异本身才是本项目最主要的发现。

## 上游

视觉代码来自 TTFSFormer（ICML 2025）：

> TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer.
> [Proceedings of ICML 2025](https://proceedings.mlr.press/v267/zhao25n.html)

论文 PDF 见 [`assets/TTFSFormer_ICML2025.pdf`](assets/TTFSFormer_ICML2025.pdf)。
