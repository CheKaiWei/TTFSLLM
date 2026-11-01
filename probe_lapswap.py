"""LapSwap 可行性探针：把 softmax(QK^T) 改写成「距离 + 逐 key 偏置」后到底损失多少。

核心恒等式（本探针首先验证它）：
    s * q_i . k_j  =  s/2 * ( ||q_i||^2 + ||k_j||^2 - ||q_i - k_j||^2 )
行常数 s/2*||q_i||^2 在 softmax 里抵消，于是

    softmax_j( s q_i.k_j )  ==  softmax_j( s/2*||k_j||^2  -  s/2*||q_i-k_j||^2 )

右边把「通道维乘法」换成「逐通道相减 + 平方 + 累加」。在 T 级整数码上，
平方就是一张 T 项小表（T=15 时 16 项）的查表 —— 于是 QK 路径没有乘法器。

关键工程点（本探针要验证的三个）：
  a) 码本用「逐 head 百分位」而不是 min/max —— LLM 的 Q/K 有极端 outlier，
     min/max 会把 15 级网格几乎全部分配给 outlier，等效分辨率归零。
  b) ||k||^2 偏置用**未量化**的 key 精确算，每 key 每 head 只多存一个标量。
     它与距离是两条独立的量，不要为了省一个标量把它并进码本。
  c) 一个 head 一个共享标量尺度（q 与 k 同网格），这样
     ||q-k||^2 = c_h^2 * d2_code 严格成立，码域平方表才真的等于欧氏距离。

消融链（每一步只加一个误差源）：
    native       参考实现
    rbf_fp       fp32 上做上面的改写，不做任何量化      -> 应为恒等（验证代数）
    d2_T         逐 head 百分位码本 + 精确偏置 + softmax  -> 只含「量化」误差
    d2_T_2k      再加 2^k 行归一化                        -> 加「位移归一化」误差
    d2_T_nobias  丢掉 ||k||^2 偏置                        -> 量化 attention sink 值多钱
    i8dot        int8 对称量化 + 点积（对照组）

不改权重、不训练。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import ttfs_lm.modeling_ann as M
from ttfs_lm.checkpoint import load_official_ann, load_tokenizer

ABLATIONS = ("native", "rbf_fp", "d2_T", "d2_T_2k", "d2_T_nobias", "i8dot")


# ----------------------------------------------------------------------------- 码本
def codebook(query: torch.Tensor, key: torch.Tensor, pct: float):
    """逐 head 的共享区间。q 与 k 必须同一网格，距离才有意义。

    pct=None -> min/max；否则取 pct/1-pct 分位（抗 outlier）。
    返回 (lo, span)，形状 (1,H,1,1)。
    """
    H = query.shape[1]
    joint = torch.stack([query, key], 0)                       # (2,B,H,S,C)
    flat = joint.permute(2, 0, 1, 3, 4).reshape(H, -1)         # (H, 2*B*S*C)
    if pct is None:
        lo, hi = flat.amin(-1), flat.amax(-1)
    else:
        lo = torch.quantile(flat.float(), pct, dim=-1)
        hi = torch.quantile(flat.float(), 1.0 - pct, dim=-1)
    span = (hi - lo).clamp_min(1e-12)
    return lo.view(1, H, 1, 1), span.view(1, H, 1, 1)


def to_code(x: torch.Tensor, lo: torch.Tensor, span: torch.Tensor, levels):
    """返回 (码, 饱和度)。levels 可以是 int 或 (1,H,1,1) 张量（逐 head 不同位宽）。"""
    lv = levels if torch.is_tensor(levels) else torch.tensor(float(levels), device=x.device)
    raw = (x - lo) / span * lv
    z = torch.clamp(torch.round(raw), min=0.0)
    z = torch.minimum(z, lv)
    sat = float(((raw < -0.5) | (raw > lv + 0.5)).float().mean())
    return z, sat


def quant_i8(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().amax() / 127.0
    if float(scale) < 1e-12:
        return x
    return torch.round(x / scale).clamp_(-127, 127) * scale


# ----------------------------------------------------------------------------- 距离
def _pairwise(q, k, fn, chunk=128):
    B, H, S, _ = q.shape
    S2 = k.shape[2]
    out = torch.empty((B, H, S, S2), dtype=torch.float32, device=q.device)
    for i in range(0, S, chunk):
        d = q[:, :, i : i + chunk, None, :] - k[:, :, None, :, :]
        out[:, :, i : i + chunk, :] = fn(d)
    return out


def dist2(q, k):
    """平方欧氏距离。T 级整数码上 = 「查 T 项平方表 + 累加」，无乘法器。"""
    return _pairwise(q, k, lambda d: (d * d).sum(-1))


def dist1(q, k):
    return _pairwise(q, k, lambda d: d.abs().sum(-1))


# ----------------------------------------------------------------------------- 归一化
def norm_pow2(A: torch.Tensor, mask: torch.Tensor, step: float = 1.0):
    """Lapis 的 2 的幂归一化：除法 -> 位移。

    step=1 是 Lapis 原版（纯位移，行质量误差 ±41%）；
    step<1 把指数量化到 2^-step 网格（如 0.25 => 1 位移 + 4 项小表），误差收窄到 2^(step/2)。
    """
    Z = (A * mask).sum(-1, keepdim=True).clamp_min(1e-30)
    lg = torch.log2(Z)
    kk = torch.round(lg / step) * step
    W = torch.where(mask, A * torch.pow(2.0, -kk), torch.zeros_like(A))
    return W, W.sum(-1), float((lg - kk).abs().mean())


def stabilize(lg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """越界项置 -inf（否则 exp 会溢出成 inf，再乘 0 得 nan），然后行内减最大值。"""
    work = lg.masked_fill(~mask, float("-inf"))
    mx = work.amax(-1, keepdim=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return work - mx


# ----------------------------------------------------------------------------- 指标
def head_metrics(ref: torch.Tensor, tgt: torch.Tensor, rows: torch.Tensor):
    """ref,tgt: (B,H,S,S)；rows: (B,H,S)。返回逐 head 的 (TV, top1, 熵比, 行质量误差)。"""
    tv = 0.5 * (ref - tgt).abs().sum(-1)
    agree = (ref.argmax(-1) == tgt.argmax(-1)).to(torch.float32)
    ep = -(ref.clamp_min(1e-12).log() * ref).sum(-1)
    eq = -(tgt.clamp_min(1e-12).log() * tgt).sum(-1)
    mss = (tgt.sum(-1) - 1.0).abs()
    w = rows.to(torch.float32)
    n = w.sum(dim=(0, 2)).clamp_min(1)
    red = lambda t: (t * w).sum(dim=(0, 2)) / n
    return red(tv), red(agree), red(eq) / red(ep).clamp_min(1e-12), red(mss)


def sink_mass(p: torch.Tensor, rows: torch.Tensor, n: int = 4) -> float:
    w = rows.to(p.dtype)
    return float((p[..., :n].sum(-1) * w).sum() / w.sum().clamp_min(1))


# ----------------------------------------------------------------------------- 补丁
class Probe:
    """一个变体一次前向；前向内直接与 native 对比，不跨调用存中间量。"""

    def __init__(self, variant: str, levels: int, pct: float | None = None, taus=(), sink_n: int = 4,
                 skip: frozenset[int] = frozenset(), norm_step: float = 1.0, levels_map=None):
        self.variant = variant
        self.levels = levels
        self.pct = pct
        self.taus = taus
        self.sink_n = sink_n
        self.skip = skip
        self.norm_step = norm_step
        self.levels_map = levels_map or {}
        self.stats: dict[int, dict] = {}
        self.orig = None

    def __enter__(self):
        probe = self

        def bound(module, hidden_states, position_embeddings, position_ids,
                  current_valid, cache, observer, prefix):
            return probe._forward(module, hidden_states, position_embeddings, position_ids,
                                  current_valid, cache, observer, prefix)

        self.orig = M.QwenAttention.forward
        M.QwenAttention.forward = bound
        return self

    def __exit__(self, *exc):
        M.QwenAttention.forward = self.orig
        return False

    def _record(self, layer, ref, tgt, rows, extra=None):
        rec = self.stats.setdefault(layer, {})
        tv, top, ent, mss = head_metrics(ref, tgt, rows)
        rec.setdefault("tv", []).append(tv.tolist())
        rec.setdefault("top", []).append(top.tolist())
        rec.setdefault("ent", []).append(ent.tolist())
        rec.setdefault("mass", []).append(mss.tolist())
        rec.setdefault("sink", []).append(sink_mass(tgt, rows, self.sink_n))
        for k, v in (extra or {}).items():
            rec.setdefault(k, []).append(v)

    def _forward(self, self_, hidden_states, position_embeddings, position_ids,
                 current_valid, cache, observer, prefix):
        batch, seq_len, _ = hidden_states.shape
        query = self_.q_proj(hidden_states)
        key = self_.k_proj(hidden_states)
        value = self_.v_proj(hidden_states)
        H = self_.config.num_attention_heads
        query = query.view(batch, seq_len, H, self_.head_dim).transpose(1, 2)
        key = key.view(batch, seq_len, self_.config.num_key_value_heads, self_.head_dim).transpose(1, 2)
        value = value.view(batch, seq_len, self_.config.num_key_value_heads, self_.head_dim).transpose(1, 2)
        query, key = M.apply_rotary_pos_emb(query, key, *position_embeddings)

        if cache is not None:
            entry = cache.update(self_.layer_idx, key, value, position_ids, current_valid)
            key, value, key_positions, key_valid = entry.key, entry.value, entry.positions, entry.valid
        else:
            key_positions, key_valid = position_ids, current_valid
        key = M.repeat_kv(key, self_.num_key_value_groups)
        value = M.repeat_kv(value, self_.num_key_value_groups)
        valid = M.build_valid_attention_mask(position_ids, key_positions, key_valid)
        rows = (valid & current_valid[:, None, :, None]).any(-1).expand(batch, H, seq_len)
        s = self_.scaling
        layer = self_.layer_idx

        ref = M.masked_softmax(torch.matmul(query, key.transpose(2, 3)) * s, valid)
        probs, extra = ref, {}

        if layer in self.skip:          # 逐层选择：这些层保持原样（dense 点积 + softmax）
            context = torch.matmul(ref, value)
            context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
            return self_.o_proj(context)

        if self.variant == "i8dot":
            probs = M.masked_softmax(
                torch.matmul(quant_i8(query), quant_i8(key).transpose(2, 3)) * s, valid
            )
        elif self.variant == "rbf_fp":
            nk = (key * key).sum(-1)[:, :, None, :]
            probs = M.masked_softmax(0.5 * s * (nk - dist2(query, key)), valid)
        elif self.variant.startswith("d2_T") or self.variant == "dot_q":
            lv = float(self.levels_map.get(layer, self.levels))
            lo, span = codebook(query, key, self.pct)
            zq, sat_q = to_code(query, lo, span, lv)
            zk, sat_k = to_code(key, lo, span, lv)
            step = (span / lv).expand(-1, H, 1, 1)                     # 每码对应的值域宽度
            qd, kd = lo + zq * step, lo + zk * step                             # 解码（硬件只有码字）
            extra = {
                "sat": 0.5 * (sat_q + sat_k),
                "span_over_std": float((span / query.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-9)).mean()),
            }
            if self.variant == "dot_q":   # 对照：同一套码字，解码后老老实实做点积
                probs = M.masked_softmax(torch.matmul(qd, kd.transpose(2, 3)) * s, valid)
            else:
                d2 = dist2(zq, zk) * (step * step)                              # 回到值域单位
                nk = (kd * kd).sum(-1)[:, :, None, :]                           # 偏置必须来自解码码字
                bias = nk if self.variant != "d2_T_nobias" else 0.0
                lg = 0.5 * s * (bias - d2)
                if self.variant == "d2_T":
                    probs = M.masked_softmax(lg, valid)
                else:
                    probs, mass, logres = norm_pow2(torch.exp(stabilize(lg, valid)), valid, self.norm_step)
                    extra["rowmass"] = float((mass * rows.to(mass.dtype)).sum() / rows.sum().clamp_min(1))
                    extra["logres"] = logres
        elif self.variant in ("l1_T_2k", "l1_T_2k_b"):
            lo, span = codebook(query, key, self.pct)
            zq, _ = to_code(query, lo, span, self.levels)
            zk, _ = to_code(key, lo, span, self.levels)
            d1 = dist1(zq, zk)
            # 公平对照：ℓ1 也要保留 ||k||^2 的 sink 偏置（否则测的是"丢偏置"而不是"换核"）
            bias1 = 0.0
            if self.variant == "l1_T_2k_b":
                kd1 = lo + zk * (span / self.levels).expand(-1, H, 1, 1)
                bias1 = 0.5 * s * (kd1 * kd1).sum(-1)[:, :, None, :]
            sub = self.stats.setdefault(layer, {}).setdefault("tau", {})
            for tau in self.taus:
                W, _, _ = norm_pow2(torch.exp(stabilize(bias1 - d1 / tau, valid)), valid, self.norm_step)
                tv, top, ent, _ = head_metrics(ref, W, rows)
                sub.setdefault(tau, []).append({"tv": tv.tolist(), "top": top.tolist(), "ent": ent.tolist()})
            probs = W   # 最后一个 tau 参与前向，只为让 NLL 有值；形状质量看 stats

        if self.variant != "native":
            self._record(layer, ref, probs, rows, extra)

        context = torch.matmul(probs, value)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self_.o_proj(context)


# ----------------------------------------------------------------------------- 数据
def load_ids(tok, target_tokens: int):
    import pandas as pd

    parquet = Path.home() / (
        ".cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    )
    if not parquet.exists():
        raise SystemExit(f"missing {parquet}")
    text = "\n\n".join(t.strip() for t in pd.read_parquet(parquet)["text"].tolist() if t.strip())
    return tok(text, return_tensors="pt").input_ids[0][:target_tokens]


def run(model, ids, variant, seq_len, batches, levels, pct, taus, device, skip=frozenset(),
        norm_step=1.0, levels_map=None):
    total, n = 0.0, 0
    with Probe(variant, levels, pct, taus, skip=skip, norm_step=norm_step,
               levels_map=levels_map) as probe, torch.no_grad():
        for b in range(batches):
            chunk = ids[b * seq_len : (b + 1) * seq_len].unsqueeze(0).to(device)
            out = model(input_ids=chunk, labels=chunk)
            total += float(out.loss) * (seq_len - 1)
            n += seq_len - 1
    return total / n, probe.stats


def aggregate(stats):
    out = {}
    for layer, rec in sorted(stats.items()):
        row = {}
        for key in ("tv", "top", "ent", "mass"):
            if key in rec:
                row[key] = torch.tensor(rec[key]).mean(0).tolist()
        for key in ("sink", "rowmass", "logres", "sat", "span_over_std"):
            if key in rec:
                row[key] = sum(rec[key]) / len(rec[key])
        if "tau" in rec:
            row["tau"] = {
                str(tau): {k: torch.tensor([d[k] for d in vals]).mean(0).tolist()
                           for k in ("tv", "top", "ent")}
                for tau, vals in rec["tau"].items()
            }
            means = {t: sum(v["tv"]) / len(v["tv"]) for t, v in row["tau"].items()}
            best = min(means, key=means.get)
            row["best_tau"], row["best_tv"] = float(best), means[best]
        out[layer] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--levels", nargs="+", type=int, default=[15, 63, 255])
    ap.add_argument("--pct", type=float, nargs="+", default=[None],
                    help="码本分位；给 0 表示 min/max")
    ap.add_argument("--variants", default=",".join(ABLATIONS))
    ap.add_argument("--skip-layers", default="", help="逗号分隔的层号，这些层保持 dense")
    ap.add_argument("--norm-steps", type=float, nargs="+", default=[1.0],
                    help="2^k 归一的指数量化步长；1.0=Lapis 原版")
    ap.add_argument("--taus", default="10", help="L1 变体的温度；多个用逗号分隔")
    ap.add_argument("--levels-map", default="", help='逐层位宽，如 "0:1023,1:255"，其余层用 --levels')
    ap.add_argument("--output", default="probe_lapswap.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_official_ann(device=device, dtype="float32", local_files_only=True).eval()
    tok = load_tokenizer(local_files_only=True)
    ids = load_ids(tok, args.seq_len * args.batches)
    print(f"tokens={len(ids)} levels={args.levels} pct={args.pct} seq={args.seq_len} b={args.batches}", flush=True)

    skip = frozenset(int(x) for x in args.skip_layers.split(",") if x.strip() != "")
    taus = tuple(float(t) for t in args.taus.split(","))
    levels_map = {}
    for item in args.levels_map.split(","):
        if ":" in item:
            k, v = item.split(":")
            levels_map[int(k)] = float(v)
    print("skip_layers", sorted(skip), flush=True)
    result = {"seq_len": args.seq_len, "batches": args.batches, "skip_layers": sorted(skip), "runs": {}}
    nll_native, _ = run(model, ids, "native", args.seq_len, args.batches, 0, None, (), device)
    result["nll_native"], result["ppl_native"] = nll_native, math.exp(nll_native)
    print(f"native      nll={nll_native:.6f} ppl={result['ppl_native']:.4f}", flush=True)

    for step in args.norm_steps:
      for pct in args.pct:
        pct = None if (pct is None or pct <= 0) else pct
        for levels in args.levels:
            for variant in args.variants.split(","):
                if variant == "native":
                    continue
                nll, stats = run(model, ids, variant, args.seq_len, args.batches, levels, pct, taus,
                                 device, skip, step, levels_map)
                key = (f"{variant}@T{levels}@p{'mm' if pct is None else pct}@s{len(skip)}@n{step}"
                       + (f"@L{len(levels_map)}" if levels_map else ""))
                ppl = math.exp(nll)
                result["runs"][key] = {
                    "variant": variant, "levels": levels, "pct": pct, "nll": nll, "ppl": ppl,
                    "dppl_pct": 100.0 * (ppl / result["ppl_native"] - 1.0),
                    "layers": aggregate(stats),
                }
                print(f"{key:26s} nll={nll:8.4f} dPPL={result['runs'][key]['dppl_pct']:+10.3f}%", flush=True)

    Path(args.output).write_text(json.dumps(result, indent=2))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()
