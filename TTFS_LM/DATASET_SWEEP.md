# Qwen2.5 TTFS cross-corpus sweep

Run date: 2026-08-29

This experiment keeps the MVP conversion fixed: Qwen2.5-0.5B, context 128, hybrid LM head, and one set of ranges
calibrated from 100,000 WikiText-2 train tokens with margin 1.0. No evaluation activation is added to the registry.
Dataset provenance, licenses, materialized hashes, and packing semantics are documented in
[`DATASETS.md`](DATASETS.md).

## Corpora

| Name | Pinned source | Evaluation scope | Qwen tokens |
| --- | --- | --- | ---: |
| WikiText-103 | `Salesforce/wikitext@b08601e`, `wikitext-103-raw-v1` | complete test split | 299,241 |
| Penn Treebank | `wojzaremba/lstm@7687025`, `data/ptb.test.txt` | complete test file | 99,256 |
| LAMBADA OpenAI | `EleutherAI/lambada_openai@900124b` | complete default test split | 396,547 |
| C4 English | `allenai/c4@1588ec4`, first validation shard | first 262,420 tokens | 262,420 |

The C4 gzip is additionally pinned by SHA256
`1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f`.

## Results

All four strict continuous runs stopped at their first out-of-range element. The explicit-clipping diagnostic then
completed the same corpus and shows that these failures are extremely sparse outliers rather than broad continuous-time
degradation.

| Corpus | ANN PPL | Clipped continuous PPL | Relative PPL change | Top-1 | Clips / encoded elements |
| --- | ---: | ---: | ---: | ---: | ---: |
| WikiText-103 | 28.080128 | 28.080084 | `-1.55e-6` | 99.9987% | 4 / 242,102,407,930 |
| Penn Treebank | 77.913091 | 77.911432 | `-2.13e-5` | 99.9960% | 13 / 80,302,133,450 |
| LAMBADA OpenAI | 41.968384 | 41.968106 | `-6.63e-6` | 99.9997% | 11 / 320,828,158,122 |
| C4 English | 29.992336 | 29.992325 | `-3.50e-7` | 99.9992% | 2 / 212,311,813,548 |

The first strict failures are concentrated in layer 22 RMSNorm square ranges. WikiText-103, LAMBADA, and C4 first
exceed `model.layers.22.input_layernorm.square`; PTB first exceeds
`model.layers.22.post_attention_layernorm.square`. Later diagnostic violations remain concentrated in layers 21-23.
Because those rare internal activations are clipped, the maximum logits error is still large (`8.42` to `10.85`) even
though aggregate PPL and top-1 barely move. The diagnostic therefore supports aggregate quality, not strict numerical
equivalence.

`P=4096` completed on every corpus but remains unusable:

| Corpus | P=4096 PPL | Top-1 | Clipped ratio | Violating range roles |
| --- | ---: | ---: | ---: | ---: |
| WikiText-103 | `6.30e106` | 0.1821% | 0.8998% | 149 |
| Penn Treebank | `4.04e107` | 0.1531% | 0.9003% | 134 |
| LAMBADA OpenAI | `3.95e106` | 0.1440% | 0.8695% | 145 |
| C4 English | `2.99e107` | 0.1589% | 0.8885% | 142 |

The official Hugging Face and local ANN logits remained exactly equal on the 99,999-token calibration check. The shared
cache test had max-absolute error `1.1587e-4` for both ANN and continuous TTFS, and eight-token generation agreement was
100%. The full offline sweep completed in 1,373.10 seconds.

## Interpretation

1. The functional continuous conversion remains high fidelity after explicit clipping, across all tested corpora.
2. The strict zero-clipping contract does not generalize from a 100K-token, single-domain min/max calibration set.
3. A few late-layer norm/residual outliers should be handled with robust range design or targeted granularity rather than
   by silently relaxing the global contract.
4. Uniform `P=4096` error compounds across layers and is not repaired by the larger guard band.

## Scope limits

- LAMBADA is measured here as causal corpus PPL, not the benchmark's last-word accuracy.
- C4 is a deterministic sample from one pinned validation shard, not the complete 364K-document validation split.
- PTB is the preprocessed, lower-cased text evaluated with the Qwen tokenizer. It is not comparable to traditional
  word-level PTB perplexity.
- The experiment covers English language-model corpora. Downstream reasoning tasks, code, Chinese, and long-context
  benchmarks require separate evaluation protocols.

Raw artifacts are under `TTFS_LM/artifacts/dataset_sweep/`: `ranges.json`, `metrics.json`, `summary.json`,
`sweep_manifest.json`, and `run.log`.
