# TTFS-LM: minimal Qwen2.5 ANN-to-TTFS baseline

This directory is the language-model development baseline described in
[`docs/LANGUAGE_MIGRATION_PLAN.md`](../docs/LANGUAGE_MIGRATION_PLAN.md). It implements a local Qwen2 decoder, loads the
official weights without renaming or approximating parameters, calibrates activation ranges, and converts the model to a
functional time-to-first-spike simulator.

## Baseline contract

The default checkpoint is the official base model
[`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B), pinned to revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`.

It is selected instead of a community tiny checkpoint or a hand-truncated model because it is Qwen's smallest standard
pretrained base model and retains the complete production architecture:

| Property | Value |
| --- | ---: |
| Parameters | 0.49B |
| Decoder layers | 24 |
| Hidden/intermediate size | 896 / 4864 |
| Query/KV heads | 14 / 2 (GQA) |
| Head dimension | 64 |
| Context limit | 32768 |
| Operators | RoPE, RMSNorm, SwiGLU, QKV bias, tied embeddings |
| Weight dtype | BF16 |

Qwen3-0.6B is newer but is not the smallest Qwen base checkpoint. The Instruct checkpoint is not the numerical baseline
because chat templates and post-training add unrelated variables to ANN-to-SNN validation.

## What is implemented

- A local PyTorch ANN whose state-dict keys match Hugging Face `Qwen2ForCausalLM` exactly.
- Mask-aware streaming min/max calibration and versioned JSON range artifacts.
- `SpikeTensor`, containing normalized first-spike time, range identity, and value dtype.
- Continuous time and finite precision `round(P*t)/P` modes with strict overflow telemetry.
- RMSNorm, RoPE, GQA causal attention, SwiGLU, residuals, and tied embedding/LM Head.
- Valid-set masked softmax: masks remain boolean sideband data and invalid probabilities are exactly zero.
- ANN and TTFS KV caches, including padding history validation and batch reorder.
- Hybrid LM Head by default, plus an opt-in fully encoded LM Head.
- Tiny random-model unit tests and an opt-in official-weight integration test.

This first backend is deliberately a **functional decode-compute-encode simulator**. It proves conversion semantics and
finite-time error behavior; it does not claim event-driven latency or energy savings. Embedding lookup and the default
hybrid LM Head remain dense and must be included in any future system cost report.

P0 commands execute the official BF16 weight values in FP32 by default. This follows the migration plan's validation
order and makes the local ANN mathematically identical to the Transformers eager reference. Pass `--dtype auto` to run
the checkpoint in its native BF16 dtype. BF16 fused/reduction kernels may accumulate small layerwise differences, so FP32
and BF16 results are separate baselines rather than interchangeable tolerances.

## Install

From the repository root:

```bash
python -m pip install -e ./TTFS_LM
```

The code requires Python 3.10+, PyTorch, Transformers, Safetensors, and huggingface-hub. The validated environment uses
PyTorch 2.11, Transformers 5.6, Safetensors 0.7, and huggingface-hub 1.13. The package
pins Transformers to the tested 5.x API because its checkpoint loader uses the `dtype` loading contract.

Direct Hugging Face access is used by default. In an environment that uses a mirror, export it before Python starts:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Weights are stored in the normal Hugging Face cache and are never copied into this repository.

## Verify the implementation

Run all fast tests:

```bash
python -m unittest discover -s TTFS_LM/tests -v
```

After the official checkpoint is cached, run its integration test:

```bash
TTFS_LM_RUN_OFFICIAL=1 python -m unittest \
  TTFS_LM.tests.test_official_checkpoint -v
```

The fast test suite proves:

- local ANN vs Hugging Face Qwen2 logits;
- no future-token information flow;
- padding and masked-softmax behavior;
- ANN and TTFS cache/no-cache equivalence;
- continuous TTFS numerical agreement;
- finite precision execution and strict range violations.

## Run the WikiText-2 MVP

The MVP pins both the official Qwen checkpoint and `Salesforce/wikitext` revisions. It uses at most 100,000 tokens from
the train split for range calibration, then evaluates the complete validation split with context length 128 and batch
size 4. Validation data never participates in calibration.

After the model and dataset are cached, run the complete offline workflow from the repository root:

```bash
HF_HUB_OFFLINE=1 ttfs-lm mvp \
  --config TTFS_LM/configs/mvp_wikitext2.yaml \
  --local-files-only
```

To populate an empty cache, omit `HF_HUB_OFFLINE=1` and `--local-files-only`. The command runs the official and local
ANN references, train-only calibration, strict continuous TTFS evaluation, a `P=4096` finite-precision smoke run, and
an eight-token cached generation check. It exits nonzero when a required acceptance check fails.

All numerical gates are explicit under `acceptance` in the YAML. The cache gate allows `2e-4` FP32 max-absolute error
between full-sequence and token-wise execution, and separately requires TTFS cache error to remain within `1e-6` of the
ANN cache error.

Results are written to `TTFS_LM/artifacts/mvp_wikitext2/`:

- `ranges.json`: calibrated tensor ranges and calibration provenance;
- `metrics.json`: PPL, NLL, logits agreement, KL, cache checks, generation, and range telemetry;
- `run_manifest.json`: resolved config, dataset fingerprints, environment, git revision, status, and acceptance checks;
- `run.log`: timestamped JSONL stage progress.

The continuous fidelity configuration uses a deliberately conservative train-range guard band. This preserves a strict
zero-clipping contract and is not a tuned finite-precision range policy. The `P=4096` result only verifies that finite
encoding completes with finite, fully reported outputs; it is not a quality claim.

## Run the cross-corpus sweep

The dataset sweep reuses one WikiText-2 train calibration artifact across complete WikiText-103 test, Penn Treebank
test, complete LAMBADA OpenAI test, and a pinned 262K-token C4 validation sample. Populate the caches once:

```bash
HF_ENDPOINT=https://hf-mirror.com ttfs-lm dataset-sweep \
  --config TTFS_LM/configs/dataset_sweep.yaml \
  --allow-download
```

Then reproduce without network access:

```bash
HF_HUB_OFFLINE=1 ttfs-lm dataset-sweep \
  --config TTFS_LM/configs/dataset_sweep.yaml \
  --local-files-only
```

Strict continuous range failures are recorded rather than hidden. A failed corpus is rerun once with explicit clipping
to quantify the affected elements and quality, after which `P=4096` is still evaluated. Results and interpretation are
in [`DATASET_SWEEP.md`](DATASET_SWEEP.md); raw artifacts are written under
`TTFS_LM/artifacts/dataset_sweep/`.

## Calibrate the official model

The built-in bilingual corpus is only a smoke calibration set. It exercises both prefill and token-by-token decode, but
must be replaced by the 5M-token P0/P1 calibration corpus before reporting perplexity or downstream quality.

```bash
ttfs-lm calibrate \
  --device cuda \
  --output TTFS_LM/artifacts/qwen2.5-0.5b-ranges.json
```

The artifact records the model ID, pinned revision, calibration mode, observed element counts, and one range per tensor
role and layer. Padding tokens are excluded. Softmax probability ranges are fixed to `[0, 1]`.

## Compare ANN and TTFS

Continuous-time correctness:

```bash
ttfs-lm compare \
  --device cuda \
  --ranges TTFS_LM/artifacts/qwen2.5-0.5b-ranges.json \
  --prompt "Time-to-first-spike language models"
```

Finite precision with 1024 intervals:

```bash
ttfs-lm compare \
  --device cuda \
  --ranges TTFS_LM/artifacts/qwen2.5-0.5b-ranges.json \
  --precision 1024 \
  --allow-clipping
```

`--allow-clipping` is explicit because out-of-range values must never be silently clipped. Without it, any overflow raises
`RangeViolationError` and records the responsible tensor role.

Use `--spike-lm-head` to encode the vocabulary logits too. The calibration artifact already contains `lm_head`; the flag
is off by default to match the staged Hybrid Head design.

## Generate

ANN generation:

```bash
ttfs-lm generate --device cuda --prompt "Neuromorphic language models"
```

TTFS generation:

```bash
ttfs-lm generate \
  --device cuda \
  --ranges TTFS_LM/artifacts/qwen2.5-0.5b-ranges.json \
  --prompt "Neuromorphic language models"
```

Generation is greedy and intentionally minimal. It exists to validate autoregressive state and TTFS KV-cache semantics,
not to reproduce chat behavior from an instruction-tuned checkpoint.

## Source layout

```text
TTFS_LM/
├── ttfs_lm/
│   ├── calibration.py    # RangeSpec, registry, mask-aware collection
│   ├── checkpoint.py     # pinned official weights and strict state-dict load
│   ├── cli.py            # calibrate, compare, generate, mvp
│   ├── config.py         # official architecture and TTFS configuration
│   ├── conversion.py     # checked ANN-to-TTFS entry point
│   ├── generation.py     # minimal cached greedy decoding
│   ├── modeling_ann.py   # local Qwen2 ANN reference
│   ├── modeling_ttfs.py  # functional TTFS conversion and cache
│   ├── workflow.py       # calibration and comparison helpers
│   ├── data.py           # pinned Hub/Git/HTTP corpora and deterministic packing
│   ├── evaluation.py     # causal-LM PPL and model-pair metrics
│   ├── mvp.py            # configured end-to-end MVP runner and artifacts
│   └── dataset_sweep.py  # shared-range cross-corpus evaluation
└── tests/
```

## Current boundary and next development gates

Before using results in a paper or energy claim, the next gates from the migration plan remain mandatory:

1. replace smoke calibration with a fixed, hashed, multi-domain token corpus;
2. evaluate WikiText-103 perplexity and teacher-forcing token agreement;
3. separate prefill/decode and length-bucket ranges;
4. implement an event-native backend and report its operation-weighted coverage;
5. include embedding, full LM Head, KV memory, routing, and encode/decode costs;
6. compare against BF16 and INT8 at equal quality and generation settings.

The current one-prompt `precision=1024` run is only a mechanism smoke test. It executes without NaN after enforcing
non-negative square/variance ranges, but a tiny min/max calibration set causes substantial downstream overflow and quality
loss. It does **not** satisfy the plan's finite-precision quality gate; do not cite it as a successful P=1024 result.
