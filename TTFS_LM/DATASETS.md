# Dataset reference

This document defines the exact corpora used by the Qwen2.5 TTFS MVP and cross-corpus sweep. The executable source of
truth is [`configs/dataset_sweep.yaml`](configs/dataset_sweep.yaml); this file explains why each corpus is present, how
it is loaded, and what its metrics do and do not mean.

## Experiment roles

The sweep deliberately separates calibration from evaluation:

- **Calibration only:** WikiText-2 raw train, capped at 100,000 Qwen tokens.
- **Evaluation only:** WikiText-103 test, Penn Treebank test, LAMBADA OpenAI test, and a fixed C4 validation sample.

One `ranges.json` is produced from the calibration corpus. Evaluation activations never update that registry. This makes
the sweep a test of range generalization rather than per-dataset range tuning.

## Pinned sources

| Role | Corpus | Source and revision | Split / field | Loader | License or usage note |
| --- | --- | --- | --- | --- | --- |
| Calibration | WikiText-2 raw | `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1` | `train` / `text` | Hugging Face | CC BY-SA 3.0 and GFDL |
| Evaluation | WikiText-103 raw | `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-103-raw-v1` | `test` / `text` | Hugging Face | CC BY-SA 3.0 and GFDL |
| Evaluation | Penn Treebank | `wojzaremba/lstm@76870253cfca069477f06b7056af87f98490b6eb`, `data/ptb.test.txt` | test file / line | pinned Git blob | Original PTB terms are the LDC User Agreement; verify redistribution rights independently |
| Evaluation | LAMBADA OpenAI | `EleutherAI/lambada_openai@900124bf3b8235c6daf21033af9948b3f07346c4`, `default` | `test` / `text` | Hugging Face | Dataset card declares MIT |
| Evaluation | C4 English | `allenai/c4@1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, `en/c4-validation.00000-of-00008.json.gz` | `validation` / `text` | pinned HTTP JSON gzip | ODC-BY and Common Crawl terms |

Source pages: [WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
[Penn Treebank preprocessing](https://github.com/wojzaremba/lstm/tree/76870253cfca069477f06b7056af87f98490b6eb/data),
[LAMBADA OpenAI](https://huggingface.co/datasets/EleutherAI/lambada_openai), and
[C4](https://huggingface.co/datasets/allenai/c4).

The PTB file is convenient for reproducing the historical language-model preprocessing, but its presence in a public
Git repository should not be interpreted as a new open-data license. The sweep caches it locally and does not vendor it
into this repository.

## Why these corpora

### WikiText-103

WikiText-103 retains case, punctuation, numbers, and article-style English. It is the closest larger-domain extension of
the WikiText-2 calibration source and therefore isolates whether range coverage transfers beyond the MVP split.

The sweep evaluates the complete raw test split. It does not calibrate on WikiText-103 train.

### Penn Treebank

PTB provides a small, classic newswire distribution with aggressive historical preprocessing, including lower-casing
and unknown-token placeholders. Its role is distribution-shift diagnosis, not a modern general-language quality claim.

The reported value is Qwen-token causal PPL over `ptb.test.txt`. It is not comparable to traditional word-level PTB PPL
because the vocabulary, token count, boundary handling, and denominator differ.

### LAMBADA OpenAI

LAMBADA contains narrative passages selected for long-range final-word prediction. It provides longer coherent examples
than line-oriented WikiText or PTB and probes whether rare activations appear in narrative contexts.

This sweep computes causal PPL over every next-token target in the complete `default/test` split. It does not report the
standard LAMBADA last-word accuracy, so it must not be compared directly with leaderboard accuracy.

### C4 English sample

C4 adds broad, noisy web text and is the strongest OOD corpus in this first sweep. The complete English validation split
contains hundreds of thousands of documents, so the experiment uses one revision-pinned validation shard and stops at
exactly 262,420 Qwen tokens. This matches the order of magnitude of the original MVP evaluation while keeping download
and GPU cost bounded.

The raw gzip must match SHA256
`1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f`. This is a deterministic sample, not a full C4
validation result.

## Materialized corpus identities

These values were produced on 2026-08-29 with the pinned Qwen2.5 tokenizer revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`.

| Corpus | Source rows scanned | Non-empty rows | Qwen tokens | Prediction targets | Token SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| WikiText-2 train calibration | 1,456 | 944 | 100,000 | 99,999 | `9abef75c8d3d767f11485cce96acfd83c6820cdbb579447b600ad961ebdc071e` |
| WikiText-103 test | 4,358 | 2,891 | 299,241 | 299,240 | `950454728a4bb1acfe692fddbcb0f006951ee05ba44ecde2ce25d55e62415dfa` |
| Penn Treebank test | 3,761 | 3,761 | 99,256 | 99,255 | `ce86bdbb66b8e7cb761213c9fc46863467832af854814d10a222a5ae07cb2598` |
| LAMBADA OpenAI test | 5,153 | 5,153 | 396,547 | 396,546 | `76fa7f1389f57263fb76e5842af57163fb9bdbcb0e7becedc2161c21db088729` |
| C4 validation sample | 554 | 554 | 262,420 | 262,419 | `09e4977368d0a2822fa7fca04921b9cfced3f5ecbdcc54dec3a08f4637f2ecee` |

The calibration and C4 corpora are intentionally truncated at their token caps. The other three evaluation corpora use
their complete selected test source.

## Text and packing contract

All sources enter the same deterministic `pack_texts` path:

1. Convert the source value to text, trim surrounding whitespace, and discard empty rows.
2. Tokenize one source row at a time with `add_special_tokens=False`.
3. Append the Qwen EOS token to every non-empty source row.
4. Concatenate the resulting row sequences in source order.
5. Pack context-128 windows with stride 127, so adjacent windows share one token and each transition is scored once.
6. Right-pad only the final short window; padding uses the EOS token ID but is excluded by the attention mask.
7. For a capped corpus, stop exactly at `max_tokens` and force its final retained token to EOS.

This contract preserves source-row boundaries through EOS markers. A source row is not always a complete semantic
document: WikiText rows can be headings or lines, while PTB rows are preprocessed sentences. Reports should therefore say
"source row + EOS," not claim that all original document structures are reconstructed.

PPL, KL, and top-1 agreement use only valid next-token prediction positions. The first token of each packed window and
masked padding positions are not added as extra targets.

## Cache and reproducibility

- Hugging Face datasets use the normal datasets cache and are pinned by repository revision.
- PTB is cloned without checkout under the sweep's ignored `sources/` cache, then read with `git show revision:path`.
- C4 is copied into a content-addressed local cache and verified before JSON parsing, allowing true offline reuse without
  an HTTP HEAD request.
- Packed source-text and token hashes, dataset fingerprints, configuration SHA256, environment, and run command are
  recorded in `TTFS_LM/artifacts/dataset_sweep/sweep_manifest.json`.
- Raw corpora and run artifacts are ignored by Git. Only configuration, code, and summarized results are committed.

To introduce or update a corpus, create a new named experiment configuration, pin an immutable revision and raw-file
hash where applicable, run once with downloads enabled, then verify the materialized token hash before using offline
results. Do not silently move an existing experiment name to a new dataset revision.
