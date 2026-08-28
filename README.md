# Code of TTFSFormer (ICML 25)

This is the official implementation of our work **TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer**.

## Setup

The code are tested under `Python 3.12`, with a minimum requirement of `3.10`.

Run
```bash
pip install -e .
```
to install all required packages.

## File Structure

```
/
├── test
│   ├── eval_origin.py          # Run ANN and get parameters for SNN.
│   └── eval_ttfs.py            # Evaluate on SNN.
└── ttfsformer/
    ├── model/
    │   ├── layers.py           # Original Layers
    │   ├── spiking_layers.py   # Spiking Layers
    │   ├── vit.py              # Original ViT
    │   ├── spiking_vit.py      # Spiking ViT
    │   ├── eva.py              # Original EVA
    │   └── spiking_eva.py      # Spiking EVA
    ├── spike/
    │   ├── kernel/             # SNN Kernels
    │   │   └── ...
    │   ├── conversion.py       # Convert ANN to SNN
    │   ├── module.py           # Spiking version of basic modules
    │   └── ...
    └── util/
        └── ...
```

## Testing

For an end-to-end reproduction record, including the exact environment, dataset checks,
commands, paper-to-local metric comparison, and known discrepancies, see
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

For the staged design to migrate TTFSFormer to encoder and causal language tasks,
including calibration, KV-cache semantics, evaluation gates, and system-level energy
accounting, see [`docs/LANGUAGE_MIGRATION_PLAN.md`](docs/LANGUAGE_MIGRATION_PLAN.md).

The first runnable language baseline, using the pinned official Qwen2.5-0.5B weights,
is in [`TTFS_LM/`](TTFS_LM/README.md).

### Weights

Please download the weight from hugging face before you run the code.

| Model    | Checkpoint                                                                                                                        |
| -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| ViT-S/16 | [timm/vit_small_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k)         |
| ViT-B/16 | [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)         |
| ViT-L/16 | [timm/vit_large_patch16_224.augreg_in21k_ft_in1k](https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k)         |
| EVA02-S  | [timm/eva02_small_patch14_336.mim_in22k_ft_in1k](https://huggingface.co/timm/eva02_small_patch14_336.mim_in22k_ft_in1k)           |
| EVA02-L  | [timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k](https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k) |
| EVA-G    | [timm/eva_giant_patch14_224.clip_ft_in1k](https://huggingface.co/timm/eva_giant_patch14_224.clip_ft_in1k)         |

### Running

All testing code are in the folder `./test`. `--help` or `-h` option lists the arguments for the program. Run
```bash
python test/eval_origin.py
```
to get the parameters for SNN, and run
```bash
python test/eval_ttfs.py
```
to evaluate it on SNN.
