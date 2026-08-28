from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from .calibration import RangeRegistry
from .checkpoint import load_official_ann, load_official_pair, load_tokenizer
from .config import DEFAULT_MODEL_ID, DEFAULT_REVISION, TTFSConfig
from .conversion import convert_ann_to_ttfs
from .generation import greedy_generate
from .workflow import DEFAULT_CALIBRATION_TEXTS, calibrate_ann, tensor_comparison, tokenize_texts


def _common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hf-endpoint", default=None)


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_id": args.model_id,
        "revision": args.revision,
        "calibration": "built-in bilingual smoke corpus",
        "max_length": args.max_length,
        "include_decode": not args.no_decode_calibration,
    }


def _calibration_batch(args, tokenizer, device):
    return tokenize_texts(tokenizer, DEFAULT_CALIBRATION_TEXTS, device, args.max_length)


def command_calibrate(args: argparse.Namespace) -> None:
    model = load_official_ann(args.model_id, args.revision, args.device, args.dtype, args.local_files_only)
    tokenizer = load_tokenizer(args.model_id, args.revision, args.local_files_only)
    batch = _calibration_batch(args, tokenizer, args.device)
    registry = calibrate_ann(
        model,
        [batch],
        margin=args.margin,
        include_decode=not args.no_decode_calibration,
        metadata=_metadata(args),
    )
    registry.save(args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "ranges": len(registry)}, indent=2))


def command_compare(args: argparse.Namespace) -> None:
    pair = load_official_pair(args.model_id, args.revision, args.device, args.dtype, args.local_files_only)
    tokenizer = load_tokenizer(args.model_id, args.revision, args.local_files_only)
    sample = tokenizer(args.prompt, return_tensors="pt")
    sample = {name: value.to(args.device) for name, value in sample.items() if name in {"input_ids", "attention_mask"}}
    with torch.no_grad():
        reference_logits = pair.reference(**sample, use_cache=False).logits
        ann_logits = pair.ann(**sample, use_cache=False).logits
    ann_metrics = tensor_comparison(reference_logits, ann_logits)

    del pair.reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.ranges:
        registry = RangeRegistry.load(args.ranges)
    else:
        calibration_batch = _calibration_batch(args, tokenizer, args.device)
        registry = calibrate_ann(
            pair.ann,
            [calibration_batch, sample],
            margin=args.margin,
            include_decode=not args.no_decode_calibration,
            metadata=_metadata(args),
        )
    ttfs = convert_ann_to_ttfs(
        pair.ann,
        registry,
        TTFSConfig(
            precision=args.precision,
            strict_ranges=not args.allow_clipping,
            spike_lm_head=args.spike_lm_head,
        ),
    ).eval()
    with torch.no_grad():
        ttfs_logits = ttfs(**sample, use_cache=False).logits
    comparison_mask = sample.get("attention_mask")
    if comparison_mask is not None:
        comparison_mask = comparison_mask[:, :, None]
    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "official_vs_local_ann": ann_metrics,
        "ann_vs_ttfs": tensor_comparison(ann_logits, ttfs_logits, comparison_mask),
        "precision": args.precision,
        "range_count": len(registry),
        "violating_ranges": {
            name: {"below": stats.below, "above": stats.above}
            for name, stats in ttfs.codec.telemetry.items()
            if stats.below or stats.above
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def command_generate(args: argparse.Namespace) -> None:
    ann = load_official_ann(args.model_id, args.revision, args.device, args.dtype, args.local_files_only)
    tokenizer = load_tokenizer(args.model_id, args.revision, args.local_files_only)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    encoded = {name: value.to(args.device) for name, value in encoded.items() if name in {"input_ids", "attention_mask"}}
    model = ann
    if args.ranges:
        registry = RangeRegistry.load(args.ranges)
        model = convert_ann_to_ttfs(
            ann,
            registry,
            TTFSConfig(args.precision, not args.allow_clipping, args.spike_lm_head),
        ).eval()
    generated = greedy_generate(
        model,
        encoded["input_ids"],
        encoded.get("attention_mask"),
        args.max_new_tokens,
        tokenizer.eos_token_id,
    )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Qwen2.5 ANN-to-TTFS reference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate", help="collect and save activation ranges")
    _common_model_arguments(calibrate)
    calibrate.add_argument("--output", default="artifacts/qwen2.5-0.5b-ranges.json")
    calibrate.add_argument("--max-length", type=int, default=128)
    calibrate.add_argument("--margin", type=float, default=0.05)
    calibrate.add_argument("--no-decode-calibration", action="store_true")
    calibrate.set_defaults(handler=command_calibrate)

    compare = subparsers.add_parser("compare", help="compare official, local ANN, and TTFS logits")
    _common_model_arguments(compare)
    compare.add_argument("--prompt", default="Time-to-first-spike language models")
    compare.add_argument("--ranges", default=None)
    compare.add_argument("--max-length", type=int, default=128)
    compare.add_argument("--margin", type=float, default=0.05)
    compare.add_argument("--no-decode-calibration", action="store_true")
    compare.add_argument("--precision", type=int, default=None)
    compare.add_argument("--allow-clipping", action="store_true")
    compare.add_argument("--spike-lm-head", action="store_true")
    compare.set_defaults(handler=command_compare)

    generate = subparsers.add_parser("generate", help="run greedy ANN or calibrated TTFS generation")
    _common_model_arguments(generate)
    generate.add_argument("--prompt", default="The future of neuromorphic language models is")
    generate.add_argument("--ranges", default=None)
    generate.add_argument("--precision", type=int, default=None)
    generate.add_argument("--allow-clipping", action="store_true")
    generate.add_argument("--spike-lm-head", action="store_true")
    generate.add_argument("--max-new-tokens", type=int, default=20)
    generate.set_defaults(handler=command_generate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    args.handler(args)


if __name__ == "__main__":
    main()
