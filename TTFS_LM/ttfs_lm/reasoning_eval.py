from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .calibration import RangeRegistry
from .checkpoint import load_official_ann, load_tokenizer
from .config import TTFSConfig
from .conversion import convert_ann_to_ttfs


@dataclass(frozen=True)
class GPQAItem:
    question: str
    correct: str
    incorrect: tuple[str, str, str]


def _field(row: dict[str, Any], *names: str) -> str:
    for name in names:
        if name in row:
            value = row[name]
            if value is None:
                break
            return str(value)
    raise KeyError(f"GPQA row is missing fields {names!r}; available={sorted(row)}")


def _load_gpqa(
    local_files_only: bool,
    max_questions: int | None,
) -> list[GPQAItem]:
    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {
        "split": "train",
        "verification_mode": "no_checks",
    }
    if local_files_only:
        load_kwargs["download_mode"] = "reuse_dataset_if_exists"
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", **load_kwargs)
    items = [
        GPQAItem(
            question=_field(row, "Question", "question"),
            correct=_field(row, "Correct Answer", "correct_answer", "answer"),
            incorrect=(
                _field(row, "Incorrect Answer 1", "incorrect_answer_1"),
                _field(row, "Incorrect Answer 2", "incorrect_answer_2"),
                _field(row, "Incorrect Answer 3", "incorrect_answer_3"),
            ),
        )
        for row in dataset
    ]
    return items[:max_questions] if max_questions is not None else items


def _prompt(question: str, choices: list[str]) -> str:
    labels = "ABCD"
    rendered = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(choices))
    return f"Question:\n{question}\n\nChoices:\n{rendered}\n\nAnswer:"


def _choice_scores(
    model,
    tokenizer,
    prompt: str,
    choices: list[str],
    device: str | torch.device,
) -> list[float]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    sequences: list[list[int]] = []
    for choice in choices:
        continuation = tokenizer(" " + choice, add_special_tokens=False)["input_ids"]
        if continuation and isinstance(continuation[0], list):
            continuation = continuation[0]
        sequences.append([int(token) for token in prompt_ids + continuation])

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    lengths: list[int] = []
    for index, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[index, :length] = 1
        lengths.append(length)

    logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
    scores: list[float] = []
    prompt_length = len(prompt_ids)
    for index, length in enumerate(lengths):
        choice_length = length - prompt_length
        log_probs = F.log_softmax(logits[index, : length - 1], dim=-1)
        targets = input_ids[index, prompt_length:length]
        scores.append(float(log_probs[prompt_length - 1 : prompt_length - 1 + choice_length]
                            .gather(-1, targets[:, None])
                            .sum()
                            .item()))
    return scores


@torch.no_grad()
def evaluate_gpqa_model(
    model,
    tokenizer,
    items: list[GPQAItem],
    device: str | torch.device,
    seed: int,
) -> dict[str, Any]:
    correct = 0
    for index, item in enumerate(items):
        choices = [item.correct, *item.incorrect]
        random.Random(seed + index).shuffle(choices)
        prompt = _prompt(item.question, choices)
        prediction = int(torch.tensor(_choice_scores(model, tokenizer, prompt, choices, device)).argmax())
        correct_choice = choices.index(item.correct)
        correct += int(prediction == correct_choice)
    return {
        "accuracy": correct / len(items) if items else float("nan"),
        "correct": correct,
        "questions": len(items),
        "seed": seed,
        "scoring": "continuation log-likelihood sum",
    }


def run_gpqa(
    model_id: str,
    revision: str,
    device: str,
    dtype: str,
    ranges_path: str | None,
    precision: int,
    allow_clipping: bool,
    local_files_only: bool,
    seed: int,
    max_questions: int | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(model_id, revision, local_files_only)
    ann = load_official_ann(model_id, revision, device, dtype, local_files_only)
    items = _load_gpqa(local_files_only, max_questions)
    models: dict[str, Any] = {"ann": ann}

    if ranges_path is not None:
        registry = RangeRegistry.load(ranges_path)
        models["continuous"] = convert_ann_to_ttfs(
            ann,
            registry,
            TTFSConfig(precision=None, strict_ranges=not allow_clipping),
        ).eval()
        models[f"p{precision}"] = convert_ann_to_ttfs(
            ann,
            registry,
            TTFSConfig(precision=precision, strict_ranges=not allow_clipping),
        ).eval()

    result: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "dataset": "Idavidrein/gpqa:gpqa_diamond",
        "questions": len(items),
        "seed": seed,
        "modes": {},
    }
    for name, model in models.items():
        try:
            metrics = evaluate_gpqa_model(model, tokenizer, items, device, seed)
            if hasattr(model, "codec"):
                metrics["telemetry"] = {
                    role: {
                        "below": stats.below,
                        "above": stats.above,
                        "elements": stats.elements,
                    }
                    for role, stats in model.codec.telemetry.items()
                    if stats.below or stats.above
                }
            result["modes"][name] = {"status": "completed", **metrics}
        except Exception as exc:
            result["modes"][name] = {
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
