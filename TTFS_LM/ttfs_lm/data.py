from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import torch


DEFAULT_DATASET_ID = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


@dataclass
class PackedCorpus:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("PackedCorpus tensors must have matching [sequences, context] shapes")
        if self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long")
        if self.input_ids.shape[0] == 0:
            raise ValueError("PackedCorpus must contain at least one sequence")

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    @property
    def target_tokens(self) -> int:
        prediction_mask = self.attention_mask[:, :-1].bool() & self.attention_mask[:, 1:].bool()
        return int(prediction_mask.sum().item())

    def batches(self, batch_size: int, device: str | torch.device) -> Iterator[dict[str, torch.Tensor]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            end = min(start + batch_size, len(self))
            yield {
                "input_ids": self.input_ids[start:end].to(device),
                "attention_mask": self.attention_mask[start:end].to(device),
            }


def _tokenize_document(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Expected one document per tokenizer call")
        input_ids = input_ids[0]
    return [int(token) for token in input_ids]


def pack_texts(
    texts: Iterable[str],
    tokenizer,
    context_length: int,
    max_tokens: int | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> PackedCorpus:
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if max_tokens is not None and max_tokens < 2:
        raise ValueError("max_tokens must be at least 2 or None")
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    tokens: list[int] = []
    source_hash = hashlib.sha256()
    nonempty_rows = 0
    source_characters = 0
    truncated = False
    for row_index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        source_hash.update(f"{row_index}\0{text}\n".encode("utf-8"))
        document_tokens = _tokenize_document(tokenizer, text)
        if not document_tokens:
            continue
        document_tokens.append(int(tokenizer.eos_token_id))
        nonempty_rows += 1
        source_characters += len(text)
        if max_tokens is not None and len(tokens) + len(document_tokens) > max_tokens:
            remaining = max_tokens - len(tokens)
            if remaining > 0:
                partial = document_tokens[:remaining]
                partial[-1] = int(tokenizer.eos_token_id)
                tokens.extend(partial)
            truncated = True
            break
        tokens.extend(document_tokens)

    if len(tokens) < 2:
        raise ValueError("The selected text produced fewer than two tokens")

    token_tensor = torch.tensor(tokens, dtype=torch.long)
    token_hash = hashlib.sha256(token_tensor.numpy().tobytes()).hexdigest()
    stride = context_length - 1
    packed_ids: list[torch.Tensor] = []
    packed_masks: list[torch.Tensor] = []
    for start in range(0, len(tokens) - 1, stride):
        chunk = token_tensor[start : start + context_length]
        if chunk.numel() < 2:
            break
        valid = chunk.numel()
        if valid < context_length:
            padding = torch.full((context_length - valid,), int(pad_token_id), dtype=torch.long)
            chunk = torch.cat((chunk, padding))
        mask = torch.zeros(context_length, dtype=torch.long)
        mask[:valid] = 1
        packed_ids.append(chunk)
        packed_masks.append(mask)

    input_ids = torch.stack(packed_ids)
    attention_mask = torch.stack(packed_masks)
    metadata = dict(source_metadata or {})
    metadata.update(
        {
            "context_length": context_length,
            "stride": stride,
            "max_tokens": max_tokens,
            "token_count": len(tokens),
            "target_token_count": int(
                (attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()).sum().item()
            ),
            "sequence_count": input_ids.shape[0],
            "nonempty_rows": nonempty_rows,
            "source_characters": source_characters,
            "truncated": truncated,
            "token_sha256": token_hash,
            "source_text_sha256": source_hash.hexdigest(),
            "tokenizer_name": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
            "tokenizer_vocab_size": len(tokenizer),
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id": int(pad_token_id),
            "packing_rule": "non-empty rows + EOS, overlapping windows with stride=context_length-1",
        }
    )
    return PackedCorpus(input_ids, attention_mask, metadata)


def load_wikitext2_corpora(
    tokenizer,
    calibration_split: str = "train",
    evaluation_split: str = "validation",
    calibration_max_tokens: int = 100_000,
    evaluation_max_tokens: int | None = None,
    context_length: int = 128,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    revision: str = DEFAULT_DATASET_REVISION,
    local_files_only: bool = False,
) -> tuple[PackedCorpus, PackedCorpus]:
    if calibration_split == evaluation_split:
        raise ValueError("Calibration and evaluation splits must be different")
    from datasets import DownloadConfig, load_dataset

    download_config = DownloadConfig(local_files_only=local_files_only)
    calibration_data, evaluation_data = load_dataset(
        dataset_id,
        dataset_config,
        split=[calibration_split, evaluation_split],
        revision=revision,
        download_config=download_config,
    )

    def metadata(split: str, dataset) -> dict[str, Any]:
        return {
            "dataset_id": dataset_id,
            "dataset_config": dataset_config,
            "dataset_revision": revision,
            "split": split,
            "dataset_fingerprint": dataset._fingerprint,
            "source_rows": len(dataset),
        }

    calibration = pack_texts(
        calibration_data["text"],
        tokenizer,
        context_length,
        calibration_max_tokens,
        metadata(calibration_split, calibration_data),
    )
    evaluation = pack_texts(
        evaluation_data["text"],
        tokenizer,
        context_length,
        evaluation_max_tokens,
        metadata(evaluation_split, evaluation_data),
    )
    return calibration, evaluation
