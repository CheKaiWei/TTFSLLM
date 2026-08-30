from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


DEFAULT_DATASET_ID = "Salesforce/wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusSource:
    name: str
    source: str
    id: str
    revision: str
    split: str
    text_field: str
    max_tokens: int | None
    config: str | None = None
    streaming: bool = False
    data_files: dict[str, str | list[str]] | None = None
    path: str | None = None
    file_sha256: str | None = None

    def validate(self) -> None:
        if not self.name or not self.id or not self.revision or not self.split or not self.text_field:
            raise ValueError("Corpus source identity, revision, split, and text_field must be non-empty")
        if self.source not in {"huggingface", "git_text", "http_json"}:
            raise ValueError(f"Unsupported corpus source {self.source!r}")
        if self.max_tokens is not None and self.max_tokens < 2:
            raise ValueError("Corpus max_tokens must be at least two or null")
        if self.source == "git_text" and not self.path:
            raise ValueError("git_text corpus sources require path")
        if self.source == "huggingface" and self.path is not None:
            raise ValueError("Hugging Face corpus sources do not use path")
        if self.source == "http_json" and self.path is not None:
            raise ValueError("HTTP JSON corpus sources do not use path")
        if self.file_sha256 is not None and len(self.file_sha256) != 64:
            raise ValueError("file_sha256 must be a 64-character digest or null")


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
    source_rows_scanned = 0
    source_characters = 0
    truncated = False
    for row_index, raw_text in enumerate(texts):
        source_rows_scanned = row_index + 1
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
            "source_rows_scanned": source_rows_scanned,
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


def _git_texts(
    source: CorpusSource,
    cache_dir: str | Path,
    local_files_only: bool,
) -> tuple[list[str], dict[str, Any]]:
    cache_root = Path(cache_dir)
    repository_key = hashlib.sha256(source.id.encode("utf-8")).hexdigest()[:16]
    repository = cache_root / repository_key
    if not repository.exists():
        if local_files_only:
            raise FileNotFoundError(f"Git corpus cache is missing: {repository}")
        cache_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", source.id, str(repository)],
            check=True,
            capture_output=True,
            text=True,
        )

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    try:
        resolved_revision = git("rev-parse", f"{source.revision}^{{commit}}").stdout.strip()
    except subprocess.CalledProcessError:
        if local_files_only:
            raise FileNotFoundError(
                f"Git corpus revision {source.revision} is not cached in {repository}"
            ) from None
        git("fetch", "--depth=1", "origin", source.revision)
        resolved_revision = git("rev-parse", "FETCH_HEAD^{commit}").stdout.strip()
    if resolved_revision != source.revision:
        raise ValueError(
            f"Git corpus resolved to {resolved_revision}, expected pinned revision {source.revision}"
        )

    payload = git("show", f"{resolved_revision}:{source.path}").stdout
    payload_bytes = payload.encode("utf-8")
    texts = payload.splitlines()
    return texts, {
        "dataset_id": source.id,
        "dataset_config": source.config,
        "dataset_revision": resolved_revision,
        "split": source.split,
        "dataset_fingerprint": hashlib.sha256(payload_bytes).hexdigest(),
        "source_rows": len(texts),
        "source_type": source.source,
        "source_path": source.path,
        "source_file_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }


def _http_corpus_file(
    source: CorpusSource,
    cache_dir: str | Path,
    local_files_only: bool,
) -> tuple[Path, str]:
    cache_root = Path(cache_dir) / "http"
    cache_key = source.file_sha256 or hashlib.sha256(source.id.encode("utf-8")).hexdigest()
    suffix = "".join(Path(source.id).suffixes) or ".data"
    cached_file = cache_root / f"{cache_key}{suffix}"
    if not cached_file.exists():
        if local_files_only:
            raise FileNotFoundError(f"HTTP corpus cache is missing: {cached_file}")
        from datasets import DownloadConfig, DownloadManager

        download_config = DownloadConfig(cache_dir=str(cache_root / "datasets-downloads"))
        downloaded_path = Path(
            DownloadManager(download_config=download_config).download(source.id)
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        temporary = cached_file.with_suffix(cached_file.suffix + ".tmp")
        shutil.copyfile(downloaded_path, temporary)
        temporary.replace(cached_file)

    digest = _sha256_file(cached_file)
    if source.file_sha256 is not None and digest != source.file_sha256:
        raise ValueError(
            f"HTTP corpus SHA256 mismatch for {source.name}: {digest} != {source.file_sha256}"
        )
    return cached_file, digest


def load_corpus(
    source: CorpusSource,
    tokenizer,
    context_length: int,
    local_files_only: bool = False,
    source_cache_dir: str | Path = "TTFS_LM/artifacts/dataset_sources",
) -> PackedCorpus:
    source.validate()
    if source.source == "git_text":
        texts, metadata = _git_texts(source, source_cache_dir, local_files_only)
    elif source.source == "http_json":
        from datasets import DownloadConfig, load_dataset

        downloaded_path, digest = _http_corpus_file(
            source, source_cache_dir, local_files_only
        )
        download_config = DownloadConfig(local_files_only=True)
        dataset = load_dataset(
            "json",
            data_files={source.split: str(downloaded_path)},
            split=source.split,
            download_config=download_config,
        )
        if source.text_field not in dataset.column_names:
            raise ValueError(
                f"Corpus {source.name!r} does not contain text field {source.text_field!r}: "
                f"{dataset.column_names}"
            )
        metadata = {
            "dataset_id": source.id,
            "dataset_config": source.config,
            "dataset_revision": source.revision,
            "split": source.split,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "source_rows": len(dataset),
            "source_type": source.source,
            "source_file_sha256": digest,
        }
        texts = (row[source.text_field] for row in dataset)
    else:
        from datasets import DownloadConfig, load_dataset

        download_config = DownloadConfig(local_files_only=local_files_only)
        dataset = load_dataset(
            source.id,
            source.config,
            split=source.split,
            revision=source.revision,
            streaming=source.streaming,
            data_files=source.data_files,
            download_config=download_config,
        )
        if source.text_field not in dataset.column_names:
            raise ValueError(
                f"Corpus {source.name!r} does not contain text field {source.text_field!r}: "
                f"{dataset.column_names}"
            )
        try:
            source_rows = len(dataset)
        except TypeError:
            source_rows = None
        metadata = {
            "dataset_id": source.id,
            "dataset_config": source.config,
            "dataset_revision": source.revision,
            "split": source.split,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "source_rows": source_rows,
            "source_type": source.source,
            "streaming": source.streaming,
            "data_files": source.data_files,
        }
        texts = (row[source.text_field] for row in dataset)

    metadata["corpus_name"] = source.name
    metadata["text_field"] = source.text_field
    return pack_texts(
        texts,
        tokenizer,
        context_length,
        source.max_tokens,
        metadata,
    )


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
    calibration = load_corpus(
        CorpusSource(
            name=f"{dataset_config}_{calibration_split}",
            source="huggingface",
            id=dataset_id,
            config=dataset_config,
            revision=revision,
            split=calibration_split,
            text_field="text",
            max_tokens=calibration_max_tokens,
        ),
        tokenizer,
        context_length,
        local_files_only,
    )
    evaluation = load_corpus(
        CorpusSource(
            name=f"{dataset_config}_{evaluation_split}",
            source="huggingface",
            id=dataset_id,
            config=dataset_config,
            revision=revision,
            split=evaluation_split,
            text_field="text",
            max_tokens=evaluation_max_tokens,
        ),
        tokenizer,
        context_length,
        local_files_only,
    )
    return calibration, evaluation
