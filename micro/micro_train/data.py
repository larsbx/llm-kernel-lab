"""Text in, integer tensors out. Bytes are the vocabulary, so nothing is downloaded
and every string (any script, any symbol) has exactly one encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ByteTokenizer:
    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def split_text(text: str, held_out: float) -> tuple[str, str]:
    """The last `held_out` fraction is held out: deterministic, and no window straddles both."""
    cut = int(len(text) * (1 - held_out))
    return text[:cut], text[cut:]


def load_labelled(path: Path) -> tuple[list[str], list[int], list[str]]:
    """JSONL rows {"text", "label"}; labels are indexed in sorted order."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = sorted({str(r["label"]) for r in rows})
    index = {n: i for i, n in enumerate(names)}
    return [r["text"] for r in rows], [index[str(r["label"])] for r in rows], names
