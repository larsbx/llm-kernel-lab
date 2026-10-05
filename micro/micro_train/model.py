"""One small transformer, two heads: next-byte prediction (causal) or classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MicroConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    context: int = 256
    dropout: float = 0.0
    vocab_size: int = 256

    def to_dict(self) -> dict:
        return asdict(self)


class MicroTransformer(nn.Module):
    """`n_classes=None` is a causal LM returning per-position logits over bytes;
    otherwise a bidirectional encoder, mean-pooled over real tokens, returning class logits."""

    def __init__(self, config: MicroConfig, n_classes: int | None):
        super().__init__()
        self.config, self.n_classes = config, n_classes
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position = nn.Embedding(config.context, config.d_model)
        layer = nn.TransformerEncoderLayer(config.d_model, config.n_heads, 4 * config.d_model, config.dropout,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, n_classes or config.vocab_size)

    def forward(self, ids: torch.Tensor, padding: torch.Tensor | None = None) -> torch.Tensor:
        n = ids.shape[1]
        h = self.embed(ids) + self.position(torch.arange(n, device=ids.device))
        if self.n_classes is None:
            causal = nn.Transformer.generate_square_subsequent_mask(n, device=ids.device)
            return self.head(self.norm(self.blocks(h, mask=causal, is_causal=True)))
        h = self.norm(self.blocks(h, src_key_padding_mask=padding))
        keep = (~padding).unsqueeze(-1).float() if padding is not None else torch.ones_like(h[..., :1])
        return self.head((h * keep).sum(1) / keep.sum(1).clamp(min=1))
