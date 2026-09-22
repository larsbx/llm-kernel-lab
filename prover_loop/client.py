"""Sampling from `max serve` over its OpenAI-compatible /v1/completions endpoint.

k samples are k concurrent requests with seeds seed, seed+1, ..., so the
server's continuous batching fills its batch and a run is reproducible to the
extent the server honours `seed`.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .prompt import FENCE


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: int
    seconds: float


Sampler = Callable[[str, int], list[Completion]]


def sampler(url: str, model: str, *, temperature: float, top_p: float, max_tokens: int, seed: int,
            timeout: float = 3600) -> Sampler:
    def one(prompt: str, s: int) -> Completion:
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                "top_p": top_p, "seed": s, "stop": [FENCE]}
        request = urllib.request.Request(f"{url}/v1/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.load(response)
        return Completion(reply["choices"][0]["text"], reply.get("usage", {}).get("completion_tokens", 0),
                          time.monotonic() - start)

    def sample(prompt: str, k: int) -> list[Completion]:
        with ThreadPoolExecutor(max_workers=k) as pool:
            return list(pool.map(lambda i: one(prompt, seed + i), range(k)))

    return sample
