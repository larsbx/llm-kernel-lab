"""Benchmark problems: a Lean statement ending in `:= by`, and its informal text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

THEOREM = re.compile(r"^theorem\s+(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class Problem:
    name: str
    informal: str
    formal_statement: str

    @property
    def theorem(self) -> str:
        """The declared theorem's name, read from the statement itself."""
        found = THEOREM.findall(self.formal_statement)
        if len(found) != 1:
            raise ValueError(f"{self.name}: expected one theorem in the statement, found {len(found)}")
        return found[0]


def statement_key(formal_statement: str) -> str:
    """What a statement asserts, independent of its name, header and layout:
    the text after `theorem <name>` up to `:= by`, whitespace collapsed. Two
    problems with equal keys are the same problem for contamination purposes."""
    body = formal_statement[formal_statement.index("theorem"):]
    body = re.sub(r"^theorem\s+\S+", "", body)
    body = re.split(r":=\s*by\b", body, maxsplit=1)[0]
    return " ".join(body.split())


def load(path: Path) -> list[Problem]:
    return [Problem(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
