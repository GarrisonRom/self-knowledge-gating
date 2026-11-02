"""Replaceable retrieval interfaces and a small in-memory baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from .schemas import Evidence, Question


class Retriever(Protocol):
    def search(self, question: Question, top_k: int = 5) -> Sequence[Evidence]:
        """Return evidence ordered by descending relevance."""


class NullRetriever:
    def search(self, question: Question, top_k: int = 5) -> Sequence[Evidence]:
        return ()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w.-]+", text.casefold()))


@dataclass
class InMemoryRetriever:
    """Simple lexical retriever for fixtures and local smoke runs.

    It deliberately has no embedding dependency. Production adapters can use
    BM25, a vector index, a Materials Project API or a hybrid retriever without
    changing the pipeline contract.
    """

    corpus: Sequence[Evidence]

    def search(self, question: Question, top_k: int = 5) -> Sequence[Evidence]:
        if top_k <= 0:
            return ()
        query_tokens = _tokens(question.text)
        ranked: list[Evidence] = []
        for evidence in self.corpus:
            if not bool(evidence.metadata.get("access_allowed", True)):
                continue
            document_tokens = _tokens(evidence.text)
            overlap = len(query_tokens & document_tokens)
            lexical_score = overlap / max(len(query_tokens), 1)
            score = max(evidence.relevance, min(1.0, lexical_score))
            if score <= 0.0 and not bool(evidence.metadata.get("always_retrieve", False)):
                continue
            ranked.append(replace(evidence, relevance=score))
        ranked.sort(
            key=lambda item: (item.relevance, item.reliability, item.freshness),
            reverse=True,
        )
        return tuple(ranked[:top_k])
