"""BGE-M3 dense retrieval with BGE reranking.

The optional imports stay inside the constructor so the core pipeline remains
usable without the retrieval extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .schemas import Evidence, Question


@dataclass
class BGERetriever:
    """Local, offline BGE-M3 plus bge-reranker-v2-m3 retriever."""

    encoder_path: str
    reranker_path: str
    corpus: Sequence[Evidence]
    candidate_top_k: int = 100
    final_top_k: int = 20
    batch_size: int = 16
    max_length: int = 8192
    use_fp16: bool = True
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.candidate_top_k <= 0 or self.final_top_k <= 0:
            raise ValueError("candidate_top_k and final_top_k must be positive")
        if self.final_top_k > self.candidate_top_k:
            raise ValueError("final_top_k cannot exceed candidate_top_k")
        if self.batch_size <= 0 or self.max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        try:
            from FlagEmbedding import BGEM3FlagModel, FlagReranker
        except ImportError as exc:
            raise RuntimeError(
                "BGERetriever requires the retrieval extra: pip install -e '.[retrieval]'"
            ) from exc

        self._encoder = BGEM3FlagModel(self.encoder_path, use_fp16=self.use_fp16)
        self._reranker = FlagReranker(self.reranker_path, use_fp16=self.use_fp16)
        self._doc_embeddings: np.ndarray | None = None

    def search(self, question: Question, top_k: int = 5) -> Sequence[Evidence]:
        if not self.corpus:
            return ()
        requested_k = min(top_k, self.final_top_k, len(self.corpus))
        if requested_k <= 0:
            return ()
        self._ensure_index()
        query_embedding = self._encode([question.text])[0]
        dense_scores = self._doc_embeddings @ query_embedding  # type: ignore[operator]
        candidate_k = min(self.candidate_top_k, len(self.corpus))
        candidate_indices = np.argsort(-dense_scores)[:candidate_k]
        pairs = [[question.text, self.corpus[index].text] for index in candidate_indices]
        rerank_scores = self._reranker.compute_score(pairs, normalize=True)
        rerank_scores = np.asarray(rerank_scores, dtype=np.float32).reshape(-1)

        ranked = sorted(
            zip(candidate_indices.tolist(), rerank_scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        results: list[Evidence] = []
        for index, rerank_score in ranked[:requested_k]:
            evidence = self.corpus[index]
            metadata = dict(evidence.metadata)
            metadata.update(
                {
                    "dense_score": float(dense_scores[index]),
                    "reranker_score": float(rerank_score),
                    "retrieval_encoder": "BAAI/bge-m3",
                    "retrieval_reranker": "BAAI/bge-reranker-v2-m3",
                }
            )
            results.append(
                replace(
                    evidence,
                    relevance=float(max(0.0, min(1.0, rerank_score))),
                    metadata=metadata,
                )
            )
        return tuple(results)

    def _ensure_index(self) -> None:
        if self._doc_embeddings is None:
            self._doc_embeddings = self._encode([item.text for item in self.corpus])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self._encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        vectors = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        if self.normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        return vectors
