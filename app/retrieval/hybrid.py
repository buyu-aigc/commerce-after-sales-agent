from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.vector_store import (
    VectorDocument,
    VectorSearchHit,
    VectorStore,
    build_vector_store_from_env,
)

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "data" / "rescommons_derived" / "support_corpus.jsonl"
VECTOR_CANDIDATES = 100
FUSION_CANDIDATES = 40
_VECTOR_STORE_CACHE: dict[tuple[str, int, str], VectorStore] = {}


@dataclass(frozen=True)
class RetrievalHit:
    doc: dict
    score: float


class HybridSupportRetriever:
    """Production-shaped local hybrid retrieval.

    The interfaces mirror an ES/BM25 + vector DB + reranker stack while keeping
    the project runnable without external services or API keys.
    """

    def __init__(
        self,
        corpus_path: Path = DEFAULT_CORPUS,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._corpus_path = corpus_path
        self._provided_vector_store = vector_store
        self.docs: list[dict] = []
        self._doc_idx_by_id: dict[str, int] = {}
        self.doc_tokens: list[list[str]] = []
        self.doc_counters: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avgdl = 0.0
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.vector_store: VectorStore | None = None
        self._initialized = False

    def _ensure_loaded(self) -> None:
        if self._initialized:
            return
        self.docs = _load_jsonl(self._corpus_path)
        self._doc_idx_by_id = {doc["doc_id"]: idx for idx, doc in enumerate(self.docs)}
        self.doc_tokens = [_tokens(doc["text"]) for doc in self.docs]
        self.doc_counters = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        for idx, counts in enumerate(self.doc_counters):
            for token, tf in counts.items():
                self.inverted[token].append((idx, tf))
        vector_documents = [
            VectorDocument(
                doc_id=str(doc["doc_id"]),
                text=str(doc["text"]),
                metadata=doc,
            )
            for doc in self.docs
        ]
        self.vector_store = self._provided_vector_store or _cached_vector_store(
            self._corpus_path,
            vector_documents,
        )
        self._initialized = True

    def bm25_search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        self._ensure_loaded()
        return self._rank(self._bm25_scores(query), k)

    def vector_search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        self._ensure_loaded()
        bm25_scores = self._bm25_scores(query)
        candidate_ids = {
            str(self.docs[idx]["doc_id"])
            for idx, _ in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[
                :VECTOR_CANDIDATES
            ]
        }
        return self._vector_hits_to_retrieval_hits(
            self.vector_store.search(query, k=k, candidate_ids=candidate_ids or None)
        )

    def hybrid_search(self, query: str, k: int = 5) -> list[RetrievalHit]:
        self._ensure_loaded()
        bm25_scores = self._bm25_scores(query)
        candidates = [
            idx
            for idx, _ in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[
                :VECTOR_CANDIDATES
            ]
        ]
        candidate_ids = {str(self.docs[idx]["doc_id"]) for idx in candidates}
        vector_scores = self._vector_scores(query, candidate_ids)
        scores: dict[int, float] = defaultdict(float)
        bm25_ranked = sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[:FUSION_CANDIDATES]
        vector_ranked = sorted(vector_scores.items(), key=lambda item: item[1], reverse=True)[
            :FUSION_CANDIDATES
        ]
        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            scores[idx] += 0.2 / (rank + 20)
        for rank, (idx, _) in enumerate(vector_ranked, start=1):
            scores[idx] += 2.0 / (rank + 20)
        return self._rank(scores, k)

    def _bm25_scores(self, query: str) -> dict[int, float]:
        query_tokens = _tokens(query)
        scores: dict[int, float] = defaultdict(float)
        n_docs = len(self.docs)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            postings = self.inverted.get(token, [])
            if not postings:
                continue
            idf = math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for idx, tf in postings:
                dl = self.doc_lengths[idx] or 1
                denom = tf + k1 * (1 - b + b * dl / self.avgdl)
                scores[idx] += idf * (tf * (k1 + 1) / denom)
        return scores

    def _vector_scores(self, query: str, candidate_ids: set[str] | None = None) -> dict[int, float]:
        assert self.vector_store is not None
        scores = {}
        for hit in self.vector_store.search(query, k=FUSION_CANDIDATES, candidate_ids=candidate_ids):
            idx = self._doc_idx_by_id.get(hit.doc_id)
            if idx is not None:
                scores[idx] = hit.score
        return scores

    def _rank(self, scores: dict[int, float], k: int) -> list[RetrievalHit]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [RetrievalHit(doc=self.docs[idx], score=score) for idx, score in ranked]

    def _vector_hits_to_retrieval_hits(self, hits: list[VectorSearchHit]) -> list[RetrievalHit]:
        retrieval_hits = []
        for hit in hits:
            idx = self._doc_idx_by_id.get(hit.doc_id)
            if idx is not None:
                retrieval_hits.append(RetrievalHit(doc=self.docs[idx], score=hit.score))
        return retrieval_hits


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _cached_vector_store(corpus_path: Path, documents: list[VectorDocument]) -> VectorStore:
    backend = os.environ.get("SUPPORT_VECTOR_BACKEND", "local").strip().lower()
    if backend not in {"local", "memory", "inmemory", ""}:
        return build_vector_store_from_env(documents)
    stat = corpus_path.stat()
    cache_key = (str(corpus_path.resolve()), int(stat.st_mtime), int(stat.st_size))
    cached = _VECTOR_STORE_CACHE.get(cache_key)
    if cached is None:
        cached = build_vector_store_from_env(documents)
        _VECTOR_STORE_CACHE[cache_key] = cached
    return cached


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]
