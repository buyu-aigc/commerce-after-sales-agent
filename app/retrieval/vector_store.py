from __future__ import annotations

import hashlib
import math
import os
import re
from array import array
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class VectorDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchHit:
    doc_id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def search(
        self,
        query: str,
        *,
        k: int = 5,
        candidate_ids: set[str] | None = None,
    ) -> list[VectorSearchHit]:
        """Return nearest documents for a text query."""


class HashingTextEmbedder:
    """Deterministic low-cost text embedder used for local tests and demos.

    Production deployments can replace this with a real embedding model while
    keeping the same VectorStore boundary.
    """

    def __init__(self, dimensions: int = 65536) -> None:
        self.dimensions = dimensions

    def embed_sparse(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        for token, weight in _features(text):
            bucket = _stable_bucket(token, self.dimensions)
            counts[bucket] = max(counts.get(bucket, 0.0), weight)
        norm = math.sqrt(sum(value * value for value in counts.values()))
        if norm <= 0:
            return {}
        return {bucket: value / norm for bucket, value in counts.items()}

    def embed_dense(self, text: str) -> list[float]:
        sparse = self.embed_sparse(text)
        dense = [0.0] * self.dimensions
        for bucket, value in sparse.items():
            dense[bucket] = value
        return dense


class LocalVectorStore:
    """Compact in-process hashed-vector index.

    The original demo representation retained every character 4-gram as a
    Python dict entry and then retained a second posting list for it.  With a
    35k-document support corpus that turns a small local demo into a multi-GB
    process. This representation stores high-dimensional sparse features in
    packed integer/float arrays. BM25 supplies the lexical candidate set and
    this index supplies product-name and typo-sensitive reranking without a
    second Python-object-heavy inverted index.
    """

    def __init__(
        self,
        documents: list[VectorDocument],
        embedder: HashingTextEmbedder | None = None,
    ) -> None:
        self._embedder = embedder or HashingTextEmbedder(dimensions=65536)
        self._documents = {doc.doc_id: doc for doc in documents}
        self._vectors = {
            doc.doc_id: _compact_sparse(self._embedder, doc.text)
            for doc in documents
        }

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        candidate_ids: set[str] | None = None,
    ) -> list[VectorSearchHit]:
        query_vector = self._embedder.embed_sparse(query)
        if not query_vector:
            return []
        document_ids = candidate_ids if candidate_ids is not None else self._vectors.keys()
        scores = {
            doc_id: _sparse_dot(query_vector, self._vectors[doc_id])
            for doc_id in document_ids
            if doc_id in self._vectors
        }
        # A candidate filter can restrict the search to documents with no
        # semantic overlap. Returning a zero-score hit looks like evidence to
        # callers, while Qdrant treats an empty meaningful result as no hit.
        ranked = sorted(
            ((doc_id, score) for doc_id, score in scores.items() if score > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )[:k]
        return [
            VectorSearchHit(
                doc_id=doc_id,
                score=score,
                metadata=self._documents[doc_id].metadata,
            )
            for doc_id, score in ranked
        ]


def _compact_sparse(embedder: HashingTextEmbedder, text: str) -> tuple[array, array]:
    """Pack sparse hash buckets into arrays rather than persistent dicts."""
    sparse = embedder.embed_sparse(text)
    buckets = array("H")
    values = array("f")
    for bucket, value in sorted(sparse.items()):
        buckets.append(bucket)
        values.append(value)
    return buckets, values


def _sparse_dot(query: dict[int, float], packed: tuple[array, array]) -> float:
    buckets, values = packed
    return sum(query.get(bucket, 0.0) * values[index] for index, bucket in enumerate(buckets))


class QdrantVectorStore:
    """Qdrant-backed vector store adapter.

    This adapter is intentionally thin: the project can run without Qdrant, and
    Docker deployments can switch to it by installing `qdrant-client` and
    setting SUPPORT_VECTOR_BACKEND=qdrant.
    """

    def __init__(
        self,
        *,
        url: str,
        collection_name: str,
        documents: list[VectorDocument],
        embedder: HashingTextEmbedder | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, PointStruct, VectorParams
        except ModuleNotFoundError as exc:  # pragma: no cover - optional backend
            raise RuntimeError("Install vector support with `pip install .[vector]`.") from exc

        self._embedder = embedder or HashingTextEmbedder(dimensions=384)
        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        existing = {collection.name for collection in self._client.get_collections().collections}
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self._embedder.dimensions, distance=Distance.COSINE),
            )
        count = self._client.count(collection_name=collection_name, exact=False).count
        if count == 0:
            self._client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=_qdrant_point_id(doc.doc_id),
                        vector=self._embedder.embed_dense(doc.text),
                        payload={"doc_id": doc.doc_id, **doc.metadata},
                    )
                    for doc in documents
                ],
            )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        candidate_ids: set[str] | None = None,
    ) -> list[VectorSearchHit]:
        query_vector = self._embedder.embed_dense(query)
        query_filter = None
        if candidate_ids:
            try:
                from qdrant_client.http.models import FieldCondition, Filter, MatchAny
            except ModuleNotFoundError as exc:  # pragma: no cover - optional backend
                raise RuntimeError("Install vector support with `pip install .[vector]`.") from exc
            query_filter = Filter(
                must=[FieldCondition(key="doc_id", match=MatchAny(any=list(candidate_ids)))]
            )
        hits = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=k,
        )
        return [
            VectorSearchHit(
                doc_id=str(hit.payload.get("doc_id", hit.id)),
                score=float(hit.score),
                metadata=dict(hit.payload or {}),
            )
            for hit in hits
        ]


def build_vector_store_from_env(documents: list[VectorDocument]) -> VectorStore:
    backend = os.environ.get("SUPPORT_VECTOR_BACKEND", "local").strip().lower()
    if backend in {"local", "memory", "inmemory", ""}:
        return LocalVectorStore(documents)
    if backend == "qdrant":
        return QdrantVectorStore(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.environ.get("QDRANT_COLLECTION", "olist_support_examples"),
            documents=documents,
        )
    raise RuntimeError(f"Unsupported SUPPORT_VECTOR_BACKEND: {backend}")


def _features(text: str) -> list[tuple[str, float]]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    words = re.findall(r"[a-z0-9]+", normalized)
    if words:
        # ResCommons is English e-commerce dialogue.  Word features avoid
        # materialising every character window while retaining lexical and
        # short-phrase affinity for the vector-fusion signal.
        words = words[:96]
        features = [(f"w:{word}", 1.0) for word in words]
        features.extend((f"b:{left}:{right}", 1.25) for left, right in zip(words, words[1:]))
        # The corpus stores the user utterance at the beginning of each
        # example.  Bounded character features recover typo/product-name
        # affinity without indexing every character of long generated replies.
        prefix = normalized[:640]
        features.extend((f"c:{prefix[index : index + 4]}", 0.45) for index in range(max(0, len(prefix) - 3)))
        return features

    # Policy titles may be Chinese and have no whitespace token boundaries.
    compact = re.sub(r"\s+", "", normalized)[:192]
    if len(compact) < 2:
        return [(f"c:{compact}", 1.0)] if compact else []
    return [(f"c:{compact[index : index + 2]}", 1.0) for index in range(len(compact) - 1)]


def _stable_bucket(token: str, dimensions: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def _qdrant_point_id(doc_id: str) -> int:
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)
