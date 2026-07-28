from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class HybridEmbedding:
    """One provider result with dense and optional sparse coordinates."""

    dense: list[float]
    sparse: dict[int, float]

class EmbeddingProvider(ABC):
    name: str = 'base'
    dimensions: int = 0
    egress_kind: str | None = None

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def embed_hybrid_many(
        self,
        texts: list[str],
        *,
        text_type: str = 'document',
        instruct: str | None = None,
    ) -> list[HybridEmbedding]:
        _ = instruct
        return [HybridEmbedding(dense=vector, sparse={}) for vector in self.embed_many(texts)]

    def embed_query_hybrid(self, text: str) -> HybridEmbedding:
        return HybridEmbedding(dense=self.embed_query(text), sparse={})
