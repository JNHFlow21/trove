from __future__ import annotations
import hashlib
import math

from .base import EmbeddingProvider

class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic offline embedding provider for tests and CI."""
    name = 'fake'

    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        buckets = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            idx = int.from_bytes(digest[:2], 'big') % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            buckets[idx] += sign
        norm = math.sqrt(sum(v*v for v in buckets)) or 1.0
        return [round(v / norm, 6) for v in buckets]

    def _tokens(self, text: str) -> list[str]:
        chars = [c for c in text.lower() if not c.isspace()]
        grams = chars[:]
        grams.extend(''.join(chars[i:i+2]) for i in range(max(0, len(chars)-1)))
        return grams or ['']
