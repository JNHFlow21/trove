from __future__ import annotations
from abc import ABC, abstractmethod
from sqlite3 import Row

class VectorStore(ABC):
    @abstractmethod
    def index_all_messages(self, provider) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, filters: dict[str, str] | None = None, limit: int = 10) -> list[Row]:
        raise NotImplementedError
