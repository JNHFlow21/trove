from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_MISSING = object()


@dataclass(frozen=True)
class BoundSpec:
    minimum: int
    maximum: int
    default: int


SEARCH_RESULTS = BoundSpec(1, 50, 10)
PRIVATE_LIST = BoundSpec(1, 500, 100)
TRACE_EVENTS_APPROVALS = BoundSpec(1, 200, 100)
PROFILE_WIKI_REPORT = BoundSpec(1, 50, 5)
RETRIEVAL_CANDIDATES = BoundSpec(1, 200, 200)
FUSION_CANDIDATES = BoundSpec(1, 200, 200)
RERANK_CANDIDATES = BoundSpec(1, 200, 50)
CONTEXT_WINDOW = BoundSpec(0, 200, 5)


class BoundedInputError(ValueError):
    """Protocol-neutral validation failure for resource-bounded integers."""

    code = 'invalid_limit'

    def __init__(self, field: str, value: Any, spec: BoundSpec):
        self.field = field
        self.value_type = type(value).__name__
        self.minimum = spec.minimum
        self.maximum = spec.maximum
        super().__init__(f'{field} must be an integer between {spec.minimum} and {spec.maximum}')

    def to_dict(self) -> dict[str, Any]:
        return {
            'code': self.code,
            'field': self.field,
            'minimum': self.minimum,
            'maximum': self.maximum,
            'message': str(self),
        }


class BoundedLimit(int):
    """An exact integer proven to be inside one declared resource budget."""

    def __new__(
        cls,
        value: Any = _MISSING,
        *,
        field: str = 'limit',
        spec: BoundSpec = SEARCH_RESULTS,
    ) -> 'BoundedLimit':
        candidate = spec.default if value is _MISSING else value
        # bool is an int subclass, but accepting true as a result budget makes
        # JSON and Python adapters disagree.  Require the exact protocol type.
        if type(candidate) not in (int, BoundedLimit) or not spec.minimum <= candidate <= spec.maximum:
            raise BoundedInputError(field, candidate, spec)
        return int.__new__(cls, candidate)


def bounded_limit(value: Any = _MISSING, *, field: str = 'limit', spec: BoundSpec = SEARCH_RESULTS) -> int:
    return int(BoundedLimit(value, field=field, spec=spec))
