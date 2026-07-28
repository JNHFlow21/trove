from __future__ import annotations

from typing import Any, Mapping

from trove_protocol.provider import ProviderManifest, validate_provider_hello


class CurrentSourceAdapter:
    """Temporary contract wrapper used before source package extraction in U7."""

    def __init__(self, manifest: ProviderManifest, source: object):
        self.manifest = manifest
        self.source = source

    def hello(self) -> Mapping[str, Any]:
        payload = self.source.hello()
        validate_provider_hello(self.manifest, payload)
        return payload

    def capabilities(self) -> Mapping[str, Any]:
        return self.source.capabilities()

    def health(self) -> Mapping[str, Any]:
        return self.source.health()

    def accounts(self) -> list[Mapping[str, Any]]:
        return self.source.accounts()

    def invoke(self, method: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.source.invoke(method, dict(payload))


__all__ = ['CurrentSourceAdapter']
