from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import os
import re
import hashlib
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from trove_core.embedding.model_registry import default_local_model_path, resolve_model_spec
from trove_core.wechat.decrypt.secrets import AgentSwitchSecretResolver, SecretResolutionError

from .config import ProviderConfig, SecretPresence, agent_switch_secret_names

_SECRET_NAME_RE = re.compile(r'^[A-Za-z0-9_]+$')
_PROVIDER_KINDS = {'asr', 'vision', 'embedding', 'rerank'}
_MAX_SECRET_BYTES = 64 * 1024


def _safe_identifier(value: str, *, prefix: str) -> str:
    if (
        type(value) is str
        and 0 < len(value) <= 256
        and not value.startswith(('/', '~'))
        and '..' not in value
        and not any(char in value for char in ('?', '#', '\\', '\r', '\n'))
        and re.fullmatch(r'[A-Za-z0-9._:/-]+', value)
    ):
        return value
    digest = hashlib.sha256(str(value).encode('utf-8', errors='replace')).hexdigest()[:16]
    return f'{prefix}-{digest}'


def _safe_endpoint(value: str) -> tuple[str, str]:
    digest = hashlib.sha256(str(value).encode('utf-8', errors='replace')).hexdigest()
    try:
        parsed = urlsplit(value)
        host = parsed.hostname or ''
        if parsed.scheme not in {'http', 'https'} or not host:
            raise ValueError
        port = f':{parsed.port}' if parsed.port is not None else ''
        return f'{parsed.scheme}://{host}{port}', digest
    except Exception:
        return 'redacted-endpoint', digest


class SecretResolver(Protocol):
    """Resolve secret presence and values through one non-logging contract."""

    def presence(self, name: str) -> SecretPresence: ...

    def resolve(self, name: str) -> str: ...


class EnvironmentAgentSwitchSecretResolver:
    """Agent Switch-only resolver with secure FD value transport.

    The historical class name is retained for API compatibility. ``env`` is
    accepted but its values are deliberately ignored: environment variables may
    select a secret *name*, but they are never a provider credential source.
    Status asks Agent Switch for names only. Invocation uses ``secret get --fd``;
    neither path exposes values through command arguments, stdout, diagnostics,
    or ``repr``.
    """

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        names_loader: Callable[[], set[str]] | None = None,
        agent_switch: AgentSwitchSecretResolver | None = None,
        assume_agent_switch_transport: bool = False,
    ) -> None:
        # Do not retain the supplied mapping.  It can contain ambient or custom
        # provider secret values and exists only for backwards-compatible call
        # signatures; Agent Switch is the sole value authority.
        _ = env
        self._names = set(agent_switch_names) if agent_switch_names is not None else None
        self._names_loader = names_loader or agent_switch_secret_names
        self._agent_switch = agent_switch or AgentSwitchSecretResolver()
        self._secure_get_supported: bool | None = True if assume_agent_switch_transport else None

    def __repr__(self) -> str:  # pragma: no cover - defensive redaction boundary
        return '<EnvironmentAgentSwitchSecretResolver redacted>'

    @staticmethod
    def _validate_name(name: str) -> str:
        if type(name) is not str or not 1 <= len(name) <= 256 or not _SECRET_NAME_RE.fullmatch(name):
            raise SecretResolutionError('invalid_secret_name')
        return name

    def _listed_names(self) -> set[str]:
        if self._names is None:
            try:
                self._names = set(self._names_loader())
            except Exception:
                self._names = set()
        return self._names

    @staticmethod
    def _valid_value(value: object) -> bool:
        return (
            type(value) is str
            and 0 < len(value.encode('utf-8')) <= _MAX_SECRET_BYTES
            and not any(char in value for char in ('\x00', '\r', '\n'))
        )

    def _secure_agent_switch_available(self) -> bool:
        if self._secure_get_supported is None:
            try:
                self._secure_get_supported = bool(self._agent_switch._supports_fd_secret_get())
            except Exception:
                self._secure_get_supported = False
        return self._secure_get_supported

    def presence(self, name: str) -> SecretPresence:
        name = self._validate_name(name)
        if name in self._listed_names():
            if self._secure_agent_switch_available():
                return SecretPresence(name=name, present=True, source='agent-switch')
            return SecretPresence(name=name, present=False, source='agent-switch-unavailable')
        return SecretPresence(name=name, present=False, source=None)

    def resolve(self, name: str) -> str:
        name = self._validate_name(name)
        if name not in self._listed_names() or not self._secure_agent_switch_available():
            raise SecretResolutionError('missing_key')
        value = self._agent_switch.get_secret(name)
        if not self._valid_value(value):
            raise SecretResolutionError('missing_key')
        return value


class ProviderUnavailable(RuntimeError):
    """Typed, redacted failure for an explicitly selected provider."""

    def __init__(self, kind: str, code: str) -> None:
        if kind not in _PROVIDER_KINDS and kind != 'local_embedding':
            kind = 'provider'
        self.kind = kind
        self.code = code
        super().__init__(code)

    def to_dict(self) -> dict[str, Any]:
        return {
            'type': 'provider_unavailable',
            'provider_kind': self.kind,
            'reason_code': self.code,
            'retryable': self.code in {'provider_credential_resolution_failed'},
            'secret_value_included': False,
            'raw_content_included': False,
            'raw_paths_included': False,
        }


@dataclass(frozen=True)
class ProviderReadiness:
    kind: str
    provider: str
    model: str
    endpoint: str
    endpoint_hash: str
    enabled: bool
    configured: bool
    dependency: str | None
    dependency_available: bool
    ready: bool
    state: str
    reason_code: str | None
    secret: SecretPresence
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['secret'] = self.secret.to_dict()
        data['notes'] = list(self.notes)
        data['secret_value_included'] = False
        data['raw_content_included'] = False
        data['raw_paths_included'] = False
        return data


@dataclass(frozen=True)
class LocalProviderSelection:
    provider: object | None
    state: str
    reason_code: str | None
    explicit: bool

    def status(self) -> dict[str, Any]:
        return {
            'state': self.state,
            'reason_code': self.reason_code,
            'explicit': self.explicit,
            'provider_configured': self.provider is not None,
            'fallback_mode': 'exact_fts' if self.provider is None else None,
            'raw_paths_included': False,
        }


class ProviderFactory:
    """Single construction/readiness authority for optional providers."""

    @staticmethod
    def adapt_current_source(manifest: Any, source: Any):
        """Expose an existing source through the process-neutral provider contract."""
        from .adapters.current_source import CurrentSourceAdapter

        return CurrentSourceAdapter(manifest, source)

    def __init__(
        self,
        config: ProviderConfig,
        *,
        env: Mapping[str, str] | None = None,
        secret_resolver: SecretResolver | None = None,
        dependency_probe: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config
        source = os.environ if env is None else env
        # Only the two local-provider selection switches are needed after
        # configuration resolution.  Never retain ambient provider secrets in
        # the factory simply because the caller supplied an environment map.
        self.env = {
            key: str(source[key])
            for key in ('TROVE_DISABLE_LOCAL_EMBEDDING', 'TROVE_EMBEDDING_MODEL_PATH')
            if key in source
        }
        self.secret_resolver = secret_resolver or EnvironmentAgentSwitchSecretResolver(self.env)
        self._dependency_probe = dependency_probe or self._default_dependency_probe

    @classmethod
    def resolve(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        secret_resolver: SecretResolver | None = None,
        dependency_probe: Callable[[str], bool] | None = None,
        agent_switch_names: set[str] | None = None,
    ) -> 'ProviderFactory':
        source = os.environ if env is None else env
        config = ProviderConfig.resolve(source, check_agent_switch=False)
        resolver = secret_resolver or EnvironmentAgentSwitchSecretResolver(
            source,
            agent_switch_names=agent_switch_names,
        )
        return cls(
            config,
            env=source,
            secret_resolver=resolver,
            dependency_probe=dependency_probe,
        )

    @staticmethod
    def _default_dependency_probe(module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except Exception:
            return False

    def _settings(self, kind: str) -> tuple[str, str, str, str, bool, str | None, tuple[str, ...], bool]:
        cfg = self.config
        if kind == 'asr':
            return (
                'volcengine-asr-flash',
                f'{cfg.asr_model_name}:{cfg.asr_resource_id}',
                cfg.asr_endpoint,
                cfg.asr_secret_name,
                cfg.cloud_asr_enabled,
                None,
                tuple(
                    note
                    for condition, note in (
                        (cfg.asr_model_name != 'bigmodel', 'ASR model override differs from the pinned plan default.'),
                        (cfg.asr_resource_id != 'volc.bigasr.auc_turbo', 'ASR resource override differs from the pinned plan default.'),
                        (not cfg.cloud_asr_enabled, 'Cloud ASR calls are disabled until TROVE_ENABLE_CLOUD_ASR is explicit.'),
                    )
                    if condition
                ),
                bool(cfg.asr_model_name and cfg.asr_resource_id and cfg.asr_endpoint),
            )
        if kind == 'vision':
            return (
                'volcengine-ark-vision-lite',
                cfg.ark_vision_model,
                cfg.ark_base_url.rstrip('/') + cfg.ark_chat_path,
                cfg.vision_secret_name,
                cfg.cloud_vision_enabled,
                None,
                tuple(
                    note
                    for condition, note in (
                        (cfg.ark_vision_model != 'doubao-seed-2-0-lite-260215', 'Ark vision model override differs from the pinned plan default.'),
                        (not cfg.cloud_vision_enabled, 'Cloud vision calls are disabled until TROVE_ENABLE_CLOUD_VISION is explicit.'),
                    )
                    if condition
                ),
                bool(cfg.ark_vision_model and cfg.ark_base_url and cfg.ark_chat_path),
            )
        if kind == 'embedding':
            return (
                f'{cfg.cloud_embedding_provider}-embedding',
                f'{cfg.cloud_embedding_model}:{cfg.cloud_embedding_dimensions}',
                cfg.cloud_embedding_endpoint,
                cfg.cloud_embedding_secret_name,
                cfg.cloud_embedding_enabled,
                'httpx',
                tuple(
                    note
                    for condition, note in (
                        (not cfg.cloud_embedding_enabled, 'Cloud embedding calls are disabled until TROVE_ENABLE_CLOUD_EMBEDDING is explicit.'),
                        (cfg.cloud_embedding_provider != 'aliyun', 'Cloud embedding provider override differs from the default Alibaba text path.'),
                        (cfg.cloud_embedding_dimensions <= 0, 'Cloud embedding dimensions are not explicitly configured.'),
                    )
                    if condition
                ),
                bool(
                    cfg.cloud_embedding_provider
                    and cfg.cloud_embedding_model
                    and cfg.cloud_embedding_endpoint
                    and cfg.cloud_embedding_dimensions > 0
                    and cfg.cloud_embedding_request_format in {'openai', 'dashscope-native', 'volcengine-multimodal'}
                ),
            )
        if kind == 'rerank':
            return (
                f'{cfg.cloud_rerank_provider}-rerank',
                cfg.cloud_rerank_model,
                cfg.cloud_rerank_endpoint,
                cfg.cloud_rerank_secret_name,
                cfg.cloud_rerank_enabled,
                'httpx',
                tuple([
                    f'Default candidate window: top {cfg.cloud_rerank_top_k}.',
                    *(
                        ['Cloud rerank calls are disabled until TROVE_ENABLE_CLOUD_RERANK is explicit.']
                        if not cfg.cloud_rerank_enabled else []
                    ),
                    *(
                        ['Cloud rerank provider override differs from the default Alibaba qwen3 path.']
                        if cfg.cloud_rerank_provider != 'aliyun' else []
                    ),
                ]),
                bool(cfg.cloud_rerank_provider and cfg.cloud_rerank_model and cfg.cloud_rerank_endpoint),
            )
        raise ValueError('unknown_provider_kind')

    def readiness(self, kind: str) -> ProviderReadiness:
        provider, model, endpoint, secret_name, enabled, dependency, notes, valid = self._settings(kind)
        try:
            secret = self.secret_resolver.presence(secret_name)
        except Exception:
            secret = SecretPresence(name='invalid_secret_name', present=False, source=None)
        public_endpoint, endpoint_hash = _safe_endpoint(endpoint)
        public_provider = _safe_identifier(provider, prefix='provider')
        public_model = _safe_identifier(model, prefix='model')
        dependency_available = dependency is None or self._dependency_probe(dependency)
        if not enabled:
            state, reason = 'disabled', 'provider_not_enabled'
        elif not valid:
            state, reason = 'unavailable', 'provider_configuration_invalid'
        elif not secret.present:
            state, reason = 'unavailable', 'provider_credential_missing'
        elif not dependency_available:
            state, reason = 'unavailable', 'provider_dependency_missing'
        else:
            state, reason = 'available', None
        return ProviderReadiness(
            kind=kind,
            provider=public_provider,
            model=public_model,
            endpoint=public_endpoint,
            endpoint_hash=endpoint_hash,
            enabled=enabled,
            configured=secret.present and valid,
            dependency=dependency,
            dependency_available=dependency_available,
            ready=state == 'available',
            state=state,
            reason_code=reason,
            secret=secret,
            notes=notes,
        )

    def status_payload(self) -> dict[str, Any]:
        statuses = {kind: self.readiness(kind).to_dict() for kind in ('asr', 'vision', 'embedding', 'rerank')}
        enabled = [status['provider'] for status in statuses.values() if status['enabled']]
        return {
            'cloud_exception_only': enabled,
            'cloud_retrieval_requires_explicit_opt_in': True,
            'cloud_embedding_enabled': self.config.cloud_embedding_enabled,
            'cloud_rerank_enabled': self.config.cloud_rerank_enabled,
            'cloud_chat_enabled': False,
            'providers': statuses,
            'status_invocation_contract': 'provider-factory-v2',
            'secret_transport': 'agent-switch-fd-only',
            'environment_secret_values_accepted': False,
            'secret_values_included': False,
        }

    def _secret_for(self, kind: str) -> str:
        readiness = self.readiness(kind)
        if not readiness.ready:
            raise ProviderUnavailable(kind, readiness.reason_code or 'provider_unavailable')
        try:
            value = self.secret_resolver.resolve(readiness.secret.name)
        except Exception:
            raise ProviderUnavailable(kind, 'provider_credential_resolution_failed') from None
        if type(value) is not str or not value:
            raise ProviderUnavailable(kind, 'provider_credential_resolution_failed')
        return value

    def create_asr(self, **kwargs: Any):
        from trove_core.asr.volcengine_flash import VolcengineASRFlashProvider

        secret = self._secret_for('asr')
        provider = VolcengineASRFlashProvider(
            api_key=secret,
            endpoint=self.config.asr_endpoint,
            **kwargs,
        )
        provider.model_name = self.config.asr_model_name
        provider.resource_id = self.config.asr_resource_id
        return provider

    def create_vision(self, **kwargs: Any):
        from trove_core.vision.volcengine_ark import VolcengineArkVisionProvider

        secret = self._secret_for('vision')
        provider = VolcengineArkVisionProvider(
            api_key=secret,
            base_url=self.config.ark_base_url,
            responses_path=self.config.ark_chat_path,
            **kwargs,
        )
        provider.model = self.config.ark_vision_model
        return provider

    def create_cloud_embedding(self, **kwargs: Any):
        from trove_core.embedding.openai_compatible_provider import OpenAICompatibleEmbeddingProvider

        secret = self._secret_for('embedding')
        return OpenAICompatibleEmbeddingProvider(
            enabled=True,
            endpoint=self.config.cloud_embedding_endpoint,
            model=self.config.cloud_embedding_model,
            api_key=secret,
            api_key_name=self.config.cloud_embedding_secret_name,
            dimensions=self.config.cloud_embedding_dimensions,
            request_format=self.config.cloud_embedding_request_format,
            provider_name=self.config.cloud_embedding_provider,
            **kwargs,
        )

    def create_cloud_reranker(self, **kwargs: Any):
        from trove_core.search.cloud_reranker import CloudRerankProvider

        secret = self._secret_for('rerank')
        return CloudRerankProvider(
            enabled=True,
            endpoint=self.config.cloud_rerank_endpoint,
            model=self.config.cloud_rerank_model,
            api_key=secret,
            api_key_name=self.config.cloud_rerank_secret_name,
            provider_name=self.config.cloud_rerank_provider,
            **kwargs,
        )

    def select_local_embedding(self, model_path: str | Path | None = None) -> LocalProviderSelection:
        from trove_core.embedding.local_provider import LocalEmbeddingProvider

        explicit = model_path is not None or bool(self.env.get('TROVE_EMBEDDING_MODEL_PATH'))
        if self.env.get('TROVE_DISABLE_LOCAL_EMBEDDING') == '1':
            return LocalProviderSelection(None, 'unavailable_fallback', 'local_embedding_disabled', explicit)
        configured_path = model_path or self.env.get('TROVE_EMBEDDING_MODEL_PATH')
        if configured_path:
            path = Path(configured_path).expanduser()
            auto_discovered = False
        else:
            discovered = default_local_model_path()
            if discovered is None:
                return LocalProviderSelection(None, 'unavailable_fallback', 'local_embedding_model_missing', False)
            path = discovered
            auto_discovered = True
        if not path.exists():
            if explicit:
                raise ProviderUnavailable('local_embedding', 'local_embedding_model_missing')
            return LocalProviderSelection(None, 'unavailable_fallback', 'local_embedding_model_missing', False)
        if not self._dependency_probe('sentence_transformers'):
            if explicit:
                raise ProviderUnavailable('local_embedding', 'local_embedding_dependency_missing')
            return LocalProviderSelection(None, 'unavailable_fallback', 'local_embedding_dependency_missing', False)
        spec = resolve_model_spec(None)
        provider = LocalEmbeddingProvider(path, dimensions=spec.dimensions if auto_discovered else 0)
        provider.auto_discovered = auto_discovered
        provider.model_id = spec.model_id if auto_discovered else getattr(provider, 'model_id', '')
        return LocalProviderSelection(provider, 'available', None, explicit)
