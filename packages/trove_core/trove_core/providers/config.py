from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import subprocess
from typing import Any, Mapping

from trove_core.bounds import RERANK_CANDIDATES
from trove_core.security.subprocess_env import agent_switch_subprocess_environment

ASR_SECRET_NAME = 'VOLCENGINE_ASR_API_KEY'
VISION_SECRET_NAME = 'VOLCENGINE_ARK_API_KEY'
CLOUD_RETRIEVAL_SECRET_NAME = 'DASHSCOPE_API_KEY'
CLOUD_COST_CAP_ENV = 'TROVE_CLOUD_COST_CAP_RMB'
LEGACY_COST_CAP_ENV = 'TROVE_WECHAT_COST_CAP_RMB'

DEFAULT_ASR_ENDPOINT = 'https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash'
DEFAULT_ASR_MODEL_NAME = 'bigmodel'
DEFAULT_ASR_RESOURCE_ID = 'volc.bigasr.auc_turbo'

DEFAULT_ARK_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'
DEFAULT_ARK_RESPONSES_PATH = '/responses'
DEFAULT_ARK_CHAT_COMPLETIONS_PATH = '/chat/completions'  # legacy-compatible fallback, not the default image path
DEFAULT_ARK_VISION_MODEL = 'doubao-seed-2-0-lite-260215'

DEFAULT_CLOUD_EMBEDDING_PROVIDER = 'aliyun'
DEFAULT_ALIYUN_EMBEDDING_ENDPOINT = 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding'
DEFAULT_ALIYUN_EMBEDDING_MODEL = 'text-embedding-v4'
DEFAULT_ALIYUN_EMBEDDING_DIMENSIONS = 1024
DEFAULT_VOLCENGINE_EMBEDDING_ENDPOINT = 'https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal'
DEFAULT_VOLCENGINE_EMBEDDING_MODEL = 'doubao-embedding-vision-251215'

DEFAULT_CLOUD_RERANK_PROVIDER = 'aliyun'
DEFAULT_ALIYUN_RERANK_ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-api/v1/reranks'
DEFAULT_ALIYUN_RERANK_MODEL = 'qwen3-rerank'
DEFAULT_CLOUD_RERANK_TOP_K = 20


def agent_switch_secret_names(timeout: float = 2.0) -> set[str]:
    """Return Agent Switch secret names only; values are never requested or logged."""
    try:
        proc = subprocess.run(
            ['agent-switch', 'secret', 'list'],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=agent_switch_subprocess_environment(),
        )
    except Exception:
        return set()
    if proc.returncode != 0:
        return set()
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        name = line.strip().split()[0] if line.strip() else ''
        if name and all(ch.isalnum() or ch == '_' for ch in name):
            names.add(name)
    return names


@dataclass(frozen=True)
class SecretPresence:
    name: str
    present: bool
    source: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderDiagnostic:
    provider: str
    configured: bool
    enabled: bool
    secret: SecretPresence
    model: str
    endpoint: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Keep the public contract explicit: secret values never appear here.
        data['secret'] = self.secret.to_dict()
        data['secret_value_included'] = False
        return data


@dataclass(frozen=True)
class ProviderConfig:
    asr_secret_name: str = ASR_SECRET_NAME
    vision_secret_name: str = VISION_SECRET_NAME
    cloud_embedding_secret_name: str = CLOUD_RETRIEVAL_SECRET_NAME
    cloud_rerank_secret_name: str = CLOUD_RETRIEVAL_SECRET_NAME
    asr_endpoint: str = DEFAULT_ASR_ENDPOINT
    asr_model_name: str = DEFAULT_ASR_MODEL_NAME
    asr_resource_id: str = DEFAULT_ASR_RESOURCE_ID
    ark_base_url: str = DEFAULT_ARK_BASE_URL
    ark_chat_path: str = DEFAULT_ARK_RESPONSES_PATH
    ark_vision_model: str = DEFAULT_ARK_VISION_MODEL
    cloud_embedding_provider: str = DEFAULT_CLOUD_EMBEDDING_PROVIDER
    cloud_embedding_endpoint: str = DEFAULT_ALIYUN_EMBEDDING_ENDPOINT
    cloud_embedding_model: str = DEFAULT_ALIYUN_EMBEDDING_MODEL
    cloud_embedding_dimensions: int = DEFAULT_ALIYUN_EMBEDDING_DIMENSIONS
    cloud_embedding_request_format: str = 'dashscope-native'
    cloud_rerank_provider: str = DEFAULT_CLOUD_RERANK_PROVIDER
    cloud_rerank_endpoint: str = DEFAULT_ALIYUN_RERANK_ENDPOINT
    cloud_rerank_model: str = DEFAULT_ALIYUN_RERANK_MODEL
    cloud_rerank_top_k: int = DEFAULT_CLOUD_RERANK_TOP_K
    cloud_asr_enabled: bool = False
    cloud_vision_enabled: bool = False
    cloud_embedding_enabled: bool = False
    cloud_rerank_enabled: bool = False

    @classmethod
    def resolve(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
    ) -> 'ProviderConfig':
        env = env if env is not None else os.environ
        embedding_provider = env.get('TROVE_CLOUD_EMBEDDING_PROVIDER', DEFAULT_CLOUD_EMBEDDING_PROVIDER).strip().lower()
        embedding_endpoint_default = DEFAULT_VOLCENGINE_EMBEDDING_ENDPOINT if embedding_provider == 'volcengine' else DEFAULT_ALIYUN_EMBEDDING_ENDPOINT
        embedding_model_default = DEFAULT_VOLCENGINE_EMBEDDING_MODEL if embedding_provider == 'volcengine' else DEFAULT_ALIYUN_EMBEDDING_MODEL
        embedding_key_default = VISION_SECRET_NAME if embedding_provider == 'volcengine' else CLOUD_RETRIEVAL_SECRET_NAME
        embedding_format_default = 'volcengine-multimodal' if embedding_provider == 'volcengine' else 'dashscope-native'
        try:
            embedding_dimensions = int(env.get('TROVE_CLOUD_EMBEDDING_DIMENSIONS', str(DEFAULT_ALIYUN_EMBEDDING_DIMENSIONS)))
        except ValueError:
            embedding_dimensions = DEFAULT_ALIYUN_EMBEDDING_DIMENSIONS
        try:
            rerank_top_k = int(env.get('TROVE_CLOUD_RERANK_TOP_K', str(DEFAULT_CLOUD_RERANK_TOP_K)))
        except ValueError:
            rerank_top_k = DEFAULT_CLOUD_RERANK_TOP_K

        return cls(
            asr_secret_name=env.get('TROVE_ASR_SECRET_NAME', ASR_SECRET_NAME),
            vision_secret_name=env.get('TROVE_VISION_SECRET_NAME', VISION_SECRET_NAME),
            cloud_embedding_secret_name=env.get('TROVE_CLOUD_EMBEDDING_KEY_ENV', embedding_key_default),
            cloud_rerank_secret_name=env.get('TROVE_CLOUD_RERANK_KEY_ENV', CLOUD_RETRIEVAL_SECRET_NAME),
            asr_endpoint=env.get('TROVE_VOLCENGINE_ASR_ENDPOINT', DEFAULT_ASR_ENDPOINT),
            asr_model_name=env.get('TROVE_VOLCENGINE_ASR_MODEL_NAME', DEFAULT_ASR_MODEL_NAME),
            asr_resource_id=env.get('TROVE_VOLCENGINE_ASR_RESOURCE_ID', DEFAULT_ASR_RESOURCE_ID),
            ark_base_url=env.get('TROVE_VOLCENGINE_ARK_BASE_URL', DEFAULT_ARK_BASE_URL),
            ark_chat_path=env.get('TROVE_VOLCENGINE_ARK_RESPONSES_PATH') or env.get('TROVE_VOLCENGINE_ARK_CHAT_PATH', DEFAULT_ARK_RESPONSES_PATH),
            ark_vision_model=env.get('TROVE_VOLCENGINE_ARK_VISION_MODEL', DEFAULT_ARK_VISION_MODEL),
            cloud_embedding_provider=embedding_provider,
            cloud_embedding_endpoint=env.get('TROVE_CLOUD_EMBEDDING_ENDPOINT', embedding_endpoint_default),
            cloud_embedding_model=env.get('TROVE_CLOUD_EMBEDDING_MODEL', embedding_model_default),
            cloud_embedding_dimensions=embedding_dimensions,
            cloud_embedding_request_format=env.get('TROVE_CLOUD_EMBEDDING_REQUEST_FORMAT', embedding_format_default),
            cloud_rerank_provider=env.get('TROVE_CLOUD_RERANK_PROVIDER', DEFAULT_CLOUD_RERANK_PROVIDER).strip().lower(),
            cloud_rerank_endpoint=env.get('TROVE_CLOUD_RERANK_ENDPOINT', DEFAULT_ALIYUN_RERANK_ENDPOINT),
            cloud_rerank_model=env.get('TROVE_CLOUD_RERANK_MODEL', DEFAULT_ALIYUN_RERANK_MODEL),
            cloud_rerank_top_k=max(1, min(rerank_top_k, RERANK_CANDIDATES.maximum)),
            cloud_asr_enabled=env.get('TROVE_ENABLE_CLOUD_ASR', '').lower() in {'1', 'true', 'yes'},
            cloud_vision_enabled=env.get('TROVE_ENABLE_CLOUD_VISION', '').lower() in {'1', 'true', 'yes'},
            cloud_embedding_enabled=env.get('TROVE_ENABLE_CLOUD_EMBEDDING', '').lower() in {'1', 'true', 'yes'},
            cloud_rerank_enabled=env.get('TROVE_ENABLE_CLOUD_RERANK', '').lower() in {'1', 'true', 'yes'},
        )

    def secret_presence(
        self,
        name: str,
        env: Mapping[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
    ) -> SecretPresence:
        from .factory import EnvironmentAgentSwitchSecretResolver

        source = os.environ if env is None else env
        names = agent_switch_names
        if names is None and not check_agent_switch:
            names = set()
        return EnvironmentAgentSwitchSecretResolver(
            source,
            agent_switch_names=names,
            assume_agent_switch_transport=bool(agent_switch_names is not None and not check_agent_switch),
        ).presence(name)


    def _factory_readiness(
        self,
        kind: str,
        env: Mapping[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
    ) -> Any:
        from .factory import EnvironmentAgentSwitchSecretResolver, ProviderFactory

        source = os.environ if env is None else env
        names = agent_switch_names
        if names is None and not check_agent_switch:
            names = set()
        resolver = EnvironmentAgentSwitchSecretResolver(
            source,
            agent_switch_names=names,
            assume_agent_switch_transport=bool(agent_switch_names is not None and not check_agent_switch),
        )
        return ProviderFactory(self, env=source, secret_resolver=resolver).readiness(kind)

    def asr_status(
        self,
        env: dict[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
    ) -> Any:
        return self._factory_readiness(
            'asr', env,
            agent_switch_names=agent_switch_names,
            check_agent_switch=check_agent_switch,
        )

    def vision_status(
        self,
        env: dict[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
    ) -> Any:
        return self._factory_readiness(
            'vision', env,
            agent_switch_names=agent_switch_names,
            check_agent_switch=check_agent_switch,
        )

    def embedding_status(
        self,
        env: dict[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
    ) -> Any:
        return self._factory_readiness(
            'embedding', env,
            agent_switch_names=agent_switch_names,
            check_agent_switch=check_agent_switch,
        )

    def rerank_status(
        self,
        env: dict[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
    ) -> Any:
        return self._factory_readiness(
            'rerank', env,
            agent_switch_names=agent_switch_names,
            check_agent_switch=check_agent_switch,
        )

    def to_redacted_dict(
        self,
        env: Mapping[str, str] | None = None,
        *,
        agent_switch_names: set[str] | None = None,
        check_agent_switch: bool = True,
        secret_resolver: Any | None = None,
        dependency_probe: Any | None = None,
    ) -> dict[str, Any]:
        # Import lazily to keep the immutable configuration module independent
        # from optional provider implementations.
        from .factory import EnvironmentAgentSwitchSecretResolver, ProviderFactory

        source = os.environ if env is None else env
        if secret_resolver is None:
            names = agent_switch_names
            if names is None and not check_agent_switch:
                names = set()
            secret_resolver = EnvironmentAgentSwitchSecretResolver(
                source,
                agent_switch_names=names,
                assume_agent_switch_transport=bool(agent_switch_names is not None and not check_agent_switch),
            )
        return ProviderFactory(
            self,
            env=source,
            secret_resolver=secret_resolver,
            dependency_probe=dependency_probe,
        ).status_payload()


def configured_cost_cap_rmb(env: Mapping[str, str] | None = None) -> float | None:
    env = env if env is not None else os.environ
    raw = env.get(CLOUD_COST_CAP_ENV) or env.get(LEGACY_COST_CAP_ENV)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def provider_status_payload(
    env: Mapping[str, str] | None = None,
    *,
    check_agent_switch: bool = True,
    secret_resolver: Any | None = None,
    dependency_probe: Any | None = None,
) -> dict[str, Any]:
    source = os.environ if env is None else env
    cfg = ProviderConfig.resolve(source, check_agent_switch=False)
    data = cfg.to_redacted_dict(
        source,
        agent_switch_names=None if check_agent_switch else set(),
        check_agent_switch=check_agent_switch,
        secret_resolver=secret_resolver,
        dependency_probe=dependency_probe,
    )
    cap = configured_cost_cap_rmb(source)
    data['cost_cap'] = {'configured': cap is not None, 'currency': 'RMB', 'value_included': False}
    return data
