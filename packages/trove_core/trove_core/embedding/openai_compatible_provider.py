from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Callable

from .base import EmbeddingProvider, HybridEmbedding


def _httpx_post(*args: Any, **kwargs: Any) -> Any:
    from trove_core.providers.http_pool import post

    return post(*args, **kwargs)

class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    name = 'openai-compatible'
    egress_kind = 'cloud_embedding_upload'

    def __init__(
        self,
        *,
        enabled: bool = False,
        endpoint: str | None = None,
        model: str | None = None,
        api_key_env: str = 'TROVE_CLOUD_EMBEDDING_KEY',
        api_key: str | None = None,
        api_key_name: str | None = None,
        dimensions: int | None = None,
        request_format: str = 'openai',
        provider_name: str = 'cloud',
        timeout: float = 60.0,
        max_batch_size: int = 10,
        max_workers: int = 16,
        max_retries: int = 6,
        query_instruct: str = 'Given a private Chinese chat knowledge-base query, retrieve the relevant message evidence.',
        post: Callable[..., Any] | None = None,
    ):
        if not enabled:
            raise RuntimeError('Cloud embeddings are disabled by default; enable explicitly after accepting that selected snippets leave the machine.')
        if not endpoint or not model:
            raise RuntimeError('Cloud embeddings require explicit endpoint and model configuration.')
        if not api_key:
            raise RuntimeError('Cloud embedding credential must be supplied explicitly by ProviderFactory.')
        self.endpoint = endpoint
        self.model = model
        self._api_key = api_key
        self.api_key_name = api_key_name or api_key_env
        self.dimensions = int(dimensions or 0)
        self.request_format = request_format
        self.provider_name = provider_name
        self.name = f'{provider_name}:{model}'
        self.timeout = timeout
        self.max_batch_size = max(1, min(int(max_batch_size), 10))
        self.max_workers = max(1, min(int(max_workers), 32))
        self.max_retries = max(0, min(int(max_retries), 8))
        self.query_instruct = str(query_instruct)
        self.supports_sparse = request_format == 'dashscope-native'
        self.provider_calls = 0
        self.input_tokens = 0
        self._usage_lock = threading.Lock()
        self._post = post or _httpx_post

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_query(self, text: str) -> list[float]:
        if self.request_format == 'dashscope-native':
            return self.embed_query_hybrid(text).dense
        return super().embed_query(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.request_format == 'dashscope-native':
            return [item.dense for item in self.embed_hybrid_many(texts, text_type='document')]
        if self.request_format == 'volcengine-multimodal':
            return [self._embed_volcengine_multimodal(text) for text in texts]
        return self._embed_openai_batch(texts)

    def embed_hybrid_many(
        self,
        texts: list[str],
        *,
        text_type: str = 'document',
        instruct: str | None = None,
    ) -> list[HybridEmbedding]:
        if not texts:
            return []
        if self.request_format != 'dashscope-native':
            return super().embed_hybrid_many(texts, text_type=text_type, instruct=instruct)
        if text_type not in {'query', 'document'}:
            raise ValueError('cloud_embedding_text_type_invalid')
        batches = [(index, texts[index:index + self.max_batch_size]) for index in range(0, len(texts), self.max_batch_size)]
        if len(batches) == 1:
            return self._embed_dashscope_native_batch(
                batches[0][1], text_type=text_type,
                instruct=(instruct if instruct is not None else self.query_instruct) if text_type == 'query' else None,
            )
        completed: dict[int, list[HybridEmbedding]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches)), thread_name_prefix='trove-cloud-embed') as executor:
            futures = {
                executor.submit(
                    self._embed_dashscope_native_batch,
                    batch,
                    text_type=text_type,
                    instruct=(instruct if instruct is not None else self.query_instruct) if text_type == 'query' else None,
                ): start
                for start, batch in batches
            }
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        return [item for start, _batch in batches for item in completed[start]]

    def embed_query_hybrid(self, text: str) -> HybridEmbedding:
        if self.request_format != 'dashscope-native':
            return super().embed_query_hybrid(text)
        return self.embed_hybrid_many([text], text_type='query')[0]

    def _headers(self) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_code = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(self.endpoint, headers=self._headers(), json=payload, timeout=self.timeout)
                status_code = int(getattr(response, 'status_code', 0) or 0)
                last_code = status_code
                try:
                    data = response.json()
                except Exception as exc:
                    raise RuntimeError(f'cloud_embedding_http_{status_code}') from exc
                retryable = status_code in {408, 409, 425, 429} or status_code >= 500
                if status_code >= 400:
                    if retryable and attempt < self.max_retries:
                        retry_after = 0.0
                        headers = getattr(response, 'headers', None)
                        if headers is not None:
                            try:
                                retry_after = float(headers.get('Retry-After') or 0.0)
                            except (TypeError, ValueError):
                                retry_after = 0.0
                        base = 1.0 if status_code == 429 else 0.25
                        time.sleep(min(30.0, max(retry_after, base * (2 ** attempt))))
                        continue
                    code = ''
                    if isinstance(data, dict):
                        error = data.get('error') or {}
                        if isinstance(error, dict):
                            code = str(error.get('code') or error.get('type') or '')
                        if not code:
                            code = str(data.get('code') or '')
                    code = ''.join(ch for ch in code if ch.isalnum() or ch in {'-', '_'})[:80]
                    raise RuntimeError(f'cloud_embedding_http_{status_code}{("_" + code) if code else ""}')
                if not isinstance(data, dict):
                    raise RuntimeError('cloud_embedding_invalid_response')
                usage = data.get('usage')
                tokens = int(usage.get('total_tokens') or 0) if isinstance(usage, dict) else 0
                with self._usage_lock:
                    self.provider_calls += 1
                    self.input_tokens += max(0, tokens)
                return data
            except RuntimeError:
                raise
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f'cloud_embedding_transport_{last_code}') from exc
                time.sleep(min(30.0, 0.5 * (2 ** attempt)))
        raise RuntimeError(f'cloud_embedding_http_{last_code}')

    def _embed_dashscope_native_batch(self, texts: list[str], *, text_type: str, instruct: str | None) -> list[HybridEmbedding]:
        parameters: dict[str, Any] = {
            'dimension': self.dimensions,
            'output_type': 'dense&sparse',
            'text_type': text_type,
        }
        if instruct:
            parameters['instruct'] = instruct
        try:
            data = self._post_json({
                'model': self.model,
                'input': {'texts': texts},
                'parameters': parameters,
            })
        except RuntimeError as exc:
            # DashScope applies a token budget to the whole native batch.  A
            # valid collection can therefore contain a rare batch whose ten
            # individually bounded documents exceed the aggregate budget.
            # Deterministically split only 400 responses; never skip a source
            # document or publish an incomplete collection.
            # Split only the aggregate input validation failure.  Account,
            # credential, quota, and arrearage errors apply to every child
            # request; recursively splitting those would amplify a known
            # failure into many pointless provider calls.
            if (
                str(exc).startswith('cloud_embedding_http_400_InvalidParameter')
                and len(texts) > 1
            ):
                middle = len(texts) // 2
                return [
                    *self._embed_dashscope_native_batch(texts[:middle], text_type=text_type, instruct=instruct),
                    *self._embed_dashscope_native_batch(texts[middle:], text_type=text_type, instruct=instruct),
                ]
            raise
        output = data.get('output')
        rows = output.get('embeddings') if isinstance(output, dict) else None
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise RuntimeError('cloud_embedding_count_mismatch')
        try:
            ordered = sorted(rows, key=lambda row: int(row.get('text_index') or 0))
        except Exception as exc:
            raise RuntimeError('cloud_embedding_index_invalid') from exc
        result: list[HybridEmbedding] = []
        for row in ordered:
            dense = row.get('embedding') if isinstance(row, dict) else None
            sparse = row.get('sparse_embedding') if isinstance(row, dict) else None
            if not isinstance(dense, list) or not isinstance(sparse, list):
                raise RuntimeError('cloud_embedding_shape_invalid')
            if self.dimensions and len(dense) != self.dimensions:
                raise RuntimeError('cloud_embedding_dimension_mismatch')
            sparse_values: dict[int, float] = {}
            for item in sparse:
                if not isinstance(item, dict):
                    raise RuntimeError('cloud_embedding_sparse_item_invalid')
                try:
                    index = int(item['index'])
                    value = float(item['value'])
                except Exception as exc:
                    raise RuntimeError('cloud_embedding_sparse_item_invalid') from exc
                if index < 0:
                    raise RuntimeError('cloud_embedding_sparse_item_invalid')
                sparse_values[index] = value
            result.append(HybridEmbedding([float(value) for value in dense], sparse_values))
        return result

    def _embed_openai_batch(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            'model': self.model,
            'input': texts,
            'encoding_format': 'float',
        }
        if self.dimensions:
            payload['dimensions'] = self.dimensions
        data = self._post_json(payload)
        rows = data.get('data')
        if not isinstance(rows, list):
            raise RuntimeError('cloud_embedding_missing_data')
        vectors: list[list[float]] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('embedding'), list):
                raise RuntimeError('cloud_embedding_missing_vector')
            vectors.append([float(x) for x in row['embedding']])
        if vectors and not self.dimensions:
            self.dimensions = len(vectors[0])
        return vectors

    def _embed_volcengine_multimodal(self, text: str) -> list[float]:
        payload = {
            'model': self.model,
            'encoding_format': 'float',
            'input': [{'type': 'text', 'text': text}],
        }
        if self.dimensions:
            payload['dimensions'] = self.dimensions
        data = self._post_json(payload)
        body = data.get('data')
        if not isinstance(body, dict):
            raise RuntimeError('cloud_embedding_missing_data')
        vector = body.get('embedding')
        if not isinstance(vector, list):
            raise RuntimeError('cloud_embedding_missing_vector')
        result = [float(x) for x in vector]
        if not self.dimensions:
            self.dimensions = len(result)
        return result
