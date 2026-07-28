from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import sys
import threading

from trove_core.application.dispatcher import build_default_dispatcher
from trove_core.reply import (
    APIReplyGenerator,
    CodexReplyGenerator,
    ContextBridge,
    DraftGenerationCoordinator,
    GeneratorConfig,
    ReplyAgentWorkspace,
    ReplyService,
    ReplyServiceConfig,
    ReplyStore,
)
from trove_core.vault.config import VaultConfig
from trove_core.sync import read_sync_config, resolve_snapshot_dir

from .lifecycle import RuntimeIdentity, build_identity, catalog_identity
from .runtime_owner import RuntimeOwner
from .provider_loader import (
    ProviderLoader, discover_provider_distributions, official_provider_registry,
)
from .server import DEFAULT_IDLE_TIMEOUT_SECONDS, DaemonServer
from .secrets import load_key_store_secret, read_agent_switch_secret
from .operator_auth import SignedOperatorAuthorizer


def _hash(value: str) -> str:
    if not re.fullmatch(r'[0-9a-f]{64}', value):
        raise argparse.ArgumentTypeError('expected one lowercase sha256')
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='troved')
    parser.add_argument('--vault', required=True)
    parser.add_argument('--build-hash', type=_hash)
    parser.add_argument('--catalog-hash', type=_hash)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--queue-size', type=int, default=32)
    parser.add_argument('--idle-timeout', type=float, default=DEFAULT_IDLE_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actual_build = build_identity()
    actual_catalog = catalog_identity()
    if args.build_hash is not None and args.build_hash != actual_build:
        print('troved: requested build hash does not match this artifact', file=sys.stderr)
        return 2
    if args.catalog_hash is not None and args.catalog_hash != actual_catalog:
        print('troved: requested catalog hash does not match this artifact', file=sys.stderr)
        return 2
    config = VaultConfig.resolve(args.vault, env={})
    config.ensure()
    reply_config = ReplyServiceConfig.load(config.root)
    identity = RuntimeIdentity.for_vault(
        config.root, build_hash=actual_build, catalog_hash=actual_catalog,
    )
    runtime_owner = RuntimeOwner(config, read_pool_size=args.workers, search_workers=args.workers)
    provider_registry = official_provider_registry()
    provider_loader = ProviderLoader(
        provider_registry, runtime_dir=identity.runtime_dir / 'providers',
    )
    providers = [
        item for item in discover_provider_distributions()
        if item.provider_id == 'wechat-source'
    ]
    if len(providers) != 1:
        provider_load = {
            'ok': False, 'provider_id': 'wechat-source',
            'error': {
                'code': 'provider_distribution_missing' if not providers else 'provider_distribution_ambiguous',
                'retryable': False,
            },
            'pure_vault_read_available': True,
        }
    else:
        sync_config = read_sync_config(config)
        source_root = resolve_snapshot_dir(config, sync_config)
        provider_kwargs = {
            'vault_root': config.root,
            **({'source_root': source_root} if source_root.is_dir() else {}),
        }
        if reply_config.configured:
            try:
                key_store = load_key_store_secret(
                    reply_config.key_store_secret,
                )
            except Exception:
                key_store = None
            if key_store is not None:
                provider_kwargs.update({
                    'live_config': {
                        'account_id': reply_config.account_id,
                        'source_account_id': (
                            reply_config.source_account_id
                        ),
                        'conversation_namespace': (
                            reply_config.conversation_namespace
                        ),
                        'account_id_sha256': (
                            reply_config.account_id_sha256
                        ),
                        # The daemon-owned service is the arm/disarm authority.
                        # The in-process adapter stays ready so it can be armed
                        # without rebuilding the verified provider.
                        'enabled': True,
                        'send_shortcut': reply_config.send_shortcut,
                        'max_reply_chars': reply_config.max_reply_chars,
                        'private_chats_enabled': True,
                        'groups_enabled': False,
                    },
                    'key_store': key_store,
                    'runtime_root': (
                        config.root / 'jobs' / 'reply' / 'provider'
                    ),
                })
        provider_load = provider_loader.load_distribution(
            providers[0],
            provider_kwargs=provider_kwargs,
        ).to_dict()
    runtime_owner.attach_provider_registry(provider_registry, provider_load)
    if reply_config.configured and provider_load.get('ok') is True:
        workspace = ReplyAgentWorkspace.for_vault(
            config.root, agent_id=reply_config.agent_id,
        )
        generator_config = GeneratorConfig(
            backend=reply_config.reply_backend,
            model=reply_config.model,
            style_profile_path=reply_config.style_profile_path,
            session_idle_days=reply_config.session_idle_days,
            max_reply_chars=reply_config.max_reply_chars,
            api_base_url=reply_config.api_base_url,
        )
        if reply_config.reply_backend == 'api':
            generator = APIReplyGenerator(
                generator_config,
                workspace,
                secret_supplier=lambda: read_agent_switch_secret(
                    reply_config.api_key_secret,
                ),
            )
        else:
            generator = CodexReplyGenerator(generator_config, workspace)
        store = ReplyStore.for_vault(config.root)
        generation = DraftGenerationCoordinator(
            store,
            ContextBridge(
                config,
                history_limit=reply_config.context_message_cap,
                workspace=workspace,
            ),
            generator,
        )
        reply_service = ReplyService(
            config.root,
            reply_config,
            action=lambda payload: provider_registry.invoke(
                reply_config.provider_id, 'action', payload,
            ),
            generation=generation,
            store=store,
        )
        runtime_owner.attach_reply_service(reply_service)
        reply_service.start()
    dispatcher = build_default_dispatcher(config, runtime_owner=runtime_owner)

    def operator_control(
        action: str,
        payload: dict,
    ) -> dict:
        service = runtime_owner.reply_service
        if service is None:
            raise RuntimeError('reply service is unavailable')
        if action == 'reply.arm':
            return service.arm()
        if action == 'reply.disarm':
            return service.disarm()
        if action == 'reply.set_mode':
            return service.set_mode(str(payload['mode']))
        if action in {'reply.approve', 'reply.reject'}:
            decision = (
                'approved' if action == 'reply.approve' else 'rejected'
            )
            return service.decide_review(
                str(payload['review_id']), decision,
            )
        if action == 'reply.retry':
            return service.retry_review(str(payload['review_id']))
        raise ValueError('unsupported operator action')

    operator_authorizer = SignedOperatorAuthorizer(config.root)
    server = DaemonServer(
        identity, dispatcher,
        max_workers=args.workers, max_pending=args.queue_size,
        idle_timeout=args.idle_timeout,
        managed_nonce=os.environ.get('TROVE_MANAGED_NONCE'),
        keepalive=lambda: runtime_owner.keepalive_required,
        operator_authorizer=operator_authorizer.authorize,
        operator_control=operator_control,
    )
    stopping = threading.Event()

    def stop(_signum, _frame):
        if not stopping.is_set():
            stopping.set()
            server.stop(timeout=5.0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.start()
        server.serve_forever()
    finally:
        server.stop(timeout=5.0)
        runtime_owner.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
