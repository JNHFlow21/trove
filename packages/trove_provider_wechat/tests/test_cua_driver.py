from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from trove_provider_wechat.reply.cua_driver import (
    CuaDriver,
    EXPECTED_CUA_EXECUTABLE,
)


class CuaDriverReadinessTests(unittest.TestCase):
    def test_requires_accessibility_but_not_screen_capture(self) -> None:
        driver = CuaDriver()
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_call(
            tool: str,
            payload: dict[str, object],
        ) -> dict[str, object]:
            calls.append((tool, payload))
            return {
                'accessibility': True,
                'screen_recording': True,
                'screen_recording_capturable': False,
                'direct_capture_status': 'unavailable',
                'source': {'executable': str(EXPECTED_CUA_EXECUTABLE)},
            }

        driver.call = fake_call  # type: ignore[method-assign]

        driver.ensure_ready()

        self.assertEqual(
            calls,
            [('check_permissions', {'prompt': False})],
        )

    def test_rejects_missing_accessibility(self) -> None:
        driver = CuaDriver()
        driver.call = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
            'accessibility': False,
            'source': {'executable': str(EXPECTED_CUA_EXECUTABLE)},
        }

        with self.assertRaisesRegex(
            RuntimeError,
            'cua_driver_accessibility_not_granted',
        ):
            driver.ensure_ready()

    def test_rejects_wrong_helper_identity(self) -> None:
        driver = CuaDriver()
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / 'other-helper'
            driver.call = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
                'accessibility': True,
                'source': {'executable': str(wrong)},
            }

            with self.assertRaisesRegex(
                RuntimeError,
                'cua_driver_identity_mismatch',
            ):
                driver.ensure_ready()


if __name__ == '__main__':
    unittest.main()
