from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Any

from .models import (
    RunningApp,
    WindowRef,
    WORK_EXECUTABLE_PATH,
)


_SET_FRONTMOST_SCRIPT = r'''
on run argv
    set appPid to item 1 of argv as integer
    tell application "System Events"
        set matches to application processes whose unix id is appPid
        if (count of matches) is not 1 then error "pid match failed"
        set frontmost of item 1 of matches to true
        delay 0.08
        return (unix id of first application process whose frontmost is true) as string
    end tell
end run
'''

_DEMINIATURIZE_SCRIPT = r'''
on run argv
    set appPid to item 1 of argv as integer
    tell application "System Events"
        set matches to application processes whose unix id is appPid
        if (count of matches) is not 1 then error "pid match failed"
        set proc to item 1 of matches
        set winCount to count of windows of proc
        repeat with i from 1 to winCount
            try
                set value of attribute "AXMinimized" of window i of proc to false
            end try
        end repeat
        return winCount as string
    end tell
end run
'''


class PreservedPasteboard:
    def __init__(self, appkit: Any) -> None:
        self._appkit = appkit
        self.pasteboard = appkit.NSPasteboard.generalPasteboard()
        self._items: list[dict[object, object]] = []

    def __enter__(self) -> 'PreservedPasteboard':
        for item in self.pasteboard.pasteboardItems() or []:
            values: dict[object, object] = {}
            for pasteboard_type in item.types() or []:
                data = item.dataForType_(pasteboard_type)
                if data is not None:
                    values[pasteboard_type] = data
            self._items.append(values)
        return self

    def set_text(self, value: str) -> None:
        self.pasteboard.clearContents()
        if not self.pasteboard.setString_forType_(
            value, self._appkit.NSPasteboardTypeString,
        ):
            raise RuntimeError('pasteboard_write_failed')

    def text(self) -> str:
        return str(
            self.pasteboard.stringForType_(self._appkit.NSPasteboardTypeString)
            or ''
        )

    def __exit__(self, *_args: object) -> None:
        self.pasteboard.clearContents()
        restored = []
        for values in self._items:
            item = self._appkit.NSPasteboardItem.alloc().init()
            for pasteboard_type, data in values.items():
                item.setData_forType_(data, pasteboard_type)
            restored.append(item)
        if restored:
            self.pasteboard.writeObjects_(restored)


class MacOSSenderUI:
    """Lazy PyObjC bridge so read-only Provider import has no UI dependency."""

    def __init__(self) -> None:
        try:
            import AppKit
            import Quartz
        except ImportError as exc:
            raise RuntimeError('macos_ui_dependency_missing') from exc
        self.appkit = AppKit
        self.quartz = Quartz

    def resolve_exact_running_app(
        self,
        bundle_id: str,
        app_path: str,
    ) -> RunningApp:
        expected = Path(app_path).resolve()
        applications = list(
            self.appkit.NSRunningApplication
            .runningApplicationsWithBundleIdentifier_(bundle_id)
            or []
        )
        matches: list[RunningApp] = []
        for application in applications:
            bundle_url = application.bundleURL()
            executable_url = application.executableURL()
            if bundle_url is None or executable_url is None:
                continue
            actual_path = Path(str(bundle_url.path())).resolve()
            if actual_path != expected:
                continue
            matches.append(RunningApp(
                int(application.processIdentifier()),
                bundle_id,
                actual_path,
                Path(str(executable_url.path())).resolve(),
            ))
        if len(matches) != 1:
            raise RuntimeError(f'exact_work_app_match_count:{len(matches)}')
        if matches[0].executable_path != WORK_EXECUTABLE_PATH.resolve():
            raise RuntimeError('work_app_executable_mismatch')
        return matches[0]

    def main_window_for_pid(self, pid: int) -> WindowRef:
        options = (
            self.quartz.kCGWindowListOptionOnScreenOnly
            | self.quartz.kCGWindowListExcludeDesktopElements
        )
        windows: list[WindowRef] = []
        for item in (
            self.quartz.CGWindowListCopyWindowInfo(
                options, self.quartz.kCGNullWindowID,
            )
            or []
        ):
            if int(item.get(self.quartz.kCGWindowOwnerPID, 0)) != int(pid):
                continue
            if (
                int(item.get(self.quartz.kCGWindowLayer, -1)) != 0
                or float(item.get(self.quartz.kCGWindowAlpha, 0)) <= 0
            ):
                continue
            bounds = item.get(self.quartz.kCGWindowBounds) or {}
            window = WindowRef(
                int(item.get(self.quartz.kCGWindowNumber, 0)),
                float(bounds.get('X', 0)),
                float(bounds.get('Y', 0)),
                float(bounds.get('Width', 0)),
                float(bounds.get('Height', 0)),
            )
            if window.width >= 760 and window.height >= 600:
                windows.append(window)
        if len(windows) != 1:
            raise RuntimeError(f'work_main_window_match_count:{len(windows)}')
        return windows[0]

    def frontmost_pid(self) -> int:
        script = (
            'tell application "System Events" to return '
            '(unix id of first application process whose frontmost is true) as string'
        )
        result = subprocess.run(
            ['/usr/bin/osascript', '-e', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            try:
                return int(result.stdout.strip())
            except ValueError:
                pass
        app = self.appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
        return int(app.processIdentifier()) if app is not None else 0

    @staticmethod
    def _set_frontmost(pid: int) -> bool:
        result = subprocess.run(
            ['/usr/bin/osascript', '-e', _SET_FRONTMOST_SCRIPT, str(int(pid))],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == str(int(pid))

    @staticmethod
    def _deminiaturize(pid: int) -> None:
        subprocess.run(
            ['/usr/bin/osascript', '-e', _DEMINIATURIZE_SCRIPT, str(int(pid))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )

    def activate_exact_pid(self, pid: int, *, attempts: int = 8) -> None:
        application = (
            self.appkit.NSRunningApplication
            .runningApplicationWithProcessIdentifier_(int(pid))
        )
        if application is None:
            raise RuntimeError('work_app_not_running')
        options = (
            self.appkit.NSApplicationActivateIgnoringOtherApps
            | self.appkit.NSApplicationActivateAllWindows
        )
        for attempt in range(max(1, attempts)):
            application.unhide()
            application.activateWithOptions_(options)
            if attempt >= 1:
                self._set_frontmost(pid)
            time.sleep(0.08 + attempt * 0.03)
            if self.frontmost_pid() == int(pid):
                self._deminiaturize(pid)
                time.sleep(0.1)
                return
        raise RuntimeError('frontmost_pid_mismatch')

    def restore_frontmost_pid(self, pid: int) -> None:
        if not pid:
            return
        application = (
            self.appkit.NSRunningApplication
            .runningApplicationWithProcessIdentifier_(int(pid))
        )
        if application is not None:
            application.activateWithOptions_(
                self.appkit.NSApplicationActivateIgnoringOtherApps,
            )
            if self.frontmost_pid() != int(pid):
                self._set_frontmost(pid)

    def pasteboard(self) -> PreservedPasteboard:
        return PreservedPasteboard(self.appkit)
