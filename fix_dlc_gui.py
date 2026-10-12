"""Re-apply the local DeepLabCut GUI fixes after reinstalling deeplabcut.

The GUI segfaults on startup on Windows: PySide6's Qt6Network is built against
OpenSSL 3.0.x, but Python's `ssl` module (pulled in by numpy/torch/urllib) has
already loaded OpenSSL 3.2.x into the process. Qt binds to those symbols and
dies with an access violation on the first HTTPS request -- which is the
startup update check in deeplabcut.gui.utils.UpdateChecker.

Switching Qt to the native Windows schannel backend avoids the clash. This
patches deeplabcut/gui/launch_script.py in the active environment; run it again
after any `uv sync` or `pip install` that reinstalls deeplabcut.

    uv run python fix_dlc_gui.py
"""

import sys
from pathlib import Path

HELPER = '''logger = logging.getLogger(__name__)


def _use_native_tls_backend():
    """Work around a hard crash in Qt's OpenSSL TLS backend on Windows.

    PySide6's Qt6Network is built against OpenSSL 3.0.x, but by the time the GUI
    starts, Python's own ``ssl`` module (pulled in transitively by numpy, torch
    and urllib) has already loaded OpenSSL 3.2.x into the process. Qt binds to
    those symbols and dies with an access violation on the first HTTPS request,
    which is the startup update check in ``UpdateChecker``. Schannel is the
    native Windows backend and has no such conflict.
    """
    if sys.platform != "win32":
        return
    try:
        from PySide6.QtNetwork import QSslSocket

        if (
            QSslSocket.activeBackend() != "schannel"
            and "schannel" in QSslSocket.availableBackends()
        ):
            QSslSocket.setActiveBackend("schannel")
    except Exception as e:
        logger.warning(f"Could not select the schannel TLS backend: {e}")
'''

MARKER = "_use_native_tls_backend"
ANCHOR_LOGGER = "logger = logging.getLogger(__name__)\n"
ANCHOR_DEF = "def launch_dlc():\n    app = QtWidgets.QApplication(sys.argv)"
PATCHED_DEF = (
    "def launch_dlc():\n    _use_native_tls_backend()\n\n"
    "    app = QtWidgets.QApplication(sys.argv)"
)


def main() -> int:
    import deeplabcut.gui.launch_script as mod

    path = Path(mod.__file__)
    src = path.read_text(encoding="utf-8")

    if MARKER in src:
        print(f"already patched: {path}")
        return 0

    for anchor in (ANCHOR_LOGGER, ANCHOR_DEF):
        if src.count(anchor) != 1:
            print(f"anchor not found exactly once, refusing to patch: {anchor!r}")
            return 1

    src = src.replace(ANCHOR_LOGGER, HELPER, 1).replace(ANCHOR_DEF, PATCHED_DEF, 1)
    path.write_text(src, encoding="utf-8")
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
