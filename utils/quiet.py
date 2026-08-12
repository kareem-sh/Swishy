"""
Silence third-party native logging.

MediaPipe, TensorFlow Lite and abseil write directly to the process's stderr
from C++, before any Python logging configuration can intercept them. A normal
run emits this, none of which concerns the person using Swichy:

    INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
    W0000 ... inference_feedback_manager.cc:121] Feedback manager requires ...
    W0000 ... landmark_projection_calculator.cc:81] Using NORM_RECT without ...
    E0000 ... portable_clearcut_uploader.cc:90] Failed to send to clearcut ...
    === Source Location Trace: ===

The `clearcut` lines are MediaPipe's usage telemetry failing to reach Google.
They are harmless, they are logged at ERROR level, and they make a working run
look broken. Nothing in Swichy depends on any of it.

These are environment variables read by the native libraries when they load,
so they MUST be set before the first `import mediapipe`. That is why this
module applies them at import time and why entry points import it first.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from contextlib import contextmanager

# 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR. glog and TF read these when the
# shared library initialises; setting them afterwards has no effect.
_NATIVE_LOG_ENV = {
    "GLOG_minloglevel": "3",
    "GLOG_stderrthreshold": "3",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "ABSL_MIN_LOG_LEVEL": "3",
}


def silence_native_logs() -> None:
    """Quieten MediaPipe/TFLite/absl. Safe to call more than once."""
    for key, value in _NATIVE_LOG_ENV.items():
        # setdefault: an explicit value from the shell always wins, so a
        # developer can still turn the noise back on for debugging.
        os.environ.setdefault(key, value)

    try:
        from absl import logging as absl_logging
    except ImportError:
        return
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold(absl_logging.ERROR)
    # absl installs a handler on the root logger the first time it is used,
    # which duplicates records onto stderr. Swichy configures its own.
    logging.getLogger("absl").propagate = False


silence_native_logs()


# Lines the native libraries emit on every healthy run. Each is matched
# conservatively: anything NOT listed here is re-emitted, so a real error can
# never be swallowed by this filter.
_KNOWN_NOISE = (
    re.compile(r"^[WEI]\d{4}\s"),                        # absl/glog severity stamps
    re.compile(r"TensorFlow Lite XNNPACK delegate"),
    re.compile(r"inference_feedback_manager"),
    re.compile(r"landmark_projection_calculator"),
    re.compile(r"portable_clearcut_uploader|clearcut"),  # telemetry, offline
    re.compile(r"^=== Source Location Trace"),
    re.compile(r"^wireless/android/play/playlog"),
    re.compile(r"^\s*$"),
)


def _is_known_noise(line: str) -> bool:
    return any(pattern.search(line) for pattern in _KNOWN_NOISE)


@contextmanager
def quiet_native_stderr():
    """Swallow known native-library chatter written straight to fd 2.

    MediaPipe's C++ layer writes to the file descriptor, not to `sys.stderr`,
    so Python-level redirection cannot see it. The descriptor is pointed at a
    temporary file for the duration and inspected afterwards: recognised noise
    is dropped, and **anything unrecognised is re-emitted**, so this hides
    clutter without ever hiding a genuine error.

    Degrades to a no-op wherever the descriptor cannot be duplicated.
    """
    try:
        saved_fd = os.dup(2)
    except (OSError, AttributeError, ValueError):
        yield
        return

    spool = tempfile.TemporaryFile(mode="w+b")
    try:
        sys.stderr.flush()
        os.dup2(spool.fileno(), 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        spool.seek(0)
        captured = spool.read().decode("utf-8", "replace")
        spool.close()
        kept = [ln for ln in captured.splitlines() if not _is_known_noise(ln)]
        if kept:
            sys.stderr.write("\n".join(kept) + "\n")
            sys.stderr.flush()
