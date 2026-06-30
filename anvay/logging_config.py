from __future__ import annotations

import logging
import os
import sys


class _TqdmLoggingHandler(logging.StreamHandler):
    """StreamHandler that routes records through ``tqdm.write`` so an active
    progress bar stays anchored at the bottom instead of being pushed up by
    interleaved log lines. Falls back to a plain stderr write if tqdm is
    unavailable."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            from tqdm import tqdm

            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except ImportError:
            super().emit(record)
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    """Configure process-wide logging once, with env overrides for local/dev runs."""
    level_name = os.getenv("ANVAY_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = os.getenv(
        "ANVAY_LOG_FORMAT",
        "%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    root = logging.getLogger()
    if not root.handlers:
        handler = _TqdmLoggingHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)

    root.setLevel(level)
    logging.getLogger("anvay").setLevel(level)
