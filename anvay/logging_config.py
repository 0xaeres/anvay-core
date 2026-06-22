from __future__ import annotations

import logging
import os
import sys


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
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)

    root.setLevel(level)
    logging.getLogger("anvay").setLevel(level)
