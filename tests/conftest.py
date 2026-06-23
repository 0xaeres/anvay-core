from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ANVAY_ADMIN_API_KEY",
        "ANVAY_BOOTSTRAP_ADMIN_EMAIL",
        "ANVAY_BOOTSTRAP_ADMIN_PASSWORD",
        "ANVAY_ENV",
        "ANVAY_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
