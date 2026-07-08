"""Shared role constants and legacy-role migration map.

Used by both anvay/auth/store.py (session auth users) and anvay/registry.py
(product membership users) so the mapping has one owner.
"""

from __future__ import annotations

ROLES = {"admin", "editor", "viewer"}

LEGACY_ROLE_MAP = {
    "org_admin": "admin",
    "product_admin": "editor",
    "sme": "viewer",
}


def normalize_role(role: str) -> str:
    return LEGACY_ROLE_MAP.get(role, role)
