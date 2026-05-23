"""First-run setup — org-wide skills_repo bootstrap.

The skills_repo is the single Git repository per org that holds product skill
hierarchies under `{product_id}/` and shared standards under `shared/`. This
package owns the one-time bootstrap: either create a new repo via the GitHub
API or attach to an existing one, then seed `shared/` from the bundled starter
pack so the org starts with sensible defaults.
"""

from __future__ import annotations

from nexus.setup.bootstrap import (
    BootstrapError,
    BootstrapResult,
    bootstrap_skills_repo,
    starter_pack_root,
)
from nexus.setup.kv import SetupKV

__all__ = [
    "BootstrapError",
    "BootstrapResult",
    "SetupKV",
    "bootstrap_skills_repo",
    "starter_pack_root",
]
