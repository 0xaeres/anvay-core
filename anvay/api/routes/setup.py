"""Setup routes — skills_repo configuration status.

The skills_repo URL and token are operator configuration, set via
ANVAY_SKILLS_REPO (env or anvay.yaml) and ANVAY_SKILLS_REPO_TOKEN before
starting the server. There is no runtime wizard; if either is missing the
server refuses to start in production (see app.py lifespan check).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from anvay.api.deps import get_config_dep
from anvay.config import AnvayConfig

log = logging.getLogger(__name__)

router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/status")
async def setup_status(config: AnvayConfig = Depends(get_config_dep)) -> dict:
    url = config.skills_repo or ""
    return {
        "configured": bool(url),
        "skills_repo_url": url or None,
    }
