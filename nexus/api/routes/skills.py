"""Skills — see ENGINEERING.md §11."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from nexus.api.deps import get_proposal_queue, get_skill_store
from nexus.council.queue import ProposalQueue
from nexus.skills.models import OrgSkill, Skill
from nexus.skills.store import SkillStore

router = APIRouter(tags=["skills"])


@router.get("/products/{product_id}/skills")
async def list_product_skills(
    product_id: str, store: SkillStore = Depends(get_skill_store)
) -> dict:
    out_master: dict | None = None
    out_domain: list[dict] = []
    out_adopted: list[dict] = []
    for s in store.iter_skills():
        d = s.model_dump(mode="json")
        d["id"] = s.id
        if isinstance(s, OrgSkill):
            out_adopted.append(d)
        else:
            assert isinstance(s, Skill)
            if s.product != product_id:
                continue
            if str(s.kind) == "master":
                out_master = d
            else:
                out_domain.append(d)
    return {
        "master": out_master,
        "domain": out_domain,
        "adopted": out_adopted,
    }


@router.get("/skills/{skill_id:path}/corrections")
async def get_skill_corrections(
    skill_id: str,
    store: SkillStore = Depends(get_skill_store),
    queue: ProposalQueue = Depends(get_proposal_queue),
) -> dict:
    target_name = skill_id.split("/")[-1]
    skill = None
    for s in store.iter_skills():
        if s.id == skill_id or s.name == target_name:
            skill = s
            break
    if skill is None:
        raise HTTPException(status_code=404, detail="skill not found")

    product_id = skill_id.split("/")[0] if "/" in skill_id else None
    proposals = queue.list(product_id=product_id) if product_id else []
    approved = [p for p in proposals if p.get("status") == "approved" and p.get("skill_kind") == getattr(skill, "kind", None)]
    corrections = []
    for p in approved:
        critique = p.get("adversary_critique")
        if critique:
            corrections.append({
                "proposal_id": p["id"],
                "created_at": p.get("created_at"),
                "adversary_critique": critique,
            })

    built_in = getattr(getattr(skill, "provenance", None), "adversary_critique", None)
    return {"corrections": corrections, "adversary_critique": built_in}


@router.get("/skills/{skill_id:path}/rejections")
async def get_skill_rejections(
    skill_id: str,
    store: SkillStore = Depends(get_skill_store),
    queue: ProposalQueue = Depends(get_proposal_queue),
) -> dict:
    target_name = skill_id.split("/")[-1]
    skill = None
    for s in store.iter_skills():
        if s.id == skill_id or s.name == target_name:
            skill = s
            break
    if skill is None:
        raise HTTPException(status_code=404, detail="skill not found")

    product_id = skill_id.split("/")[0] if "/" in skill_id else None
    proposals = queue.list(status="rejected", product_id=product_id) if product_id else []
    skill_kind = str(getattr(skill, "kind", ""))
    rejections = [p for p in proposals if p.get("skill_kind") == skill_kind]
    return {"rejections": rejections}


@router.get("/skills/{skill_id:path}/council-history")
async def get_skill_council_history(
    skill_id: str,
    store: SkillStore = Depends(get_skill_store),
    queue: ProposalQueue = Depends(get_proposal_queue),
) -> dict:
    target_name = skill_id.split("/")[-1]
    skill = None
    for s in store.iter_skills():
        if s.id == skill_id or s.name == target_name:
            skill = s
            break
    if skill is None:
        raise HTTPException(status_code=404, detail="skill not found")

    product_id = skill_id.split("/")[0] if "/" in skill_id else None
    if not product_id:
        return {"sessions": []}
    sessions = queue.list_sessions(product_id=product_id)
    skill_kind = str(getattr(skill, "kind", ""))
    sessions = [s for s in sessions if s.get("skill_kind") == skill_kind]
    return {"sessions": sessions}


@router.get("/skills/{skill_id:path}")
async def get_skill(
    skill_id: str, store: SkillStore = Depends(get_skill_store)
) -> dict:
    target_name = skill_id.split("/")[-1]
    for s in store.iter_skills():
        if s.id == skill_id or s.name == target_name:
            d = s.model_dump(mode="json")
            d["id"] = s.id
            return d
    raise HTTPException(status_code=404, detail="skill not found")
