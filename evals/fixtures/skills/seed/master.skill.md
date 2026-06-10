---
name: forge
kind: master
scope: product
product: forge
version: 1
confidence: 0.82
applies_to:
  files: ["**/*"]
  contexts: ["onboarding", "code-review", "security-audit"]
composes_with: []
provenance:
  council_session: cs_eval_fixture_master_forge
  validated_by: eval-fixture@nexus
  validated_at: 2026-05-18T00:00:00Z
  evidence_chunks: []
  adversary_critique: null
  revision_count: 0
---

# Forge Product Context

Golden eval fixture for Forge, a Solana constant-product AMM used to test skill retrieval.
Runtime bootstrap stays empty; real product skills are written only after human approval.

## How agents should work here

1. **Read the relevant domain skill first.** Most tasks touch a specific pattern or
   subsystem — each has its own product-domain skill. Call `find_skills` with the
   capability before reading code.

2. **Cite file:line for every claim.** Never describe code behaviour without an anchor
   `[file: src/module.ts:42]`. Reviewers reject uncited assertions.

3. **Prefer composition over duplication.** Non-master skills compose back to this master
   via `composes_with`. Use `find_skills` to check if a skill already covers the pattern
   before drafting a new one.

## Getting started

Run the onboarding wizard at `/onboarding` to connect a real codebase, then start a
council session. Approved proposals write the product's real master skill.
