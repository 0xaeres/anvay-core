---
name: pda-seed-validation
kind: product_domain
scope: product
product: forge
version: 1
confidence: 0.87
applies_to:
  files: ["**/*.rs"]
  contexts: ["security-audit", "code-review"]
composes_with:
  - master
  - org/owasp-input-validation
provenance:
  council_session: cs_eval_fixture_pda_seed_validation
  validated_by: eval-fixture@nexus
  validated_at: 2026-05-18T00:00:00Z
  evidence_chunks: []
  adversary_critique: "edge case on PDA bump seeds when client supplies an off-curve seed"
  revision_count: 1
---

# PDA Seed Validation

Solana Program Derived Addresses are derived from a tuple of seeds plus a bump byte. Any
program that accepts a client-supplied `bump` must re-derive the address and compare. Trusting
the client's bump enables address-substitution attacks where an attacker passes a different
account that happens to share the seed prefix.

## Rules

1. **Always re-derive in the handler.** Use `Pubkey::find_program_address(seeds, program_id)`
   and assert equality with the received account key. Anchor's `#[account(seeds = [...], bump)]`
   does this automatically — prefer it over manual derivation.

2. **Off-curve check is implicit.** `find_program_address` skips on-curve points by
   incrementing the bump. Manual derivation with `create_program_address` does **not** — only
   use that when you already know the bump and want to verify a specific one.

3. **Document seed schema in a single source of truth.** A `seeds!` macro or a const-fn that
   returns the seed tuple prevents drift between handlers and tests.

4. **Validate sub-seeds where they encode authority.** A `mint` seed should match the account's
   declared `mint`; otherwise the PDA is for a different vault.

## Anti-patterns

- Accepting a `bump: u8` parameter without re-derivation. Always derive.
- Mixing canonical bumps with non-canonical bumps in storage. Pick one (canonical) and
  enforce it at PDA creation.
- Storing the derived address as the seed instead of the inputs that produced it.

## Tools to reach for

- `query_code_context(symbol="find_program_address")` to audit every site.
- `hybrid_search_corpus("pda derivation bump")` for cross-source references (ADRs, audits).
