---
name: owasp-input-validation
kind: security
scope: org
version: 1
confidence: 0.91
quality_score: 0.91
external_sources:
  - https://owasp.org/www-project-application-security-verification-standard/
  - https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
ratified_by: org-admin@anvay
ratified_at: 2026-05-18T00:00:00Z
applies_to:
  files: ["**/*"]
  contexts: ["security-audit", "code-review"]
composes_with: []
---

# OWASP Input Validation

Every byte that crosses a trust boundary is hostile until proven otherwise. This applies
equally to HTTP request bodies, CLI flags, environment variables, file contents, and on-chain
account data.

## Rules

1. **Allow-list, never deny-list.** Validate against a known-good shape (length, charset,
   range, enum). Denying specific bad characters drifts as inputs evolve.

2. **Validate at the boundary, sanitise at the sink.** Validation rejects; sanitisation
   transforms. Don't conflate them. A SQL parameter binder sanitises; a request schema
   validates.

3. **Reject before deserialising untrusted blobs.** Length-limit and content-type-check
   raw bytes before handing them to a parser. Parsers are not safe input filters.

4. **Use schemas, not ad-hoc checks.** A Pydantic / Zod / Anchor `#[account]` schema is
   reviewable and testable; scattered `if x.length > 100` checks are not.

## Specific anti-patterns

- Trusting a length field inside the same blob that supplies the payload.
- Using regex `.match()` without anchors (`^...$`) — matches a *substring*, not the whole.
- Catching parser exceptions silently. Either reject the input or surface a structured
  error to the caller.
