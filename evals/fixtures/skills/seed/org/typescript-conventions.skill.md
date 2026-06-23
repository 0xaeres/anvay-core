---
name: typescript-conventions
kind: language
scope: org
version: 1
confidence: 0.88
quality_score: 0.88
external_sources:
  - https://www.typescriptlang.org/docs/handbook/declaration-files/do-s-and-don-ts.html
  - https://google.github.io/styleguide/tsguide.html
ratified_by: org-admin@anvay
ratified_at: 2026-05-18T00:00:00Z
applies_to:
  files: ["**/*.ts", "**/*.tsx"]
  contexts: ["code-review"]
composes_with: []
---

# TypeScript Conventions

These rules apply to every product written in TypeScript. Product-specific overlays can
relax individual rules with written justification.

## Type system

1. **`strict: true` in tsconfig.** Never disable. Use targeted `// @ts-expect-error` with a
   comment if needed, never `@ts-ignore`.

2. **No `any` in checked-in code.** Prefer `unknown` + a narrowing guard. `any` is allowed
   only in `.d.ts` shims for untyped libraries, and even there `unknown` is better.

3. **Prefer `type` for unions and `interface` for object shapes.** Mixing produces inconsistent
   declaration-merging behaviour.

4. **Return types on exported functions.** Inference is fine for internal helpers; explicit
   return types on the public surface prevent accidental signature drift.

## Imports & modules

1. **Path aliases via `paths` in tsconfig.** No `../../../` traversals across more than
   two levels. Aliases stay stable when files move.

2. **Re-export sparingly.** A barrel file (`index.ts` re-exporting many modules) defeats
   tree-shaking and slows the typechecker on large repos.

## React / TSX specifics

1. **Function components only.** Class components are legacy.

2. **Props types live with the component.** `interface Props { ... }` directly above the
   component definition, not in a separate `types.ts` unless shared.

3. **No prop-drilling beyond two levels.** Use composition (`children`) or context.

## Anti-patterns

- `as` casts on values you don't control — use type guards.
- Enum types with `string` values that overlap (`enum X { A = "a", B = "a" }`) — TypeScript
  doesn't catch this and runtime equality lies.
