---
name: swap-fee-math
kind: product_domain
scope: product
product: forge
version: 1
confidence: 0.79
applies_to:
  files: ["programs/swap/src/math.rs", "programs/swap/src/lib.rs"]
  contexts: ["code-review", "performance"]
composes_with:
  - master
provenance:
  council_session: cs_eval_fixture_swap_fee_math
  validated_by: eval-fixture@anvay
  validated_at: 2026-05-18T00:00:00Z
  evidence_chunks: []
  adversary_critique: "rounding direction matters for protocol-vs-user — favor protocol"
  revision_count: 1
---

# Swap Fee Math

Forge uses a constant-product AMM with a protocol fee taken from the input side before the
invariant is applied. All math runs in `u128` to avoid silent overflow on extreme deltas.

## Invariants

1. **`x * y = k`** holds across every swap, where `x` and `y` are post-fee reserves.
2. **Fee is deducted from `amount_in`**, never from `amount_out`. This makes the invariant
   computation a single multiplication after fee.
3. **Rounding always favors the protocol.** When computing `amount_out`, integer division
   rounds down; when computing `protocol_fee`, it rounds up. Never the reverse.

## Reference implementation

The canonical sequence:

```
fee = ceil(amount_in * fee_bps / 10_000)
amount_in_after_fee = amount_in - fee
amount_out = floor(reserve_out * amount_in_after_fee / (reserve_in + amount_in_after_fee))
```

`fee_bps` is a `u16`, so the multiplication fits `u64`; but `reserve_out * amount_in_after_fee`
must be `u128` because both factors can reach `u64::MAX / 2`.

## Anti-patterns

- Using `f64` anywhere — even for displaying — risks rounding drift between front-end and
  on-chain math. Format integers as decimals downstream.
- Computing `amount_out` first and *then* deducting the fee from it. This is mathematically
  different and gives the protocol less than the intended `fee_bps`.

## Verification

Property test: for any `(reserve_in, reserve_out, amount_in)` with `amount_in > 0`,
`reserve_in * reserve_out <= (reserve_in + amount_in_after_fee) * (reserve_out - amount_out)`.
