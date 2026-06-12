---
id: SEED-002
status: dormant
planted: 2026-06-12
planted_during: v0.11.0 complete / pre-v0.12 planning
trigger_when: next major version (breaking-changes window)
scope: small
---

# SEED-002: Flip allow_unsigned_entries default to False when a signing key exists

## Why This Matters

With the current default (`allow_unsigned_entries=True`), an attacker with metadata write access simply strips the signature field and verification is skipped entirely — then the pickle handler deserializes their payload. Signing only has teeth with `allow_unsigned_entries=False`. Code review finding **S1** (`docs/CODE_REVIEW_FINDINGS.md` §3). TASK-13 adds docs + `minimum_signature_version` but deliberately does NOT flip the default (breaking change).

## When to Surface

**Trigger:** next major version (breaking-changes window).

## Scope Estimate

**Small** — change default (proposal: key exists ⇒ entries must be signed), migration note, update tests that rely on unsigned entries.

## Breadcrumbs

- docs/CODE_REVIEW_FINDINGS.md S1
- docs/CODE_REVIEW_ACTIONS.md TASK-13 (prerequisite — lands first)
- src/cacheness/config.py:400 (`allow_unsigned_entries`)
- src/cacheness/_verification_mixin.py (`_verify_entry`)

## Notes

Pair with the pickle-deserialization risk already in CONCERNS.md — this default is the main mitigation lever.
