---
id: SEED-001
status: dormant
planted: 2026-06-12
planted_during: v0.11.0 complete / pre-v0.12 planning
trigger_when: next milestone touching storage mode, or any API-design phase
scope: small
---

# SEED-001: Storage-mode API hardening — refuse or warn on eviction APIs

## Why This Matters

Storage mode's contract is durability, but `cleanup_expired(ttl_seconds=...)`, size-limit eviction, and `clear_all()` are still callable on a `storage_mode=True` instance and **will delete durable entries** if invoked explicitly. `cleanup_expired()` only no-ops today because TTL happens to be None. Flagged as the open design question in `docs/CODE_REVIEW_FINDINGS.md` §1b.

## When to Surface

**Trigger:** next milestone touching storage mode, or any API-design phase.

## Scope Estimate

**Small** — decide strictness (raise vs. loud warning vs. explicit-call-means-consent), implement guard + tests in `core.py` public methods.

## Breadcrumbs

- docs/CODE_REVIEW_FINDINGS.md §1b (open design question)
- docs/CODE_REVIEW_ACTIONS.md (out-of-scope list, item 1)
- src/cacheness/_storage_mode_mixin.py
- config.py storage-mode block (~line 780)

## Notes

Recommend at minimum a loud warning log. Decision needed from owner on strictness.
