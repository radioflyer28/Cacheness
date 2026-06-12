---
id: SEED-004
status: dormant
planted: 2026-06-12
planted_during: v0.11.0 complete / pre-v0.12 planning
trigger_when: any milestone touching signing/verification code, or before tiered-cache work begins
scope: medium
---

# SEED-004: Unify the two signing schemes (UnifiedCache vs BlobStore)

## Why This Matters

`UnifiedCache` signs via `_extract_signable_fields()` (normalized `created_at`, fixed field superset) while `BlobStore.put()` signs a flattened `{**entry_data, **custom_metadata}` — two schemes sharing one signer and one metadata backend. Entries written by one API and read by the other risk spurious signature failures; duplication invites drift. Code review finding **U4**. Deferred because unification touches signature compatibility of existing caches and needs a migration plan (re-sign on read? version bump? both schemes accepted during transition?).

## When to Surface

**Trigger:** any milestone touching signing/verification, or before tiered-cache work (cross-tier promotion requires one scheme).

## Scope Estimate

**Medium** — move `_extract_signable_fields()` to a shared module, migrate BlobStore, plus a compatibility/migration strategy for existing signed entries.

## Breadcrumbs

- docs/CODE_REVIEW_FINDINGS.md U4
- src/cacheness/_verification_mixin.py (`_extract_signable_fields`)
- src/cacheness/storage/blob_store.py (`put` signing block)
- .planning/todos/pending/2026-04-03-tiered-pull-through-cache.md (prerequisite note)

## Notes

Coordinate with TASK-13 (minimum_signature_version) — a signature-version bump could carry the unified field list.
