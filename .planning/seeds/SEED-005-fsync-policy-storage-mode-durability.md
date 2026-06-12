---
id: SEED-005
status: dormant
planted: 2026-06-12
planted_during: v0.11.0 complete / pre-v0.12 planning
trigger_when: storage-mode durability milestone, or when documenting transaction guarantees
scope: medium
---

# SEED-005: fsync policy — opt-in durable writes for storage mode

## Why This Matters

Nothing fsyncs: JSON saves, blob writes, and intent files rely on atomic rename only — on power loss, "committed" entries can vanish or be empty. SQLite (WAL + synchronous=NORMAL) is the only durable component. Acceptable for a cache; **not obviously acceptable for storage mode**, whose contract is durability. Code review finding **R16**. Deferred because adding fsync is a measurable perf hit and needs discussion.

## When to Surface

**Trigger:** storage-mode durability milestone, or when updating TRANSACTION_GUARANTEES.md.

## Scope Estimate

**Medium** — design `fsync_on_write` (or storage-mode-implied) option covering blob writes, JSON saves, and intent files; benchmark the cost; document the contract either way.

## Breadcrumbs

- docs/CODE_REVIEW_FINDINGS.md R16 + §1b
- docs/TRANSACTION_GUARANTEES.md (contract documentation home)
- src/cacheness/metadata/json_backend.py (`_save_to_disk`)
- src/cacheness/storage/backends/blob_backends.py (`write_blob`)
- src/cacheness/write_intent.py

## Notes

Documentation-only resolution (explicit "no power-loss durability" statement) is acceptable as a first step; the config knob can come later.
