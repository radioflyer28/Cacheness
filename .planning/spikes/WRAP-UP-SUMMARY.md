# Spike Wrap-Up Summary

**Date:** 2026-09-10
**Spikes processed:** 5
**Feature areas:** Handler integration; lifecycle and recovery; backend mechanics
**Skill output:** `./.codex/skills/spike-findings-cacheness/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | handler-boundary | standard | VALIDATED | Handler integration |
| 002 | immutable-publication | standard | VALIDATED | Lifecycle and recovery |
| 003 | authority-boundary | standard | VALIDATED | Lifecycle and recovery |
| 004 | containment-deletion | standard | PARTIAL | Lifecycle and recovery; backend mechanics |
| 005 | streaming-replacement | standard | PARTIAL | Backend mechanics |

## Key Findings

Obstore 0.11.1 is viable below a Cacheness-owned payload-participant seam.
Existing path-based built-in and custom handlers can remain unchanged when
their single-file outputs pass through private guarded staging and reads use
private suffix-preserving snapshots.

Create-if-absent produced one complete winner under same-locator contention on
LocalStore, MemoryStore, and mocked S3Store. Deterministic locator, digest, and
size were sufficient to recover an acknowledgement lost after publication.
These are payload-effect guarantees only: SQLite/PostgreSQL authority must
continue to own lifecycle intent, visibility, promotion, cleanup debt, and
reconciliation under ADR 0001.

Containment and exact deletion work when Cacheness validates the namespace and
never reuses an immutable generation locator. Obstore has no conditional
version/e-tag delete.

The unresolved implementation decision is large conditional publication.
Direct `put(mode="create")` buffers at payload scale. Bounded-memory S3
publication requires a temporary multipart upload plus configured conditional
multipart copy, which introduces temporary-object and abandoned-upload cleanup
obligations. Obstore can replace substantial backend mechanics, but it does not
erase those lifecycle tradeoffs.

