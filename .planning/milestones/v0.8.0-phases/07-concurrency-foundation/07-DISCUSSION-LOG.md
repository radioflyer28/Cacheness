# Phase 7: Concurrency Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 07-concurrency-foundation
**Areas discussed:** Lock type fix scope

---

## Lock Type Fix Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal: just swap Lock→RLock | Change Lock→RLock at line 236 and run existing tests. Minimum viable fix. | |
| Consistent: swap + audit unprotected methods | Swap Lock→RLock AND fix iter_entry_summaries (currently has no lock) for consistency across all methods | ✓ |
| Thorough: swap + audit + review lock sharing | Above plus review whether BlobStore lock sharing creates hidden coupling | |

**User's choice:** Consistent: swap + audit unprotected methods
**Notes:** User chose the middle ground — fix the lock type AND ensure all public methods consistently acquire the lock. The BlobStore lock sharing pattern (core.py line 406) is explicitly out of scope.

---

## Gray Areas Not Selected

- **Re-entrancy testing** — How thorough the stress test should be (deferred to agent discretion)
- **Documentation scope** — Whether to document threading model in docs/ (deferred to agent discretion)

## Agent's Discretion

- Re-entrancy stress test design
- iter_entry_summaries lock granularity (whole generator vs per-yield)
- Code-level documentation approach

## Deferred Ideas

None
