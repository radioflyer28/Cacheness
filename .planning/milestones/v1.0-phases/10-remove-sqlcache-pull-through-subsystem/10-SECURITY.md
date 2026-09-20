---
phase: 10
slug: remove-sqlcache-pull-through-subsystem
status: verified
threats_open: 0
asvs_level: 1
created: 2026-09-17
---

# Phase 10 — Security

> Post-execution verification of the threat registers authored in Plans 10-01 through 10-09.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Source and package artifacts | Deleted APIs must not survive through exports, hooks, wheel members, or installed metadata. | Python modules, exports, wheel paths, distribution metadata |
| Repository and caller databases | Removal work must not discover, mutate, migrate, or claim authority over caller-owned tables. | API signatures and caller-owned SQL state |
| Current guidance and historical evidence | Current product claims must be exact without erasing dated or completed planning history. | Documentation, examples, verifier inventories |
| Manifest, lock, wheel, and isolated install | Dependency and package evidence must refer to one reviewed artifact and preserve retained integrations. | Dependency declarations, lock entries, installed requirements |

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-10-01 | Spoofing | Removed import/API surface | high | mitigate | Physical deletion plus isolated export, attribute, spec, hook, source, and wheel absence contracts | closed |
| T-10-02 | Tampering | Wheel contents and metadata | high | mitigate | One digest-bound `WheelArtifact` drives ZIP inspection, source-free install, metadata checks, and round trips | closed |
| T-10-03 | Repudiation | Current reference scan and history | medium | mitigate | Explicit current roots and exact owners exclude dated/planning history; Phase 10 changed no prior phase artifact | closed |
| T-10-04 | Tampering | Caller-owned SQL state | high | mitigate | No cleanup/migration entry point; table-shaped maintenance inputs fail fast; docs state tables are untouched | closed |
| T-10-05 | Repudiation | Historical verifier repair | medium | mitigate | Executable manifests were updated while completed historical artifacts remained unchanged | closed |
| T-10-06 | Tampering | Fixed verifier inventories | medium | mitigate | Literal normalized matrices and mirrored exact-set tests execute in the focused suite | closed |
| T-10-07 | Denial of service | Test-suite collection | medium | mitigate | Obsolete nodes were inverted before deletion; full frozen collection and non-live suite pass | closed |
| T-10-08 | Spoofing | Canonical guidance | high | mitigate | Exact-owner documentation tests require explicit no-replacement and caller-table boundary wording | closed |
| T-10-09 | Denial of service | Retained package surface | high | mitigate | Exact retained export/error contracts and full regressions pass after direct deletion | closed |
| T-10-10 | Denial of service | Retained SQL authority and dataframe support | high | mitigate | Retained dependency assertions, handler tests, PostgreSQL contracts, and all-extras regressions pass | closed |
| T-10-11 | Spoofing | Obsolete public guidance | high | mitigate | Dedicated docs/examples are physically absent and exact path/reference contracts pass | closed |
| T-10-12 | Spoofing | Current architecture maps | medium | mitigate | Final maps were cross-checked against source, manifest, ADR 0001, and acceptance tests | closed |
| T-10-SC | Tampering | Dependency resolution | high | mitigate | No new package was introduced; frozen tests, `uv lock --check`, and reviewed lock/metadata evidence pass | closed |

*Status: open · closed · open — below high threshold (non-blocking)*

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-09-17 | 13 | 13 | 0 | Codex orchestrator, ASVS L1 artifact verification |

The threat register was authored at plan time. All registered threats have concrete mitigation evidence in the final verifier and passing acceptance contracts, so the ASVS L1 short-circuit applies and no deeper boundary-placement audit is required.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-09-17
