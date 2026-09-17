---
phase: 10-remove-sqlcache-pull-through-subsystem
reviewed: 2026-09-17T18:14:25Z
depth: standard
files_reviewed: 24
files_reviewed_list:
  - AGENTS.md
  - docs/API_REFERENCE.md
  - docs/CROSS_PLATFORM_GUIDE.md
  - docs/PANDAS_API_AUDIT.md
  - docs/README.md
  - docs/STORAGE_MIGRATION.md
  - pyproject.toml
  - src/cacheness/__init__.py
  - src/cacheness/error_handling.py
  - tests/packaging/test_wheel_matrix.py
  - tests/test_full_suite_environment.py
  - tests/test_phase071_contract_verifier.py
  - tests/test_phase10_sqlcache_removal.py
  - tests/test_phase1_quality_gates.py
  - tests/test_phase4_cutover_verifier.py
  - tests/test_phase6_contract_verifier.py
  - tests/test_phase6_public_api_contract.py
  - tests/test_phase6_suite_isolation.py
  - tests/test_phase9_documentation.py
  - tests/test_public_api_contract.py
  - tools/run_phase8_packaging.py
  - tools/verify_phase071_contracts.py
  - tools/verify_phase4_cutover.py
  - tools/verify_phase6_contracts.py
findings:
  critical: 1
  warning: 3
  info: 0
  total: 4
status: issues_found
---

# Phase 10: Code Review Report

**Reviewed:** 2026-09-17T18:14:25Z
**Depth:** standard
**Files Reviewed:** 24
**Status:** issues_found

## Narrative Findings (AI reviewer)

### Summary

The direct SqlCache cut is structurally sound: the implementation module,
top-level exports, dedicated assets, DuckDB metadata, and exclusive error
reasons are gone, and no storage-lifecycle coordinator or caller-database
mutation path was added. The submitted current-facing platform guide still
makes release claims that directly contradict the canonical qualification
boundary, however. Three additional defects make the negative acceptance
contracts less fail-closed than their names and the locked Phase 10 decisions
require.

### Critical Issues

#### CR-01: Current platform guide claims unsupported universal qualification

**Classification:** BLOCKER

**File:** `docs/CROSS_PLATFORM_GUIDE.md:5-10`

**Issue:** The guide says Cacheness works identically on Windows, Linux, and
macOS, is pure Python, and has no binary dependencies. It repeats at lines
432-437 that the library is fully cross-platform with identical behavior and
475/475 tests passing on every platform. These are current product claims, not
dated history. They contradict `docs/RELEASE_QUALIFICATION.md`, which says Linux
is the full-matrix target, macOS is boundary smoke only, and Windows is
`UNAVAILABLE` / `NOT_QUALIFIED`. They are also factually incompatible with the
mandatory NumPy, cryptography, and obstore distributions in `pyproject.toml`,
which use platform-specific binary wheels. A user can therefore treat an
explicitly unqualified platform as supported based on a guide Phase 10 kept in
the current documentation surface.

**Fix:** Rewrite the overview, wheel, expected-results, and conclusion claims to
defer to `RELEASE_QUALIFICATION.md`: state the actual Python/runtime boundary,
identify Linux full qualification and macOS boundary smoke, mark native Windows
unqualified, and avoid claiming that dependencies require no native wheels.
Remove the unsubstantiated 475/475 table unless it is regenerated from current,
platform-bound evidence.

### Warnings

#### WR-01: Reference allowlist approves an entire file after one valid note

**Classification:** WARNING

**File:** `tests/test_phase10_sqlcache_removal.py:122-129`

**Issue:** `_allowed_reference()` treats every non-document allowlisted file as
valid without inspecting the occurrence, and treats an allowlisted document as
valid whenever the canonical cutover sentence appears anywhere in it. As a
result, `docs/API_REFERENCE.md` can contain the required negative note plus new
positive guidance such as `Use SqlCacheAdapter with duckdb-engine`, and
`test_current_facing_references_match_allowlist` still passes. The same
path-wide exemption permits executable or positive compatibility residue in
allowlisted tests and tools. This violates D-14's requirement for a narrow,
purpose-specific allowlist.

**Fix:** Make the allowlist occurrence-specific. For the three documents,
permit retired markers only inside one exact bounded cutover block and reject
all occurrences outside it. For tests and tools, freeze exact expected marker
counts or parse the AST so only literal negative inventories/assertions are
accepted. Add a hostile fixture containing both the valid note and a positive
SqlCache/DuckDB instruction and require the scanner to reject it.

#### WR-02: Wheel-member proof rejects only one filename spelling

**Classification:** WARNING

**File:** `tools/run_phase8_packaging.py:172-177`

**Issue:** `RETIRED_WHEEL_MEMBERS` contains only
`cacheness/sql_cache.py`, and `_assert_retired_wheel_members_are_absent()` uses
an exact set intersection. A wheel can therefore contain
`cacheness/sql_cache.pyi`, `cacheness/sql_cache/__init__.py`, a compiled
extension, or another member under that retired module path without this
archive check noticing. The runtime probe catches importable variants, but it
does not catch a non-importable type-stub tombstone; such a wheel would pass
despite D-13 requiring no retired wheel member and D-01 forbidding compatibility
aliases.

**Fix:** Reject every normalized wheel member whose path is the retired module
stem or begins with `cacheness/sql_cache/`, including `.py`, `.pyi`, extension,
and package forms. Add parameterized synthetic-wheel tests for each spelling,
including a non-importable `.pyi` member.

#### WR-03: Caller-table test does not inspect database tooling

**Classification:** WARNING

**File:** `tests/test_phase10_sqlcache_removal.py:254-264`

**Issue:** `test_phase10_has_no_caller_table_tooling` only searches production
source text for retired product-name markers. It does not inspect table
discovery, DDL, migration, export, or cleanup behavior. A generic helper that
drops or migrates caller tables while avoiding the strings `SqlCache`,
`sql_cache`, `range-aware SQL`, and DuckDB would satisfy this test. The current
Phase 10 source diff does not add such a helper, so caller databases remain
untouched in this submission, but the advertised regression test is a false
positive and does not enforce D-03.

**Fix:** Either rename this assertion to the narrower retired-reference check it
actually performs, or replace it with a structural public-boundary contract:
freeze the supported maintenance/export entry points, assert they accept
BlobStore/topology inputs rather than arbitrary SQL tables or engines, and
assert no project script/command exposes caller-table discovery, migration, or
cleanup. Keep the documentation assertion as a separate contract rather than
using product-name absence as a proxy for runtime behavior.

---

_Reviewed: 2026-09-17T18:14:25Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
