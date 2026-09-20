# Phase 3 direct implementation and qualification

Implementation: `c37f4187447528646bbad16aae7a995d694951c0`.
Final qualification tree: `5282dcabc7157037d95144527a220f51e51c9803` (one additional
test-only initialization correction; identical production code).
Planning baseline: `437375c`. The primary agent implemented the bounded gap plan
directly, as requested. No executor, reviewer, checker, or other subagent was used;
there was no automatic review/fix cycle. This is not a GSD verifier report.

## Approved scope

The user's “proceed” followed the proposal to implement directly and the two
explicit compatibility checkpoints. It approved initialization before shared
workers (without implicit online schema upgrades) and the declared post-commit
derived-state outcomes, including cache close after the engine commit. This
authorization did not approve new lifecycle coordination or broader ACID promises.

[ADR 0001](adr/0001-topology-specific-storage-guarantees.md) remains controlling.
The product is a BlobStore with authenticated application catalog metadata and
cache instances using that storage engine. Separate namespaces are sufficient;
dual-role roots are not required. SQLite/local filesystem is a single-host
topology. Memory authority is single-process, with local test payloads—not a
crash-durable memory topology.

## Delivered changes

| Finding / planned work | Implementation and finite evidence |
| --- | --- |
| CR-01: unpublished memory abort | Remove retired mutation IDs from paging; public dry-run/apply no longer encounters a dangling mutation. |
| CR-02: shifting debt offsets | Monotonic mutation/debt identities and keyset paging; direct retirement and public three-debt resumed apply preserve live generations. |
| CR-03: permissive SQLite parameters | Point/list/query use the same strict raw-type/byte decoder. Invalid observed data produces typed corruption; canonical cache reads do not consume the damaged projection. |
| CR-04: bootstrap versus shared operation | `initialize()` completes startup before workers. Independent spawned workers reopen the initialized store. Unknown empty/obsolete catalogs are not silently adopted or migrated. |
| WR-01: SQLite error classification | BUSY/LOCKED remain typed contention; IOERR/FULL/READONLY/CANTOPEN/unknown failures remain operational backend errors; CORRUPT/NOTADB remain migration-required. Causes are preserved. |
| Deep engine interface | Immutable `BlobEntryInfo`, `put_entry`, `get_entry_info`, scoped verified `open_entry`, explicit stored-None presence, and exact conditional expectations. |
| Cache composition | Canonical reads, TTL, inventory, deletion and clear use the engine's authenticated information. Engine put owns cleanup; cache no longer runs projection-repair or deferred-cleanup orchestration. |
| Derived outcomes | Optional exports warn without revoking commits; explicitly requested ORM metadata failures raise typed committed-partial outcomes. Tests cover replacement and close after commit. |
| Product workflow | Mapping metadata query/update, independent object/cache roots, TTL and clear isolation, reopen, pinned reads, cleanup and replay are exercised. |

No new lock, FIFO, lease, sidecar, scheduler, or durable projection queue was added.
The private cache cleanup/repair orchestration and implicit internal SQLite schema
migration were removed. Existing close/key admission remains; the broad adapter
protocol is still transitional and must be narrowed before backend expansion.

The pure pre-promotion cache-signing transform is retained for stored-data
compatibility, but cannot change manifest identity, locator, digest, or lifecycle
fields. No external catalog writes occur inside that transform. Payload handlers
remain NumPy/Blosc2/pickle/etc.; this work does not introduce a new payload format.

Decision coverage is bounded to the declared topology: D-01–08 are exercised by
publication/integrity/crash tests; D-09–12 by exact delete, clear and owned-close
tests (post-engine cache close follows the approved partial-outcome rule);
D-13–16 by reconciliation and the new memory paging proofs; D-17–20 by existing
key/CAS tests and initialized workers, interpreted through ADR 0001 rather than
universal progress. D-21–22 and D-31–32 retain containment/trust checks and honest
Windows non-qualification. D-23–30 remain the sole SQLite/memory authority,
external immutable effects, bounded recovery and scheduler-retirement contracts.
D-33–34 and D-36 are the supported entry seam, metadata workflow and separate
cache/object roots. D-35's formerly pending checkpoint is approved by the user's
“proceed,” including maintenance-only schema changes; general migration is deferred.
This is an implementation/evidence mapping, not an independent decision audit.

## Test changes are contract changes, not weakened safety

Tests formerly corrupting a derived cache row to revoke canonical data now inject
faults into authoritative manifests or payloads. Signature/digest/containment tests
still require rejection before deserialization and preserve outside-root files.
The test-only manifest fixture can re-sign fields specifically to exercise inner
cache-signature and path validation beyond outer authentication; it changes no
production API. Failed canonical integrity checks preserve evidence, rather than
deleting via a second metadata catalog.

Old tests targeting removed repair hooks now assert stable public outcomes:
stale absence cannot delete a later winner; stale receipts cannot borrow a newer
projection token; failed optional export cannot gate cleanup; a post-commit
cleanup failure preserves the new authority and durable debt even if a projection
still describes the prior generation.

Concurrent startup fixtures and the distinct-key benchmark explicitly initialize
before workers. Both sequential and concurrent benchmark setup exclude this
initialization. The baseline and its thresholds were not relaxed. Historical
bootstrap test names remain for inventory compatibility, but their supported
schedule is now an initialized root.

## Development evidence, including failures

Development tests ran in `/private/tmp/cacheness-direct.KPvbvD`, a disposable
tracked archive plus owned edits. Original compatibility fixture sidecars were
never used as test inputs. The initial broad run reported 50 failures / 1,477 tests:
changed projection/startup assumptions, references to removed private hooks,
duplicate manifest authentication, query preselection through strict list decoding,
and two Git-history checks that cannot run in a plain archive. These were classified
and corrected before freezing the implementation; the failures are not erased by
later passes. The redundant authentication was removed and query adapters now
select exact authority key/locator pairs before decoding derived rows.

A later development full run reported 4 failures / 1,476 tests: one remaining
uninitialized integration fixture, a renamed test in the acceptance inventory,
and the same two archive/Git-history checks. The fixture and inventory were
corrected. Final qualification uses a real detached Git checkout. The focused
77-test setup/contract group and the 28-test new local-workflow module passed
before the code was frozen. Scoped Ruff and the Phase 3 delta gate passed.

## Exact-commit qualification

Checkout: `/private/tmp/cacheness-qualified.32MC8U`, detached at the implementation
commit above; `git status --porcelain` was empty before execution. Environment:
macOS 26.6.2 arm64, CPython 3.11.16 and 3.13.15. All commands use the frozen lockfile
and isolated all-extras/dev environments:

```text
uv run --isolated --python VERSION --all-extras --group dev --frozen COMMAND
```

The first qualification at `c37f418` passed the full suite (1,453 passed,
26 skipped), Python 3.13 focused suite (115 passed), gap suite (150 passed),
integrity suite (289 passed, 2 skipped), both lint gates, and the benchmark.
The independent progress run had 56 passes and one failed five-second test join:
`test_independent_write_write_race_has_one_cas_winner`. Inspection found that this
remaining fixture still started both writers on an uninitialized root. A startup
failure before its two-party publication barrier can strand the other participant;
the join failure alone did not prove a supported-topology runtime deadlock.

Commit `5282dca` adds `first.initialize()` before those workers start. It changes
no production code, timing bound, or CAS assertion. Qualification restarted at
that separate commit; the original failed gate remains part of this ledger.
No runtime edits or benchmark-baseline changes occurred during qualification.

The final full suite passed 1,453 tests with 26 skips. The independent progress
gate passed all 57 tests; the expanded Python 3.13 gate passed all 127 tests.
The final scoped Ruff and Phase 3 delta gates exited 0.

| Final gate (`5282dca`) | Result | Duration |
| --- | --- | --- |
| Python 3.11 full repository | 1,453 passed, 26 skipped; exit 0 | 50.659 s |
| Gap/public workflow | 150 passed; exit 0 | 5.607 s |
| Integrity/compatibility/recovery | 289 passed, 2 platform skips; exit 0 | 16.938 s |
| Progress/contention | 57 passed; exit 0 | 8.864 s |
| Python 3.13 focused + concurrency | 127 passed; exit 0 | 4.009 s |
| Phase 3 Ruff delta | No unmatched findings; exit 0 | — |
| Scoped Ruff, all changed Python files | All checks passed; exit 0 | — |
| Lifecycle baseline verification | Existing distribution envelopes passed; exit 0 | — |

Gate selections are the fixed commands in `03-VALIDATION.md` (groups 1–3 and 6–7).
Full-suite command: `pytest -q -o log_cli=false --tb=line --show-capture=no`.
The Python 3.13 command selected `test_phase3_local_workflows.py`,
`test_cached_query_meta.py`, `test_blob_store_read_contract.py`,
`test_unified_cache_lifecycle_authority.py`, and `test_blob_store_concurrency.py`.
XML reports are `/private/tmp/cacheness-final-{full,gaps,integrity,progress,313}.xml`.
The crash group collected and passed both
`test_live_exclusive_stream_crash_preserves_prior_generation_and_exact_debt[stream_copy]`
and `[file_fsync]` in `test_blob_store_atomic_lifecycle.py`.

Performance command: `python benchmarks/lifecycle_authority_benchmark.py
--verify-baseline benchmarks/lifecycle_authority_baseline.json`. It checks
authority/clear/busy-wait p99 and distinct-key overlap p05 against the unchanged
baseline. The CLI emits a verdict, not the fresh raw distributions. Its deliberate
busy-lock probes reported typed retryable timeouts and concurrent JSON projection
exports reported revision conflicts, both expected bounded outcomes rather than
integrity failures. No runtime bound was derived from those observed timings.

This closes the bounded 03-21 through 03-25 work directly. The earlier
`03-VERIFICATION.md` is retained historical evidence, not a fresh independent
verifier verdict and not an instruction to re-open these closed gaps. The user's
explicit process override replaces the planned fresh GSD checker/reviewer gate
with primary-agent implementation and the named exact-commit checks above.

## Protected evidence and remaining limits

Original dirty codebase maps, configuration, milestone lock, `.claude`, cache
state, and historical review files are not part of this commit. Original
`tests/fixtures/compat/sqlite-columns-v0314/metadata.sqlite3-{wal,shm}` identity,
size, timestamps and SHA-256 were unchanged before development and after final
qualification:

| File | inode | bytes | mtime ms / ctime ms | SHA-256 |
| --- | --- | --- | --- | --- |
| WAL | 113687243 | 0 | 1788692452094.166 / 1788692452094.2122 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| SHM | 113687244 | 32768 | 1788692452094.3164 / 1788692452094.5994 | `fd4c9fda9cd3f9ae7c962b0ddf37232294d55580e1aa165aa06129b8549389eb` |

Native Windows remains UNAVAILABLE/NOT_QUALIFIED, unchanged; Phase 999.1 owns
native qualification. Live PostgreSQL service tests, the full supported-Python
matrix, richer catalog schemas, broader backend composition, and general stored
data migration are not claimed complete. The finite tests are not proof of every
interleaving or universal contender success. See
[the operational guide](STORAGE_INITIALIZATION.md) for supported use and partial
failure handling. Do not restart an open-ended race review loop from this report.
