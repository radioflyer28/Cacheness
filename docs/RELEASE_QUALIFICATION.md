# Release qualification

Phase 8 closes a deterministic local-readiness boundary. It does not publish a
release or qualify real PostgreSQL/Amazon-S3, controlled-Linux performance,
Windows, or an unexecuted platform matrix. Those boundaries remain separate:
`BACK-05` and immutable publication are deferred to
[SEED-007](../.planning/seeds/SEED-007-qualify-real-postgresql-s3-and-publish-release.md),
and `QUAL-06` remains deferred to SEED-006. A local pass, mock, skipped test,
configuration preflight, old artifact, or scheduled diagnostic cannot substitute.

Every qualifying artifact binds a candidate SHA and a reviewed source digest.
The release collector accepts only the requested 40-character lowercase Git
revision, recomputes the relevant source identity, records the workflow run ID
and fixed artifact name, and rejects an unavailable, malformed, stale, dirty,
or wrong-class input.

## Evidence matrix

| Evidence class | Producer | Blocking rule and terminal states | Identity and retention |
| --- | --- | --- | --- |
| Deterministic/local | `tools/run_phase8_local_gates.py deterministic`; `tools/verify_phase071_contracts.py --all` | `PASS` requires the fixed contracts with no skipped or incomplete result. `NOT_QUALIFIED` blocks a release; this class does not supply live, timing, or platform evidence. | Envelope contains its exact revision and source digest. Release-candidate artifact is retained for the release lifetime. |
| Packaging | `tools/run_phase8_packaging.py` through `tools/run_phase8_local_gates.py packaging` | A clean wheel and every advertised optional group must complete its literal public probe. An incompatible TensorFlow row is `UNAVAILABLE`, never a skip-based pass. | Fresh isolated environments and wheel SHA-256 are recorded with the candidate SHA/source digest. Qualifying artifact is retained for the release lifetime. |
| Platform | `tools/run_phase8_platform_gates.py` through `tools/run_phase8_local_gates.py platform` | Every required stable Linux and macOS boundary row must be `PASS`; prerelease is advisory only. Missing rows, a wrong host/interpreter identity, or `UNAVAILABLE` blocks the relevant support claim. | Each row records actual and expected OS/Python/profile plus revision/source digest. Required rows are retained with the release evidence. |
| Coverage/quality | `tools/verify_phase8_coverage.py` through `tools/run_phase8_local_gates.py coverage` | The named selectors and repository/critical statement and branch floors must pass. Direct scoped Ruff lint and formatting must pass; global legacy lint debt is not a release scope. | Coverage report/XML and canonical envelope bind the reviewed candidate SHA/source digest. Qualifying artifact is retained for the release lifetime. |
| Structural | `tools/run_phase8_scale_gates.py` through `tools/run_phase8_local_gates.py structural` | Fixed authority/participant call formulas and isolated RSS observations must pass. A missing child result or invalid units is `NOT_QUALIFIED`; elapsed time is not a structural result. | Envelope contains fixed-scale counters, normalized RSS facts, revision, and source digest. Qualifying artifact is retained for the release lifetime. |
| Controlled performance | Retained `benchmarks/phase8_benchmarks.py` and `.github/workflows/performance.yml` machinery | `DEFERRED` and `NOT_QUALIFIED` for this milestone under `QUAL-06`; it is not a current release prerequisite and cannot become a `PASS` through an unavailable runner or diagnostic result. | No controlled-performance artifact is collected or published for the current release. The retained harness, preflight, workload inventory, and baseline contract are reserved for [SEED-006](../.planning/seeds/SEED-006-qualify-controlled-linux-performance.md). |
| Live service | Preserved `tools/run_phase8_qualification.py` and `.github/workflows/live_qualification.yml` | `DEFERRED` and `NOT_QUALIFIED` under BACK-05/SEED-007. Future protected execution still requires real PostgreSQL and Amazon S3 to emit sanitized `QUALIFIED` and `CLEAN` evidence; no substitute passes. | No live evidence is attached or claimed by the local-readiness milestone. Existing exact SHA/source, cleanup, and run-ID rules remain mandatory for SEED-007. |
| Publication | Preserved `tools/verify_phase8_release.py` controller | `DEFERRED` and `NOT_PUBLISHED` under SEED-007. A local readiness report is not a draft, tag, GitHub release, or immutable-publication proof. | Existing exact tag/state/asset/digest verification remains mandatory when SEED-007 runs. |

`PASS` means the declared current class completed for its stated identity.
`UNAVAILABLE` means an external prerequisite or unsupported environment was absent.
`NOT_QUALIFIED` means evidence was incomplete, invalid, dirty, failed, deferred,
or did not meet the class contract. `NOT_PUBLISHED` means no verified immutable
GitHub release exists. `DEFERRED` is a closed nonclaim, not a pass: controlled
performance points to [SEED-006](../.planning/seeds/SEED-006-qualify-controlled-linux-performance.md),
while BACK-05 and publication point to SEED-007. None may be relabeled as success.

## Supported runtime and package scope

- Linux runs the full stable core matrix on Python 3.11, 3.12, 3.13, and 3.14.
- macOS runs the public topology boundary smoke on Python 3.11 and 3.14. It
  complements Linux rather than replacing the full Linux matrix.
- TensorFlow remains a retained optional handler only on compatible stable
  Linux rows: Python 3.11 and 3.12. It is not inferred from a core or
  dataframe result.
- Python 3.15 is a continue-on-error advisory row. Its result informs future
  support work but does not qualify or disqualify the stable matrix.
- Windows remains `UNAVAILABLE` / `NOT_QUALIFIED` until Phase 999.1 records
  eligible native Windows evidence. Linux or macOS output cannot substitute.

The base wheel proves guaranteed public imports plus generic-object and retained
NumPy behavior. Dataframe extras continue to qualify their retained Parquet
handlers. This is compatibility evidence for existing formats, not a handler
redesign.

## Local-readiness boundary

The Phase 8 local-readiness command must prove the fixed deterministic
integrity/recovery suite, a clean base-wheel import and public round trip,
coverage and scoped Ruff ratchets, and structural memory/backend-call bounds for
one exact source identity. It records the observed current-host scope and the
SEED-006/SEED-007 nonclaims. It does not contact PostgreSQL/AWS, dispatch a
workflow, create a GitHub draft, publish a release, or convert an unavailable
optional/platform row into support evidence.

## Protected service and performance boundaries

Only `workflow_dispatch` release-candidate jobs may receive the protected
PostgreSQL and Amazon S3 configuration. They require one exact candidate SHA,
detached checkout equality, frozen `uv.lock` resolution, the real frozen suite,
standard AWS credentials/IAM and bucket policy, an explicit bucket and region,
and exact bounded cleanup. Production endpoint overrides and an
`ExpectedBucketOwner` claim remain unsupported.

That protected path is preserved for SEED-007 and is not executed by Phase 8's
local closure. Until it runs successfully, PostgreSQL/Amazon-S3 is a candidate
topology, not a release-qualified support claim, and publication is
`NOT_PUBLISHED`.

Scheduled service runs are diagnostic only: they detect drift but cannot be
attached as release qualification. Remote PostgreSQL/S3 latency is also
diagnostic only. It cannot replace controlled-Linux timing evidence or change
the outcome of the live integrity/recovery suite.

Controlled regression review remains available only for future work on the
approved Linux runner. It is deferred from the current release under `QUAL-06` /
[SEED-006](../.planning/seeds/SEED-006-qualify-controlled-linux-performance.md):
controlled performance is `DEFERRED` and `NOT_QUALIFIED`, and its diagnostics are
not collected or published as current release assets. macOS timings are diagnostic
and do not establish Linux equivalence or a cross-platform budget. When the seed
is promoted, its Linux comparison must remain a performance measurement rather
than a runtime deadline, universal timeout, or stronger progress promise.

Structural evidence independently records bounded page/work, backend-call, and
peak-memory behavior. It does not measure wall-clock latency, promise
bounded-memory S3 streaming, or permit an optimization to add lifecycle locks,
queues, leases, sidecars, or a second authority.

## Storage guarantees and nonclaims

The topology vocabulary and limits in ADR 0001 are release constraints. SQLite
or PostgreSQL authority transactions do not create cross-resource ACID with a
filesystem or S3 payload participant. Immutable payload publication and
authority-owned reconciliation provide crash-consistent recovery at that
boundary, not one distributed transaction.

The project does not claim that every concurrent contender succeeds within a
fixed time, that all topologies share the same progress guarantee, or that a
benchmark result is a runtime deadline. A documented typed retryable contention
result may preserve integrity, recovery, and the topology's stated progress
contract. It does not authorize another coordination mechanism.

An unattributed pre-checkpoint orphan remains invisible and unadopted. Cacheness
does not claim exact reclamation of an unattributed pre-checkpoint orphan; only
authority-attributed immutable generations and durable cleanup debt have the
declared exact recovery path.

## Migration-fixture boundary

Offline migration and rebuild are stopped-worker maintenance operations. Release
qualification retains fixtures for the current and immediately previous released
layout; older layouts advance only through declared offline steps. Ordinary store
open never upgrades implicitly, and historical pre-production layouts do not
silently acquire compatibility through a passing Phase 8 workflow.
