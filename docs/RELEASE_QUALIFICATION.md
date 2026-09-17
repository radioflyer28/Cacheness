# Release qualification

Cacheness is a **local-ready development version**. The checked-out project is
useful for the declared local workflows, but it has not been published as an
immutable release and its API may change before the first supported release.
This is the sole detailed owner of evidence status, topology, platform,
payload-bound, and performance claims. Task guides link here instead of
restating these claims.

The retained Phase 8 local-readiness record is
[`08-LOCAL-READINESS.json`](../.planning/phases/08-production-gates-and-performance-stabilization/08-LOCAL-READINESS.json).
It records `LOCAL_READY` for one exact revision and source digest only. A
later checkout, mock, skipped test, preflight, old artifact, or scheduled
diagnostic is not a substitute for the evidence class it did not run.

## Evidence matrix

| Evidence class | Existing evidence and scope | Current status and nonclaim |
| --- | --- | --- |
| Deterministic/local | `tools/run_phase8_local_gates.py deterministic` and `tools/verify_phase071_contracts.py --all` supplied the retained local record. | `PASS` applies only to the recorded revision/source digest. It is not live-service, platform, timing, or publication evidence. |
| Packaging | `tools/run_phase8_packaging.py` supplies frozen wheel/public-import probes. | The retained base-wheel evidence is `PASS` for its recorded identity. Optional capability probes remain their own evidence. |
| Platform | `tools/run_phase8_platform_gates.py` retains the full stable Linux matrix and macOS boundary-smoke contract. | Linux is the full matrix target; macOS is boundary smoke only. The local Darwin observation is not Linux equivalence. Windows remains `UNAVAILABLE` / `NOT_QUALIFIED`. |
| Coverage/quality | `tools/verify_phase8_coverage.py` supplies scoped coverage, Ruff, and formatting evidence. | `PASS` is tied to the reviewed source identity; global legacy lint debt is not silently relabeled as a release result. |
| Structural | `tools/run_phase8_scale_gates.py` records authority/participant calls and isolated RSS facts. | `PASS` is structural only. It does not promise latency, remote streaming bounds, or turn a benchmark into a runtime deadline. |
| Controlled performance | `benchmarks/phase8_benchmarks.py` and `.github/workflows/performance.yml` are retained for controlled measurement. | controlled Linux performance remains `DEFERRED` / `NOT_QUALIFIED` under `QUAL-06` and [SEED-006](../.planning/seeds/SEED-006-qualify-controlled-linux-performance.md). macOS timings are diagnostic only and do not establish Linux equivalence. |
| Live service | `tools/run_phase8_qualification.py` and `.github/workflows/live_qualification.yml` are retained protected-service machinery. | S3 and PostgreSQL remain `NOT_QUALIFIED`; they require real Amazon S3 and PostgreSQL `QUALIFIED` / `CLEAN` evidence under [SEED-007](../.planning/seeds/SEED-007-qualify-real-postgresql-s3-and-publish-release.md). Mocks and compatible endpoints prove adapter behavior, not this topology. |
| Publication | `tools/verify_phase8_release.py` is the retained immutable-release controller. | immutable publication remains `NOT_PUBLISHED` under SEED-007. Local readiness is not a tag, draft, GitHub release, or immutable asset. |

`PASS` means a declared evidence class completed for its recorded identity.
`UNAVAILABLE` means an external prerequisite or environment was absent.
`NOT_QUALIFIED` means evidence is incomplete, invalid, failed, deferred, or
outside that class. `NOT_PUBLISHED` means no verified immutable GitHub release
exists. `DEFERRED` is a closed nonclaim, never a synthetic pass. Retained
release-candidate artifacts bind a candidate SHA and source digest and use a
30 days retention window; this local-readiness record is not a release asset.

The full Linux core matrix covers Python 3.11 through 3.14; macOS is only the
Python 3.11/3.14 public-topology boundary smoke. TensorFlow remains an optional
handler restricted to compatible Linux Python 3.11/3.12 rows. Python 3.15 is
advisory and cannot qualify or disqualify the stable matrix.

## Local-readiness boundary

The bounded local command is intentionally separate from live qualification:

```bash
uv run --isolated --all-extras --group dev --frozen python \
  tools/verify_phase8_contracts.py --local-ready \
  --output .planning/phases/08-production-gates-and-performance-stabilization/08-LOCAL-READINESS.json
```

It proves deterministic integrity/recovery, a source-free base-wheel public
round trip, coverage-plus-Ruff ratchets, and structural bounds for one clean
source identity. It does not contact PostgreSQL/AWS, dispatch a workflow,
create a GitHub draft, publish a release, turn a skipped row into support, or
qualify Windows, controlled Linux performance, real remote services, or
immutable publication.

## Payload, handler, and catalog limits

The direct conditional-create default is **128 MiB**. Built-in payload transfer
and a format handler's private staging are bounded by that ceiling: a handler
receives a private contained staging location, must return a contained regular
artifact, and cannot turn an oversized input into multipart, temporary-object,
or fallback publication. This is a payload limit, not a performance guarantee.

Canonical payload integrity is **canonical SHA-256 plus size**. Signed ETag and
version values are opaque corroborating transport evidence only; they neither
replace the canonical digest nor authorize visibility, adoption, or cleanup.
Before deserialization, mismatched digest/size, unsafe locator, malformed
metadata, failed signature, or invalid control-object identity fails closed.

The authenticated catalog accepts portable declared field values limited to
**string, signed 64-bit integer, and boolean**. Catalog queries use bounded keyset scans over committed authority descriptors; derived projections and payload listings cannot claim canonical completeness. A bounded query is not a promise of an arbitrary-index query engine or a second lifecycle authority.

## Topology and recovery limits

`BlobStore` has one lifecycle authority. Authority promotion makes a complete
immutable generation visible; payload creation precedes promotion, while
destructive cleanup is recorded as durable cleanup debt and reconciled from
authority state. SQLite is a reliable single-host authority topology, and
PostgreSQL can strengthen a multi-host authority topology, but neither creates
cross-resource ACID with filesystem or S3 payload effects.

Interrupted external effects use attributable intent/debt and deterministic
recovery. An unattributed pre-checkpoint orphan remains invisible and
unadopted; Cacheness makes no exact-reclamation claim for it. Cacheness does
not promise every concurrent contender succeeds. **typed contention outcomes** —
such as `conflict`, a retryable timeout, or a backend-specific retryable
failure — are valid safe progress results when their declared topology permits
them. They do not justify another lock, queue, lease, sidecar, or lifecycle
authority.

See [ADR 0001](adr/0001-topology-specific-storage-guarantees.md) for the
mandatory separation of integrity, recovery, progress, and performance. See
[Catalog and topology](CATALOG_AND_TOPOLOGY.md) for static composition profiles
and the catalog API contract.

## Migration-fixture boundary

Offline migration and rebuild are stopped-worker maintenance operations. The current and immediately previous released layouts have declared fixtures; older layouts advance only through declared offline steps. Ordinary opens never
upgrade implicitly, and historical pre-production layouts do not gain
compatibility merely because a later workflow passes.
