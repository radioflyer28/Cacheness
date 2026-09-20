<!-- refreshed: 2026-09-17 -->
# Testing Patterns

**Analysis date:** 2026-09-17

## Test framework and commands

- pytest is configured in `pyproject.toml` with strict markers and `tests/` as
  the test root. Tests use native `assert`, `pytest.raises`, `caplog`,
  `importorskip`, NumPy helpers, and dataframe comparison helpers as needed.
- Use the frozen all-extras environment for project gates:

  ```bash
  uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false
  uv run --isolated --all-extras --group dev --frozen pytest -q -o log_cli=false \
    -m 'not (live_postgresql or live_aws_s3 or live_remote)'
  ```

- Scope one module/node while iterating. Run `uv lock --check`, scoped Ruff,
  and the packaging matrix when source/package boundaries change.

## Test organization

| Location | Purpose |
|---|---|
| `tests/test_*.py` | Unit and focused integration behavior for core, handlers, catalog, migration, and policy. |
| `tests/contracts/` | Lifecycle authority, topology, payload, and participant contracts. |
| `tests/integration/` | Multi-component stores plus opt-in remote topology paths. |
| `tests/packaging/` | Fresh source-free wheel member, metadata, import, and local round-trip tests. |
| `tests/qualification/` | Evidence schemas, release boundaries, platform, and live-service nonclaims. |
| `tests/performance/` | Complexity/memory/benchmark harness contracts. |

`tests/conftest.py` supplies shared test configuration. Keep fixtures near their
own behavior unless multiple modules need the same invariant.

## Test construction conventions

- Prefer `tmp_path`, temporary local stores, and real serialization round trips
  for persistence, catalog, and handler contracts.
- Use small fake authorities/participants or `unittest.mock` at unavailable
  service, clock, permission, or SDK boundaries. Moto covers deterministic S3
  behavior without reaching real cloud services.
- Reset global registry state around registry tests; close every store/backend
  created by a fixture.
- Test policy through `UnifiedCache` and persistence through `BlobStore`; do
  not use a cache assertion as a substitute for an authority or exact-generation
  contract.
- Mark actual external service requirements with `live_postgresql`,
  `live_aws_s3`, or `live_remote`. Do not add unrelated exclusions to the
  bounded full-suite command.

## Required quality evidence

- Lifecycle changes require the appropriate authority, topology, reconciliation,
  and handler regression nodes plus ADR 0001 review.
- Packaging changes require `uv lock --check` and
  `tests/packaging/test_wheel_matrix.py`; the harness builds one artifact,
  inspects its archive/install metadata, rejects retired modules, requirements,
  extras, and handler identities, and exercises installed local
  `BlobStore`/`UnifiedCache` round trips.
- Documentation/maps require their exact source-contract tests and a
  current-reference scan. Do not satisfy such a scan by rewriting dated audits
  or completed phase summaries.
- The frozen non-live suite is the broad regression gate. It is intentionally
  not live PostgreSQL/Amazon-S3 or controlled-Linux performance qualification.

## Active coverage limits

- Real PostgreSQL/Amazon-S3 qualification and immutable publication evidence
  remain future work; mocked/non-live tests make no broader claim.
- Controlled-Linux performance budgets are deferred while the benchmark harness
  remains checked in for diagnostic measurement.

---

*Current testing map refreshed for the post-cut product boundary on 2026-09-19.*
