<!-- refreshed: 2026-09-17 -->
# Codebase Structure

**Analysis date:** 2026-09-17

## Directory layout

```text
cacheness/
├── src/cacheness/                  # Installable package and public facades
│   ├── core.py                      # UnifiedCache policy over one BlobStore
│   ├── config.py                    # Configuration and capability declarations
│   ├── handlers.py                  # Built-in formats and handler registry
│   ├── metadata.py                  # Metadata/projection support
│   ├── storage/                     # Lifecycle, catalog, authority, payload I/O
│   │   ├── blob_store.py             # Canonical direct-persistence API
│   │   ├── composition.py            # One authority/participant topology seam
│   │   ├── obstore_generation_io.py  # Guarded local/memory/S3 participant
│   │   ├── *_lifecycle_authority.py  # Memory, SQLite, PostgreSQL authority paths
│   │   └── migration*.py             # Explicit maintenance/migration support
│   └── ...                          # Integrity, security, paths, errors, utilities
├── tests/                           # Unit, contract, integration, qualification tests
│   ├── contracts/                   # Authority/payload/topology contracts
│   ├── integration/                 # Non-live and optional remote topology tests
│   ├── packaging/                   # Fresh isolated wheel qualification
│   ├── qualification/               # Declared evidence and live-boundary checks
│   └── performance/                 # Bounded complexity/benchmark contracts
├── examples/                        # Four canonical local usage journeys
├── docs/                            # Current API, migration, format, and qualification docs
├── tools/                           # Contract, packaging, qualification, and evidence tools
├── benchmarks/                      # Measured workloads and baselines
├── pyproject.toml                   # Package/extras/test/lint configuration
└── uv.lock                          # Frozen resolution
```

## Source ownership

| Area | Canonical location | Change rule |
|---|---|---|
| Direct persistence | `storage/blob_store.py` | Route lifecycle mutations, recovery, and exact deletion through this owner. |
| Cache policy | `core.py`, `cache_policy.py`, `decorators.py` | Add only policy behavior; do not create a second storage lifecycle. |
| Topology/authority | `storage/composition.py`, `storage/*lifecycle_authority.py` | Read ADR 0001 first; preserve one declared authority. |
| Payload mechanics | `storage/obstore_generation_io.py`, `storage/guarded_handler_io.py` | Keep managed locators and staging behind the participant boundary. |
| Handlers | `handlers.py`, `interfaces.py` | Add custom formats through the store-local registry and handler contract. |
| Catalog/projections | `storage/catalog.py`, `storage/projections.py`, `metadata.py` | Keep authoritative metadata distinct from optional derived projections. |
| Security and migration | `storage/path_security.py`, `storage/integrity.py`, `storage/migration*.py` | Preserve containment, fail-closed checks, and explicit maintenance evidence. |

## Where to add work

- A new durable object capability normally belongs in `BlobStore` or an
  authority/participant contract, only after ADR 0001 review.
- A new cache semantic belongs in the policy layer and must use the existing
  store receipt/snapshot contract.
- A file format belongs in a handler and must preserve private suffix-contained
  staging and snapshot behavior.
- A metadata field/query belongs in the catalog contract, not a new persistence
  coordinator.
- New optional integration dependencies require matching extras, guarded
  imports, lock convergence, fresh-wheel tests, and documentation.

## Test and documentation placement

- Put focused behavioral tests in `tests/test_<subject>.py`; use `contracts/`
  for authority/payload invariants, `integration/` for multi-component flows,
  and `packaging/` for installed-artifact proof.
- `examples/README.md` indexes the current four local journeys:
  `memory_blob_store.py`, `durable_catalog_store.py`, `unified_cache.py`, and
  `custom_mcap_format.py`.
- Keep current user guidance in `docs/`; retain dated audits and completed phase
  records as history rather than changing them to fit a current map.

## Generated and special directories

- `cache/`, `.blobstore/`, SQLite files, signing keys, and bytecode are local
  generated state and must not be committed or cleaned broadly.
- `.planning/codebase/` contains current maps used by GSD; phase directories
  preserve planning/implementation evidence.
- `.venv/`, `.pytest_cache/`, and Ruff caches are local build/test artifacts.

---

*Current structure map refreshed for the post-cut product boundary on 2026-09-17.*
