<!-- refreshed: 2026-09-17 -->
# External Integrations

**Analysis date:** 2026-09-17

## Storage services and database roles

### Payload object stores

- **Local filesystem and memory** are supported immutable payload participants
  for direct local persistence and deterministic tests.
- **Amazon S3 and compatible endpoints** are supported through the guarded
  `ObstoreGenerationIO` S3 participant. The configured endpoint/region and
  standard AWS credential chain are passed to the object-store client; the
  library defines no project-specific credential variable.
- ETag/version observations are opaque signed transport evidence. They may
  corroborate an object response but never replace the descriptor digest/size
  boundary or establish lifecycle visibility.

### Catalog and authority roles

- **SQLite** provides the local transactional lifecycle authority and local
  catalog/projection support.
- **Memory** supports deterministic authority/payload tests and short-lived
  stores.
- **PostgreSQL** is an optional lifecycle authority and projection/catalog path
  using SQLAlchemy plus `psycopg`. Its live-service release qualification is
  intentionally deferred; non-live contracts do not make a production claim.
- **JSON** is a local metadata/projection representation where declared by the
  topology. It is not a substitute for a transactional authority.

## Integration boundaries

| Surface | Boundary | Current claim |
|---|---|---|
| Handler extension | `store.handlers.register_handler(...)` | A custom handler owns serialize/deserialize work on private contained paths; `BlobStore` still owns publication and reads. |
| Application metadata | `BlobStore` catalog APIs | Callers can validate, query, and update application metadata without replacing lifecycle authority. |
| Cache policy | `UnifiedCache` over one store | Cache keys, TTL, invalidation, and outcomes are policy only; no second catalog is created. |
| PostgreSQL authority | topology/configuration plus `psycopg` | Optional remote authority path; real-service qualification remains future work. |
| S3 payloads | guarded obstore participant | Object I/O is below the lifecycle boundary; one response cannot prove a cross-resource transaction. |

## Authentication and secrets

- Cacheness has no accounts, login, OAuth, or authorization subsystem.
- AWS authentication uses the configured client or its standard credential
  chain. PostgreSQL credentials belong in a caller-controlled connection URL or
  driver configuration.
- Manifest signing material is managed through `SecurityConfig`; generated
  local key files and cache artifacts are ignored by version control.

## Observability and delivery

- Library modules use standard `logging`; lifecycle details and typed outcomes
  remain local library observations rather than a hosted telemetry integration.
- The repository is a `uv_build` Python package. GitHub workflow files provide
  quality and controlled-performance automation, while local commands remain
  the authoritative developer gate.
- No HTTP server, webhook receiver, outbound service client, daemon, or
  platform registration is shipped. Decorated application functions remain
  caller code and choose their own upstream clients.

## Environment inputs

- No environment variable is required for ordinary local storage/cache use.
- Optional tests may use `CACHENESS_TEST_POSTGRES_URL` for a deliberately
  configured PostgreSQL service and standard `AWS_*` inputs for real-S3
  qualification. Those live paths are excluded from the bounded non-live suite.

---

*Current integration map refreshed for the post-cut product boundary on 2026-09-17.*
