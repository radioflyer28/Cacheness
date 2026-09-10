# Spike Manifest

## Ideas

### obstore-payload-participant

Evaluate whether `obstore` can replace Cacheness's backend-specific payload
mechanics behind one unified payload-participant seam while preserving the
existing path-based handler ecosystem and ADR 0001's single-authority lifecycle.

**Requirements:**

- Preserve `CacheHandler.put(data, Path, config)` and
  `CacheHandler.get(Path, metadata)` for built-in and user-registered handlers.
- Keep handler staging private; handlers never receive managed filesystem or
  object-store locators.
- Publish payloads at deterministic, immutable generation locators using an
  atomic create-if-absent primitive.
- Treat payload publication atomicity separately from metadata-plus-payload
  crash consistency; metadata authority remains the visibility point.
- Enforce safe suffixes and normalized locators below one managed namespace.
- Delete exact immutable generations and rely on authority-owned cleanup debt
  and reconciliation rather than object presence or listings.

## Spikes

| # | Idea | Name | Type | Validates | Verdict | Tags |
|---|------|------|------|-----------|---------|------|
| 001 | obstore-payload-participant | handler-boundary | standard | Path-based built-in and user-registered MCAP handlers survive private staging through LocalStore and MemoryStore | VALIDATED | obstore, handlers, mcap, local, memory |
| 002 | obstore-payload-participant | immutable-publication | standard | Create-if-absent yields one complete winner under collision and deterministic identity reconciles a lost response on LocalStore, MemoryStore, and mocked S3 | VALIDATED | obstore, concurrency, recovery, local, memory, s3 |
| 003 | obstore-payload-participant | authority-boundary | standard | SQLite remains the visibility authority; durable intent and identity reconcile external obstore effects without cross-resource ACID | VALIDATED | adr-0001, sqlite, atomicity, recovery, obstore |
| 004 | obstore-payload-participant | containment-deletion | standard | Namespace and suffix validation fail closed and exact immutable generation deletion preserves siblings, but obstore exposes no conditional delete | PARTIAL | obstore, deletion, path-security, suffix, s3 |
| 005 | obstore-payload-participant | streaming-replacement | standard | Local/S3 downloads stream, but direct conditional publication buffers at payload scale; bounded S3 upload requires temporary multipart upload plus conditional multipart copy and cleanup | PARTIAL | obstore, streaming, memory, s3, architecture |
