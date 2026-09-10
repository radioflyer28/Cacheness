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
- Do not modify production source, `pyproject.toml`, or `uv.lock` during the
  spike; temporary dependencies use `uv run --with`.

## Spikes

| # | Idea | Name | Type | Validates | Verdict | Tags |
|---|------|------|------|-----------|---------|------|
| 001 | obstore-payload-participant | handler-boundary | standard | Path-based built-in and user-registered MCAP handlers survive private staging through LocalStore and MemoryStore | VALIDATED | obstore, handlers, mcap, local, memory |
| 002 | obstore-payload-participant | immutable-publication | standard | Create-if-absent yields one complete winner under collision and deterministic identity reconciles a lost response on LocalStore, MemoryStore, and mocked S3 | VALIDATED | obstore, concurrency, recovery, local, memory, s3 |
