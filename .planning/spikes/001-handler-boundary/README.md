---
spike: 001
idea: obstore-payload-participant
name: handler-boundary
type: standard
validates: "Given existing path-based built-in and custom handlers, when private staged files round-trip through obstore LocalStore and MemoryStore, then values, suffixes, metadata, and custom registration work without exposing obstore to handlers."
verdict: VALIDATED
related: []
tags: [obstore, handlers, mcap, local, memory]
---

# Spike 001: Handler Boundary

## What This Validates

Given Cacheness's existing path-oriented `CacheHandler` interface, prove that
an obstore-backed participant can remain below the guarded staging boundary.
The experiment exercises the built-in NumPy handler and a separately
registered handler that uses the real MCAP Python package.

## Research

| Approach | Tool | Pros | Cons | Status |
|---|---|---|---|---|
| Change handlers to streams | obstore iterables | Direct transfer | Breaks all custom handlers and native path APIs | Rejected |
| Give handlers LocalStore paths | obstore LocalStore | Minimal adapter | Cannot work for Memory/S3 and leaks managed paths | Rejected |
| Private path staging and snapshots | `GuardedHandlerIO` + obstore | Preserves handlers and one-file native formats | Requires temporary disk for path-only readers/writers | Chosen |

Obstore 0.11.1 accepts a `Path`, binary file, or iterable for `put`, supports
`mode="create"`, and exposes response iteration for streaming downloads. MCAP
1.4.0 reads and writes ordinary binary file objects, making it representative
of a user-defined native container handler.

Sources:

- <https://developmentseed.org/obstore/latest/api/store/local/>
- <https://developmentseed.org/obstore/latest/cookbook/>
- <https://mcap.dev/docs/python/raw_reader_writer_example>

## How to Run

```bash
uv run --with obstore==0.11.1 --with mcap \
  python .planning/spikes/001-handler-boundary/experiment.py
```

## What to Expect

JSON reports two successful round trips for each of MemoryStore and LocalStore:
one `.npz` NumPy payload and one `.mcap` custom payload.

## Investigation Trail

- Preserved both public handler method signatures unchanged.
- Passed the already-validated staged file descriptor to obstore rather than
  handing obstore the path, retaining the current stage identity guarantee.
- Materialized reads to a private suffix-preserving snapshot because native
  handlers require a filesystem path.
- Ran the same built-in and custom-handler cases against both MemoryStore and
  LocalStore. Four independent round trips reproduced exact values and retained
  handler metadata.

## Results

**VALIDATED.** The path-based handler contract does not need to change.
`HandlerRegistry` selected the user-provided MCAP handler at priority zero and
the built-in NumPy handler normally. Both generated native single-file
containers, obstore accepted the retained validated file descriptor, and the
handlers reconstructed their values from private snapshots.

The boundary has an important constraint: it supports materializing,
single-file handlers. A handler that returns a lazy value tied to its input path
or produces a directory/multiple files would outlive or violate the snapshot
contract and should fail explicitly rather than expand the participant API.
