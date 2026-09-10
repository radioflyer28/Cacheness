---
spike: 002
idea: obstore-payload-participant
name: immutable-publication
type: standard
validates: "Given deterministic immutable locators, when concurrent writers use create-if-absent and a response is lost after publication, then exactly one complete object wins and identity reconciliation distinguishes committed, absent, and conflicting effects."
verdict: VALIDATED
related: [001]
tags: [obstore, concurrency, recovery, local, memory, s3]
---

# Spike 002: Immutable Publication

## What This Validates

This experiment separates publication safety from progress. Twelve writers
race on one locator; one may succeed and the other eleven must receive a typed
collision without overwriting it. A second case injects a lost acknowledgement
after the write returns and reconciles the deterministic locator by SHA-256 and
byte size.

## Research

| Approach | Primitive | Pros | Cons | Status |
|---|---|---|---|---|
| Default put/upsert | `put` | Simple | Can overwrite immutable generations | Rejected |
| Rename into place | `rename` | Familiar filesystem shape | Source deletion is not universally atomic | Rejected |
| Conditional creation | `put(mode="create")` | Atomic no-overwrite contract across object_store backends | Collision is a normal typed outcome | Chosen |

Obstore delegates to Apache Arrow's Rust `object_store`, whose put contract is
whole-object atomicity: a reader does not observe a partial failed write.
`PutMode::Create` supplies the create-if-absent precondition. Obstore exposes
`AlreadyExistsError` and `PreconditionError`, so collision does not require
parsing exception text.

Sources:

- <https://developmentseed.org/obstore/latest/api/put/>
- <https://docs.rs/object_store/latest/object_store/trait.ObjectStore.html>
- <https://docs.rs/object_store/latest/object_store/enum.RenameTargetMode.html>

## How to Run

```bash
uv run --with obstore==0.11.1 --with 'moto[server]' --with boto3 \
  python .planning/spikes/002-immutable-publication/experiment.py
```

## What to Expect

JSON reports one winner and eleven collisions for MemoryStore, LocalStore, and
an obstore S3Store talking over HTTP to Moto. Response-loss recovery reports
`COMMITTED`, deliberate wrong bytes report `CONFLICT`, and an unpublished
locator reports `ABSENT`.

## Observability

The JSON output contains timestamped publication and reconciliation events,
plus aggregate event/collision counts.

## Investigation Trail

- Verified obstore 0.11.1 against Moto through its actual Rust HTTP S3 client,
  rather than mocking the Python method surface.
- Used independent byte strings during collisions so an accidental overwrite
  could be attributed to a specific writer.
- Made the response-loss injection deterministic: it raises only after obstore
  has returned success.
- The first run revealed that MemoryStore reports absence as Python's built-in
  `FileNotFoundError`, despite obstore also exporting `NotFoundError`. Recovery
  must normalize both typed absence forms, without treating arbitrary failures
  as absence.

## Results

**VALIDATED.** Each backend produced exactly one complete winner and eleven
typed collisions. Across 36 competing writes there were three winners and 33
expected collision outcomes; no losing payload replaced or mixed with the
winner.

For all three backends, a response lost after successful publication was
classified `COMMITTED` from the deterministic locator, SHA-256, and byte size.
Repeating create-if-absent produced a collision and reconciled to the same
committed effect. Deliberately wrong expected bytes classified `CONFLICT`, and
an unpublished locator classified `ABSENT`.

This validates immutable payload publication and effect recovery—not metadata
visibility or metadata-plus-payload atomicity.
