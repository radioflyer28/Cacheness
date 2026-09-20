---
spike: 003
idea: obstore-payload-participant
name: authority-boundary
type: standard
validates: "Given SQLite as lifecycle authority and obstore as an external payload participant, when failures occur before and after publication, then authority visibility and durable reconciliation restore consistency without claiming metadata-plus-payload ACID."
verdict: VALIDATED
related: [002]
tags: [adr-0001, sqlite, atomicity, recovery, obstore]
---

# Spike 003: Authority Boundary

## What This Validates

The experiment models ADR 0001's SQLite-plus-local-filesystem topology with a
minimal authority schema. It injects failures before and after external
publication and separately demonstrates why placing an object operation inside
a SQL transaction does not make the resources one transaction.

## Research

| Model | Visibility | Failure behavior | Status |
|---|---|---|---|
| Treat object presence as committed | Filesystem/object store | Creates a second authority and exposes orphans | Rejected |
| Pretend SQL encloses object put | SQL transaction | SQL rollback cannot undo the external effect | Invalidated by experiment |
| Durable intent, immutable effect, verified promotion | SQLite row | Crash windows reconcile from durable state and identity | Chosen |

ADR 0001 requires one lifecycle authority, immutable payload generations, and
deterministic reconciliation. Obstore's atomic put is therefore a participant
primitive. It does not expand SQLite's ACID scope.

Sources:

- `docs/adr/0001-topology-specific-storage-guarantees.md`
- <https://docs.rs/object_store/latest/object_store/trait.ObjectStore.html>

## How to Run

```bash
uv run --with obstore==0.11.1 \
  python .planning/spikes/003-authority-boundary/experiment.py
```

## What to Expect

The output explicitly reports payload atomicity as true and cross-resource
atomicity as false. A pre-publication interruption yields `EFFECT_REQUIRED`; a
verified pending payload promotes; wrong bytes yield `CONFLICT` and remain
invisible.

## Investigation Trail

- Used an actual SQLite WAL database and LocalStore rather than an in-memory
  state diagram.
- Proved the cross-resource boundary by rolling back a transaction after
  obstore returned success; the object survived and the SQL intent did not.
- Committed intent before the correct external effect, then made recovery depend
  only on intent state plus verified SHA-256 and size.

## Results

**VALIDATED.** The external LocalStore object remained after the SQLite
transaction containing its notional intent was rolled back. This is direct
evidence that obstore cannot supply metadata-plus-payload atomicity.

With the ADR sequence, a committed intent with no object yielded
`EFFECT_REQUIRED`; after publication, the same recovery promoted it. An object
published before an injected promotion failure remained invisible until its
identity was verified and SQLite promoted it. Wrong bytes at an expected
locator yielded `CONFLICT` and were never made visible.

The useful guarantee is therefore: obstore makes each immutable payload effect
atomic, while the metadata authority makes lifecycle visibility atomic and
records enough durable evidence to reconcile the gap between them.
