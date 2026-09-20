---
spike: 005
idea: obstore-payload-participant
name: streaming-replacement
type: standard
validates: "Given large handler staging files, when they upload and download through obstore participants, then memory behavior is measured and the exact replaceable versus retained Cacheness responsibilities are identified."
verdict: PARTIAL
related: [001, 002, 003, 004]
tags: [obstore, streaming, memory, s3, architecture]
---

# Spike 005: Streaming and Replacement Inventory

## What This Validates

The experiment measures resident-memory growth in isolated child processes. It
uses a 64 MiB handler staging file for LocalStore and MemoryStore, then compares
32 and 128 MiB scaling for direct conditional S3 publication and a multipart
temporary-upload plus conditional multipart-copy variant. Downloads are
iterated and their largest delivered chunk is recorded.

## Research

| Approach | Publication safety | Memory | Lifecycle cost | Status |
|---|---|---|---|---|
| Direct `put(mode="create")` | Atomic create-if-absent | Obstore documents full input materialization | One immutable object | Measure |
| Multipart upload to final | Atomic completion and bounded chunks | Configurable chunk × concurrency | Cannot combine with non-overwrite `PutMode` | Rejected for final locator |
| Multipart temporary upload + `copy(overwrite=False)` | Conditional final creation when S3 `copy_if_not_exists="multipart"` is configured | Bounded client upload and server-side copy | Temporary object, multipart residue, cleanup debt | Measure |

Obstore documents that non-default `PutMode` forces a non-multipart upload and
that non-multipart input is materialized in memory. It also documents S3's
optional multipart copy-if-not-exists implementation, including its best-effort
cleanup warning.

Sources:

- <https://developmentseed.org/obstore/latest/api/put/>
- <https://developmentseed.org/obstore/latest/cookbook/>
- <https://developmentseed.org/obstore/latest/api/store/aws/>

## How to Run

```bash
uv run --with obstore==0.11.1 --with 'moto[server]' --with boto3 --with psutil \
  python .planning/spikes/005-streaming-replacement/experiment.py
```

## What to Expect

JSON reports baseline and peak RSS, growth as a fraction of payload size, and
maximum download chunk size for six participant/publication runs.
See `REPLACEMENT-INVENTORY.md` for the code-responsibility assessment.

## Observability

Every worker uses a READY/GO handshake so its baseline is sampled only after
dependencies, store configuration, and sparse handler staging are complete.
The parent samples RSS every 5 ms until completion.

## Investigation Trail

- Rejected Python `tracemalloc` because it would miss Rust/native allocations;
  process RSS is sampled externally with psutil.
- A preliminary S3 call proved that `copy(overwrite=False)` is unsupported by
  default. It succeeds only after explicitly selecting S3's multipart
  copy-if-not-exists strategy, which itself carries abandoned-upload risk.
- Kept Moto in the parent process so its server-side buffering does not pollute
  the measured client's RSS.
- The first orchestrated run exposed a monitor race: psutil could observe the
  child disappear before `Popen.returncode` was refreshed, falsely reporting a
  successful worker as failed. An explicit `wait()` now closes that race; a
  direct worker run confirmed LocalStore completed normally.
- The initial 64 MiB run showed payload-proportional growth for direct local and
  S3 create, and MemoryStore delivered the full payload as one response chunk.
  The staged-copy S3 path was lower but still too close to payload size for a
  defensible boundedness claim, prompting explicit 32/128 MiB scaling runs.

## Results

**PARTIAL.** Direct conditional publication is not bounded-memory. For mocked
S3, RSS growth rose from 106,053,632 bytes for a 32 MiB payload to 319,537,152
bytes for 128 MiB—a 213,483,520-byte increase as payload size increased by 96
MiB. LocalStore direct create similarly grew 173,441,024 bytes for 64 MiB.

The S3 multipart-stage plus conditional multipart-copy variant behaved as a
bounded window: RSS growth rose only 11,894,784 bytes between the 32 and 128 MiB
runs (55,459,840 to 67,354,624 bytes). LocalStore and S3 downloads arrived in
chunks of roughly 10 MiB, while MemoryStore returned its entire 64 MiB object as
one chunk and necessarily retained the backing object in process memory.

The bounded S3 result is not free. It requires a temporary uploaded object,
`copy_if_not_exists="multipart"`, exact temporary cleanup, possible hidden
multipart residue, and corresponding cleanup/reconciliation policy. Therefore
obstore can replace a large amount of HTTP/request/multipart machinery, but it
does not eliminate the lifecycle complexity that matters most. For direct
create-if-absent, a documented maximum payload size would be required.

The replacement boundaries are detailed in `REPLACEMENT-INVENTORY.md`.
