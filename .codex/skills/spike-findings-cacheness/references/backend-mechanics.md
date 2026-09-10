# Backend Mechanics and Limits

## Requirements

- Expose one contained participant seam for LocalStore, MemoryStore, and S3Store.
- Enforce normalized locators and native suffixes before calling obstore.
- Delete only exact never-reused immutable generations.
- Keep lifecycle policy and reconciliation above obstore.

## How to Build It

1. Wrap obstore behind a Cacheness-owned participant rather than exposing its
   broad API to `BlobStore` or handlers. The minimal primitives are conditional
   create, streamed exact read, head/identity evidence, exact delete, and bounded
   inventory where maintenance requires it.
2. Validate a relative locator below the managed `generations/` namespace.
   Reject absolute paths, schemes, backslashes, empty/dot components, traversal,
   alternate prefixes, and unexpected characters before any backend operation.
3. Normalize backend-specific absence behavior after validation. LocalStore may
   raise for an absent delete while MemoryStore and S3Store may report success;
   prove absence with an exact head when lifecycle settlement requires it.
4. Choose and document one S3 publication policy before implementation:

   - **Direct conditional put:** `put(locator, source, mode="create")` is simple
     and safe but materializes input at payload scale. It requires an explicit
     payload-size bound.
   - **Bounded-memory conditional publication:** multipart-upload a temporary
     object, configure `copy_if_not_exists="multipart"`, conditionally copy to
     the immutable final locator, then settle exact temporary-object and hidden
     multipart cleanup. This preserves bounded client memory but adds effects
     and recovery obligations.

5. Replace low-level backend mechanics only. The responsibility mapping is:

   | Replace with obstore | Retain in Cacheness |
   |---|---|
   | Filesystem byte/stream CRUD, stat, list, object creation | Root policy, locator validation, durability claims |
   | In-memory object dictionary and its separate handler I/O adapter | Ephemeral topology and resource limits |
   | S3 request construction, upload/download, parts, head/delete/list | Credential/prefix policy, limits, digests, domain errors, reconciliation evidence |
   | Filesystem-specific guarded publication/snapshot calls | Private handler staging, suffix and descriptor checks, snapshot lifetime |

## What to Avoid

- Do not describe obstore as replacing `GuardedHandlerIO`, lifecycle authority,
  promotion, cleanup debt, or reconciliation.
- Do not assume `put(mode="create")` streams large files. Obstore 0.11.1 forces
  non-multipart conditional put and materializes the input.
- Do not enable staged multipart copy without modeling the temporary object and
  abandoned multipart upload as recoverable cleanup obligations.
- Do not rely on conditional deletion; obstore 0.11.1 exposes path-only delete
  with no e-tag/version precondition. Never reuse generation locators.
- Do not assume MemoryStore downloads are chunked; the spike returned the full
  64 MiB object in one chunk and retained the payload in process memory.

## Constraints

- Direct mocked-S3 conditional put grew RSS by about 106 MiB for a 32 MiB
  payload and 320 MiB for a 128 MiB payload.
- Multipart stage/copy grew from about 55 MiB to 67 MiB over the same input
  range, demonstrating bounded client-side behavior at the cost of extra effects.
- LocalStore and S3Store downloads were chunked at roughly 10 MiB in the spike.
- S3 multipart copy-if-absent is opt-in and warns that failed multipart cleanup
  is best effort; configure bucket lifecycle cleanup as defense in depth.

## Origin

Synthesized from spikes: 004, 005
Source files available in: `sources/004-containment-deletion/`,
`sources/005-streaming-replacement/`

