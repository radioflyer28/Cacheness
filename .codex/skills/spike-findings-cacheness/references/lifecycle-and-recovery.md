# Lifecycle and Recovery

## Requirements

- Use deterministic immutable generation locators and atomic create-if-absent.
- Keep metadata authority as the sole visibility point.
- Treat payload atomicity and metadata-plus-payload crash consistency as
  distinct guarantees.
- Persist exact cleanup debt and reconcile from durable authority state plus
  verifiable payload identity.

## How to Build It

1. Read `docs/adr/0001-topology-specific-storage-guarantees.md` before changing
   this flow.
2. Have the authority commit a prepared operation containing the logical key,
   immutable locator, digest, size, and expected lineage before publication.
3. Publish the payload through obstore at that exact locator with
   `mode="create"`.
4. Normalize only typed create collisions (`AlreadyExistsError` or the relevant
   precondition type). On success, collision, or ambiguous response, observe the
   exact locator and verify digest plus byte size.
5. Promote the verified generation inside one authority transaction. Readers
   consult only the promoted authority entry; object presence and listings are
   never visibility.
6. If cleanup fails after promotion, record exact locator cleanup debt in the
   authority. Retire it only after exact deletion or proven absence.
7. Recovery state must be deterministic:

   | Durable intent | Exact object identity | Action |
   |---|---|---|
   | Missing | Any object | Do not infer lifecycle state; object is not visible |
   | Pending | Absent | Complete/retry the external effect according to policy |
   | Pending | Matches digest and size | Promote idempotently |
   | Pending | Mismatch | Fail closed as conflict |
   | Promoted | Matches | Already complete |

## What to Avoid

- Do not place an object call inside a SQL transaction and call the combination
  atomic. Spike 003 rolled back SQLite after a successful LocalStore put; the
  object survived and the SQL row did not.
- Do not add filesystem paths, object listings, process-local locks, or timing
  guesses as additional lifecycle authorities.
- Do not require every contender to succeed. A typed collision or documented
  retryable contention outcome can satisfy progress while preserving safety.
- Do not convert arbitrary obstore errors to absence. MemoryStore exposed
  absence as built-in `FileNotFoundError`; normalize known typed variants only.

## Constraints

- In 12-writer collisions on LocalStore, MemoryStore, and mocked S3Store, each
  backend produced exactly one complete winner and eleven typed collisions.
- Lost-response recovery was proven for committed, absent, and conflicting
  payload identities.
- Obstore does not extend the ACID boundary of SQLite or PostgreSQL.

## Origin

Synthesized from spikes: 002, 003, 004
Source files available in: `sources/002-immutable-publication/`,
`sources/003-authority-boundary/`, `sources/004-containment-deletion/`

