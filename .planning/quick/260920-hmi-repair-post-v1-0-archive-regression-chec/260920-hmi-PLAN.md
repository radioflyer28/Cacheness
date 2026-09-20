---
quick_id: 260920-hmi
status: complete
scope: post-archive local regression repair
---

# Repair post-v1.0 archive regression checks

## Goal

Make the documented complete local regression suite runnable and truthful on
a fresh checkout after v1.0 milestone archiving. Update verifier/test paths to
the archived artifacts they actually validate, without recreating active
phase directories, weakening assertions, or changing BlobStore behavior.

## Tasks

1. Inventory failing tests and verifier inputs; distinguish relocated evidence
   from genuinely obsolete assertions (especially the existing `v1.0` tag).
2. Update current test/verifier contracts and broken current-doc links to the
   archived v1.0 locations. Preserve intentional historical content and
   current release nonclaims. Add targeted checks where the archive boundary
   needs explicit behavior.
3. Run scoped checks, Ruff on touched Python, and the complete documented
   local regression command in a fresh checkout. If clean, commit the repair
   and produce a summary; otherwise record exact unresolved failures and stop
   before main integration.

## Constraint

No compatibility symlinks, no blanket test exclusions/skips, and no rewrite
of dated evidence to manufacture a green result.
