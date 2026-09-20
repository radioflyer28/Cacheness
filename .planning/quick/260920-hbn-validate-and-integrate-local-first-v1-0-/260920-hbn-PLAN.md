---
quick_id: 260920-hbn
status: complete
scope: local-first main integration
---

# Validate and integrate the v1.0 snapshot

## Goal

Advance GitHub and local `main` to the reviewed local-first integration
history only if the live remote tip still matches the candidate's parent and
candidate-local validation passes. Preserve all prior histories and the
`v1.0` tag. Do not create or claim a qualified published release.

## Tasks

1. Confirm live HTTPS remote refs, candidate parent/tree, repository status,
   and preserved history refs. Treat the prior remote-only review as accepted
   for this integration, without reviving retired source or release tooling.
2. Validate the candidate checkout in isolation with the complete local
   regression command and fresh-wheel public smoke checks. Record failures
   accurately; do not substitute source-tree tests for packaging evidence.
3. If gates pass, carry the GSD record forward, advance local/remote main by
   fast-forward on the candidate history, and verify live refs and working
   tree. Stop safely on a changed remote or failed validation.

## Nonclaims

Local regression does not qualify live PostgreSQL/S3, controlled Linux
performance, native Windows, or immutable release publication. The tag is
not rewritten.
