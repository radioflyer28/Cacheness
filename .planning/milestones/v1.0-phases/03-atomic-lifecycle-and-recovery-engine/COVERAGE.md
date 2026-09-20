# Phase 3 API Coverage

No external API integration: Phase 3 uses Python's standard-library `sqlite3`
against a contained local authority database and an in-process memory adapter.
PostgreSQL and AWS S3 are mentioned only as downstream semantic consumers of the
`LifecycleAuthority` contract; their concrete integrations, credentials, request
shapes, and service validation remain in Phases 4 and 5.

Phase 3's native Windows verification is a local-platform compatibility target,
not an external API integration. It uses repository/uv Python selection, stdlib
SQLite, and supported OS token/DACL inspection. The local NTFS root must be
provisioned before Cacheness starts, with inheritance disabled and mutation bound
to the current token's `S-1-5-5-X-Y` logon SID rather than its persistent account
SID. Cacheness only validates that existing boundary; it never creates the Windows
root or changes its mode/DACL. Native evidence includes unchanged absent/unsafe/
drifted-root rejection, an actionable offline provisioning command, and a
different-session/service-token denial. A host
that cannot exercise that negative case—or Windows or Python 3.11 at all—reports a
blocking `UNAVAILABLE` result rather than success. This ACL is the OS-enforced
deployment boundary; it does not introduce an external service or a Cacheness lock
authority.
