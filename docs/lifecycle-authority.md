# Lifecycle Authority Deployment and Compatibility Contract

Phase 3 replaces the unreleased file-native scheduler with one transactional
authority. This document freezes the compatibility and deployment boundaries
that later implementation plans must obey; it does not create an authority
database or make a migration decision for an existing store.

## Public compatibility baseline

`CacheReason` remains a lower-snake-case public contract. Direct `BlobStore`
errors retain their existing typed bases, including lifecycle conflict, backend,
recoverable cleanup, reconciliation, closed-store, close-timeout, and lifecycle
timeout outcomes. `BlobStore.get_metadata()` retains its existing dictionary
shape and nested metadata projection.

`ReconciliationFinding.to_dict()` and `ReconciliationReport.to_dict()` with no
arguments are the **legacy v1 compatibility view**. Their existing keys, values,
redaction behavior, and `human_summary` text are frozen. Plan 03-05 must add a
separate versioned canonical machine view for the D-13 fields; it must not
overload or silently reshape this v1 view.

## Scheduler release evidence and disposal

The retired scheduler was introduced by `571dfd6` after `origin/main` at
`a22f4b4`. The deterministic repository gate checks that no release tag contains
that introduction, no released fixture or built artifact claims scheduler
support, and no supported format document makes it a stored-format promise.
The scheduler is not a released stored format. It is local, untagged development
state only.

Its only supported disposal action is an **explicit rebuild** of that local
development store after preserving any application data the operator needs.
No scheduler reader, replay engine, or migration executor is authorized by this
phase. Release evidence that contradicts this conclusion is a fail-closed stop:
do not delete, replay, or mutate the store; instead plan an explicit read-only
migration detector and obtain a new compatibility decision.

## Lazy inspection and established evidence

Until a later authority implementation receives an approved mutation request,
read-only inspection must not create a root, `.cacheness` directory, SQLite
database, journal, sidecar, temporary file, manifest projection, key, lock, or
control object. A never-written compatible root remains byte-for-byte unchanged
across `get`, `get_metadata`, `exists`, `list`, `close`, reopen, and repeated
reconciliation dry-run.

An absent or empty root is distinct from established authority-missing evidence.
Payload, metadata, legacy, scheduler, future-authority, corrupt-authority,
mixed, and wrong-object evidence each require a typed, unchanged
rebuild-or-migration outcome. Only a contained regular existing authority file
may receive a read-only identity/integrity open, with no SQLite journal or
sidecar creation.

## Confirmed local authority identity

The durable contract is one local authority at
`.cacheness/lifecycle-authority-v1.sqlite3`, SQLite application ID
`0x43414348`, schema `user_version = 1`, and a generated store-identity row.
`LifecycleAuthority` is the sole authority for lifecycle state: the JSON
projection is observational only and cannot authorize a mutation.

## Windows local root contract

Windows mutation support is restricted to a pre-provisioned **local NTFS** root
for the current token's logon SID (`S-1-5-5-X-Y`). A persistent account SID is
not a logon-session identity. Service-plus-interactive sharing, cross-session
sharing, cross-user sharing, non-local roots, and a custom Win32 lock authority
are unsupported.

Cacheness must validate the current token, existing root, and DACL before the
first authority database creation/open and before every mutation. It must never
create the root, change its mode, disable inheritance, or add, remove, or reorder
ACEs. The deployed DACL has **inheritance disabled**. Absent, unsafe, drifted,
or unprovable roots must fail typed and unchanged while returning this offline
provisioning instruction.

Run the following **only while the store is unavailable**, as the deployment
operator. Replace the root path before use; the script resolves the current
token's logon SID and removes the persistent account's ordinary mutation grant.
It leaves only the logon SID with modify rights and narrowly documented
non-mutating `SYSTEM` and `Administrators` read/execute entries.

```powershell
$root = 'C:\CachenessAuthorityRoot'
$logonSid = (whoami /groups /fo csv | ConvertFrom-Csv |
    Where-Object { $_.SID -match '^S-1-5-5-[0-9]+-[0-9]+$' } |
    Select-Object -First 1 -ExpandProperty SID)
if (-not $logonSid) { throw 'No current-token logon SID (S-1-5-X-Y) was found.' }
$accountSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
New-Item -ItemType Directory -Path $root -ErrorAction Stop | Out-Null
icacls.exe $root /inheritance:r | Out-Null
icacls.exe $root /remove:g "*$accountSid" | Out-Null
icacls.exe $root /grant:r "*$logonSid:(OI)(CI)(M)" `
    "*S-1-5-18:(OI)(CI)(RX)" "*S-1-5-32-544:(OI)(CI)(RX)" | Out-Null
icacls.exe $root /verify
Get-Acl -LiteralPath $root
```

The resulting DACL must contain no inherited, account-SID, or unrelated write
ACE. A same-logon-session deployment is the only positive topology. A different session
or service token must be denied for authority open and root mutation; if that token cannot
be exercised on a native Windows host, the evidence is `UNAVAILABLE`, never a passing substitute.
