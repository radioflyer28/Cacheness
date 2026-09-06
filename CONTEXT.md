# Cacheness storage and caching

Cacheness stores application objects and catalogs them. Caching reuses that storage engine and adds retention policy.

## Language

**BlobStore**:
The storage engine for an object's payload, authoritative catalog entry, and recovery obligations. It does not impose cache expiry or eviction.

**Catalog**:
The authoritative description of stored blobs and their application-defined metadata. Customizing catalog fields is distinct from implementing a catalog backend.
_Avoid_: Projection when referring to authoritative metadata.

**Projection**:
A rebuildable view or index derived from the catalog. Its absence, staleness, or corruption does not revoke a valid stored blob.
_Avoid_: Second catalog authority.

**Payload generation**:
One immutable version of a stored object's serialized bytes. Native format handlers determine the bytes' format.

**Cache instance**:
A consumer of BlobStore that adds keying, expiry, eviction, and statistics. It may use its own store namespace; sharing an engine does not require sharing a live store with non-cache users.
_Avoid_: Dual-role store as a prerequisite for caching.

**Cleanup debt**:
An attributable obligation to reclaim a retired or abandoned payload after a lifecycle transition. Outstanding debt is not permission to delete the current valid generation.
