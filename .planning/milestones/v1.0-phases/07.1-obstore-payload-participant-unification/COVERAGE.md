# Phase 07.1 API Coverage: obstore Payload Participant

**Scope:** The exact obstore 0.11.1 surface consumed by Cacheness for local,
memory, and Amazon S3 payload effects. `INTEGRATE` is the default. Every
`OPT-OUT` is explicit so the participant cannot silently imply lifecycle,
security, async, or broad object-store capabilities that Cacheness does not use.

| capability | decision | reason |
| --- | --- | --- |
| cacheness-obstore-adapter | INTEGRATE | |
| localstore-construction | INTEGRATE | |
| memorystore-construction | INTEGRATE | |
| s3store-construction | INTEGRATE | |
| aws-credential-chain-or-provider | INTEGRATE | |
| explicit-aws-region | INTEGRATE | |
| expected-bucket-owner | OPT-OUT | D-16 supersedes owner pinning because obstore 0.11.1 cannot sign it correctly across every consumed request. |
| production-custom-endpoint | OPT-OUT | Native Amazon S3 is the named production topology; compatible services require separate qualification. |
| conditional-create | INTEGRATE | |
| overwrite-mode | OPT-OUT | Replacement publishes a new never-reused immutable generation locator. |
| multipart-upload-or-copy | OPT-OUT | D-02 selects bounded direct create without temporary-object or abandoned-upload cleanup effects. |
| put-result-etag-version | INTEGRATE | |
| exact-head-objectmeta | INTEGRATE | |
| backend-neutral-transport-observation-capability | INTEGRATE | |
| metadata-update-head-or-evidence-rebind | OPT-OUT | D-11 binds evidence to immutable payload identity, so metadata-only authority transactions preserve it without object-store work. |
| streamed-exact-get | INTEGRATE | |
| range-or-conditional-get | OPT-OUT | Cacheness must verify the complete canonical payload before deserialization. |
| exact-delete | INTEGRATE | |
| conditional-delete | OPT-OUT | Obstore 0.11.1 lacks it; never-reused authority-bound locators make exact path deletion safe. |
| bounded-prefix-list | INTEGRATE | |
| arrow-list-output | OPT-OUT | Lifecycle maintenance consumes bounded typed object metadata rather than Arrow tables. |
| bulk-delete-copy-rename | OPT-OUT | Authority work is settled per exact immutable generation to preserve attribution. |
| object-attributes-and-tags | OPT-OUT | Canonical and application metadata commit in the selected lifecycle authority. |
| synchronous-store-methods | INTEGRATE | |
| native-async-methods | OPT-OUT | Native asynchronous storage remains deferred under EXTN-03. |
| collision-precondition-errors | INTEGRATE | |
| typed-not-found-errors | INTEGRATE | |
| permission-auth-path-config-errors | INTEGRATE | |
| bounded-sdk-retry-config | INTEGRATE | |
| sdk-checksum-diagnostics | INTEGRATE | |
| bucket-and-account-administration | OPT-OUT | Cacheness consumes an externally provisioned least-privilege bucket and does not administer AWS infrastructure. |

## Detailed role rationale

<!-- phase071-api-coverage:start -->
| role | capability | decision | reason |
| --- | --- | --- | --- |
| obstore / all stores | Cacheness-owned `ObstoreGenerationIO` adapter | INTEGRATE | Keeps the external SDK below `PayloadGenerationIOProvider`; lifecycle intent, visibility, and debt remain in `AuthorityLifecycleEngine`. |
| obstore / LocalStore | `LocalStore(prefix, mkdir=True)` | INTEGRATE | Supplies contained local object mechanics behind the same participant contract as memory and S3. |
| obstore / MemoryStore | `MemoryStore()` | INTEGRATE | Supplies process-local ephemeral object mechanics without a second Cacheness byte dictionary. |
| obstore / S3Store | `S3Store(bucket, prefix, config)` | INTEGRATE | Supplies Amazon S3 object mechanics with explicit bucket, prefix, and region. |
| obstore / S3Store | Standard AWS credential chain or injected credential provider | INTEGRATE | Uses native AWS authorization without storing secrets in Cacheness metadata, logs, or evidence. |
| obstore / S3Store | Explicit AWS region | INTEGRATE | Region is mandatory for production S3 composition and is part of the declared topology configuration. |
| obstore / S3Store | `ExpectedBucketOwner` | OPT-OUT | D-16 deliberately supersedes Phase 5 owner pinning because obstore 0.11.1 cannot sign it correctly across every consumed request; use tightly scoped IAM, bucket policy, explicit bucket/region, and stable bucket ownership. |
| obstore / S3Store | Production custom endpoint | OPT-OUT | Phase 07.1 and BACK-05 name native Amazon S3; endpoint overrides are allowed only inside deterministic moto tests and compatible services require separate qualification. |
| obstore / all stores | `put(path, source, mode="create", use_multipart=False)` | INTEGRATE | Conditionally creates one deterministic immutable generation and exposes typed collision evidence. |
| obstore / all stores | Overwrite mode | OPT-OUT | Generation locators are never reused; replacement always publishes a new immutable locator. |
| obstore / S3Store | Multipart upload and multipart copy-if-absent | OPT-OUT | D-02 selects bounded direct create and rejects temporary-object, conditional-copy, abandoned-upload, and multipart cleanup lifecycles. |
| obstore / all stores | `PutResult.e_tag` and `PutResult.version` | INTEGRATE | Captures optional opaque transport observations; canonical identity remains signed SHA-256 plus byte size. |
| obstore / all stores | `head(path)` / `ObjectMeta` | INTEGRATE | Observes exact locator size, ETag, and optional version for verification, response-loss recovery, delete settlement, and the developer comparison API. |
| Cacheness participant seam | Optional backend-neutral transport-observation protocol | INTEGRATE | BlobStore queries an optional structural capability at the existing participant seam; it never imports or type-checks the concrete obstore adapter, and unsupported providers report `UNAVAILABLE`. |
| Cacheness authority lifecycle | HEAD/regenerate/rebind evidence after catalog or user-metadata-only update | OPT-OUT | D-11 binds authenticated evidence to immutable store/key identity, generation, exact locator, canonical SHA-256, and byte size; unchanged-payload updates preserve it inside their existing authority transaction. |
| obstore / all stores | `get(path)` / `GetResult.stream(min_chunk_size=...)` | INTEGRATE | Materializes a bounded private snapshot while retaining the path-based handler contract. |
| obstore / all stores | Range and conditional-get options | OPT-OUT | Cacheness verifies the complete canonical payload before deserialization; partial reads cannot satisfy that contract. |
| obstore / all stores | `delete(path)` | INTEGRATE | Deletes one exact immutable generation; an exact follow-up head proves absence when debt settlement requires it. |
| obstore / all stores | Conditional delete by ETag/version | OPT-OUT | Obstore 0.11.1 does not expose it; never-reused authority-bound generation locators make path-only exact deletion safe. |
| obstore / all stores | `list(prefix, offset, chunk_size)` | INTEGRATE | Supplies one bounded maintenance-evidence page only; listings never establish catalog membership, visibility, adoption, or deletion authority. |
| obstore / all stores | Arrow listing output | OPT-OUT | Lifecycle maintenance consumes bounded typed object metadata and does not depend on Arrow tables. |
| obstore / all stores | Bulk delete/copy/rename | OPT-OUT | Authority-owned work is settled per exact immutable generation; broad mutations would obscure attribution and recovery. |
| obstore / all stores | Object attributes and tags | OPT-OUT | Canonical and application metadata commit in the selected authority; object metadata cannot become a second catalog. |
| obstore / all stores | Sync methods and sync stream iteration | INTEGRATE | Existing `CacheHandler` and `BlobStore` APIs are synchronous. |
| obstore / all stores | Native async methods | OPT-OUT | Native async storage remains EXTN-03 and is not part of this phase's path-based handler contract. |
| obstore / all stores | `AlreadyExistsError` / precondition collision | INTEGRATE | Maps narrowly to immutable-generation collision and exact identity observation. |
| obstore / all stores | `NotFoundError` and backend-specific built-in absence | INTEGRATE | Only documented typed absence variants map to absence; arbitrary failures remain visible. |
| obstore / all stores | Permission, authentication, invalid-path, unsupported, and configuration exceptions | INTEGRATE | Preserve causes and translate by operation into Cacheness domain failures without turning them into misses. |
| obstore / all stores | Automatic retry configuration | INTEGRATE | Bounded SDK retries remain transport policy and never change authority semantics or guarantee contender success. |
| obstore / S3Store | SDK checksum configuration | INTEGRATE | May provide transport diagnostics while signed SHA-256 plus byte size remains canonical. |
| obstore / S3Store | Bucket creation, deletion, policy, versioning, replication, notifications, and retention administration | OPT-OUT | Cacheness consumes an externally provisioned least-privilege bucket and does not manage shared AWS infrastructure. |
<!-- phase071-api-coverage:end -->

## Authority and qualification boundary

Obstore owns only object-byte effects. The selected memory, SQLite, or
PostgreSQL lifecycle authority owns durable intent, committed visibility,
reconciliation, and cleanup debt. Signed ETag/version evidence is bound to
immutable payload identity rather than mutable catalog/user metadata, so
metadata-only changes perform no transport call or second authority write.
Deterministic LocalStore, MemoryStore, and
moto-backed S3 contracts in this phase do not qualify real AWS, compatible S3
services, PostgreSQL service behavior, native platforms, or performance; Phase
8 retains those non-substitutable gates.
