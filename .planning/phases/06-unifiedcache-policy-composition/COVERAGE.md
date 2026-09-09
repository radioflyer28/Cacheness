# Phase 6 External API Coverage

No external API integration: Phase 6 introduces no external service surface.

Phase 6 composes first-party `UnifiedCache` policy over the already-delivered
`BlobStore`, `StoreTopology`, catalog, and `AuthorityLifecycleEngine` contracts.
It adds no third-party API, SDK, protocol endpoint, credential, environment
variable, dashboard configuration, webhook, or network-owned lifecycle step.

The deterministic PostgreSQL/Amazon-S3 candidate tests reuse Phase 5 contract
doubles only to prove that the same cache-policy call graph preserves typed
topology outcomes. They do not contact services, extend boto3 or psycopg API
coverage, or qualify `BACK-05`; the non-substitutable real-service gate remains
owned by Phase 8.

Optional-export behavior is a capability-reporting concern, not a generalized
runtime identity. The selected `BlobStore` composition remains the cache's sole
storage identity whether caller-injected or cache-created. Therefore the
assumption-delta probe on “optional-export” requires no identity-model change.
