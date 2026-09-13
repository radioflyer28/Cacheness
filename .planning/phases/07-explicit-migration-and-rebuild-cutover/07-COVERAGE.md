# Phase 7 API Coverage: no new external API integration

**Detector:** `/Users/akriz/.codex/gsd-core/bin/lib/api-coverage.cjs --json`

**Scope:** The Phase 7 section of `.planning/ROADMAP.md`, followed by the body
(YAML frontmatter excluded) of every `07-*-PLAN.md` in this phase directory.
The detector is the active GSD runtime module; no local detector or capability
matrix substitute was used.

<!-- phase7-api-coverage:detector-result:start -->
```json
{
  "detected": true,
  "signals": [
    {
      "verb": "(surface)",
      "noun": "api",
      "snippet": "<action>Export the supported maintenance service, immutable caller result models, stable enums/reasons, and typed domain"
    }
  ],
  "terms": {
    "verbs": [
      "integrate",
      "integrates",
      "integrating",
      "integration",
      "wrap",
      "wraps",
      "wrapping",
      "connect",
      "connects",
      "connecting",
      "consume",
      "consumes",
      "consuming",
      "wire",
      "wires",
      "wiring",
      "onboard",
      "onboarding",
      "adopt",
      "adopts",
      "adopting"
    ],
    "nouns": [
      "api",
      "apis",
      "sdk",
      "sdks",
      "rest",
      "graphql",
      "grpc",
      "endpoint",
      "endpoints",
      "oauth",
      "oauth2",
      "webhook",
      "webhooks",
      "mcp"
    ]
  }
}
```
<!-- phase7-api-coverage:detector-result:end -->

No external API integration: Phase 7 adds offline maintenance over the existing
`PostgresqlLifecycleAuthority` and `ObstoreGenerationIO` adapters; it adds no external
service capability.

The detector's one `api` surface signal is the Phase 7 Python-library public
API wording. It does not name a newly integrated SDK, external service,
endpoint, authentication flow, credential source, or remote capability. A
capability matrix would therefore fabricate an external integration instead of
describing the phase's actual work.

Phase 7 retains deterministic adapter coverage in
`tests/contracts/test_postgresql_lifecycle_authority.py`,
`tests/test_migration_remote_contract.py`, and
`tests/contracts/test_s3_generation_io.py`. These tests exercise existing authority and
payload-adapter contracts only; they do not qualify live PostgreSQL/AWS S3.
Phase 8 alone owns real PostgreSQL/AWS S3 service qualification, compatible
service scope, Windows evidence, and performance qualification.
