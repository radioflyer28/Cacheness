---
phase: 03-core-mixin-decomposition
plan: 01
subsystem: infra
tags: [refactor, core, mixins, cooperative-inheritance, decomposition]

requires:
  - phase: 02-metadata-package-split
    provides: "Clean metadata/ package — core.py can safely import from it"
provides:
  - "4 focused mixin classes beside core.py"
  - "_verification_mixin.py: VerificationMixin (249 lines) — sign/verify/integrity concern"
  - "_stats_mixin.py: StatsMixin (34 lines) — hit/miss/stats concern"
  - "_custom_metadata_mixin.py: CustomMetadataMixin (372 lines) — 9 custom metadata methods"
  - "_storage_mode_mixin.py: StorageModeMixin (186 lines) — storage-mode put/get passthrough"
  - "core.py reduced from 3,307 to ~3,100 lines via extraction"
  - "UnifiedCache class inherits from all 4 mixins via cooperative inheritance"
affects: [04-exception-handling]

tech-stack:
  added: []
  patterns:
    - "Cooperative mixin inheritance — mixins access self.config, self.metadata_backend etc. via plain self"
    - "Flat sibling files beside core.py with underscore-prefix (_*_mixin.py)"
    - "One-directional imports: core imports mixins; mixins import from metadata/handlers/interfaces/config; no mixin→core imports at module level"
    - "Lazy import from .core inside method body when mixin needs core-defined helper (_PutCleanup)"

key-files:
  created:
    - src/cacheness/_verification_mixin.py
    - src/cacheness/_stats_mixin.py
    - src/cacheness/_custom_metadata_mixin.py
    - src/cacheness/_storage_mode_mixin.py
  modified:
    - src/cacheness/core.py

key-decisions:
  - "4 mixins exactly per ROADMAP (D-01): VerificationMixin, StatsMixin, CustomMetadataMixin, StorageModeMixin"
  - "Flat files beside core.py (D-04) — no core/ sub-package, from cacheness.core import UnifiedCache works unchanged"
  - "put/get orchestration stays in UnifiedCache (D-05); mixins are leaf logic only"
  - "No __init__ in any mixin (D-06) — all initialization stays in UnifiedCache.__init__()"
  - "Plain cooperative inheritance with self.* access (D-10) — no Protocol, no ABC"
  - "MRO: UnifiedCache → VerificationMixin → StatsMixin → CustomMetadataMixin → StorageModeMixin → object (D-12)"
  - "_PutCleanup referenced in StorageModeMixin via lazy import to avoid circular dependency (D-18)"
  - "Mixin files are private (underscore prefix), not exported from __init__.py (D-15)"
  - "~85 ty unresolved-attribute warnings expected — cooperative inheritance pattern, not blocking"

patterns-established:
  - "Cooperative mixin pattern: mixins use self.* for state defined in UnifiedCache.__init__() — standard Python, ty cannot resolve but runtime works"
  - "Lazy from .core import for cross-dependency: mixin method bodies can import core-defined helpers to avoid circular module-level imports"

requirements-completed: [DECO-03, DECO-04]

duration: 25min
completed: 2026-04-03
---
