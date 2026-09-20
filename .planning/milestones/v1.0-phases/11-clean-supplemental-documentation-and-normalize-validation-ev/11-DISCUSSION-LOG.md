# Phase 11: Clean Supplemental Documentation and Normalize Validation Evidence - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-17
**Phase:** 11-clean-supplemental-documentation-and-normalize-validation-ev
**Areas discussed:** Supplemental guides, Validation artifacts, Phase 3 evidence, Completion proof

---

## Supplemental Guides

### Default document treatment

| Option | Description | Selected |
|--------|-------------|----------|
| Consolidate selectively | Move useful material into canonical guides, repair distinct guidance, and delete redundant or misleading documents. | ✓ |
| Repair all five | Preserve and update every stale guide in place. | |
| Delete all five | Rely only on the existing canonical documentation. | |
| Other | Supply a different policy. | |

**User's choice:** Consolidate selectively.

### TensorFlow documentation

| Option | Description | Selected |
|--------|-------------|----------|
| Fold into format reference | Delete the standalone guide but retain a bounded supported-feature reference. | |
| Repair standalone guide | Keep a dedicated tutorial with unsupported claims removed. | |
| Remove all TensorFlow documentation | Stop presenting the dormant feature as supported. | ✓ |
| Other | Supply a different treatment. | |

**User's choice:** Remove the TensorFlow docs in preparation for removal.
**Notes:** The user believed the handler had already been removed. Inspection showed that the handler, extra, lock entries, tests, and qualification surface remained while automatic registration was disabled.

### TensorFlow implementation scope

| Option | Description | Selected |
|--------|-------------|----------|
| Remove completely in Phase 11 | Delete the dormant implementation and its entire package/qualification surface now. | ✓ |
| Documentation removal only | Keep implementation removal in the future seed. | |
| Separate Phase 12 | Preserve Phase 11's original boundary and remove TensorFlow before milestone completion separately. | |

**User's choice:** Remove TensorFlow completely in Phase 11.

### Platform guides

| Option | Description | Selected |
|--------|-------------|----------|
| Consolidate into release qualification | Move unique verified material into the canonical qualification page and delete both platform guides. | ✓ |
| Keep one cross-platform guide | Merge Windows notes into a repaired cross-platform document. | |
| Repair both | Maintain separate cross-platform and Windows documents with bounded claims. | |
| Other | Supply a different treatment. | |

**User's choice:** Consolidate into `RELEASE_QUALIFICATION.md` and delete both guides.

---

## Validation Artifacts

### Revalidation depth

| Option | Description | Selected |
|--------|-------------|----------|
| Evidence-backed normalization | Use existing phase proof plus a bounded current non-live gate; add focused checks only for gaps. | ✓ |
| Full phase-by-phase revalidation | Replay complete validation independently for all seven phases. | |
| Metadata-only normalization | Change statuses without current execution. | |
| Other | Supply a different standard. | |

**User's choice:** Evidence-backed normalization.

### Canonical file structure

| Option | Description | Selected |
|--------|-------------|----------|
| Normalize existing files in place | Bring each existing validation artifact to the current schema. | ✓ |
| Add wrapper reports | Preserve old files and create a second canonical report per phase. | |
| One milestone index | Keep phase files non-canonical and translate them through one aggregate. | |
| Other | Supply a different structure. | |

**User's choice:** Normalize existing validation files in place.

### Obsolete rows

| Option | Description | Selected |
|--------|-------------|----------|
| Traceable supersession | Preserve the obligation and point to replacement evidence or an approved removal. | ✓ |
| Rewrite as current | Replace old wording without recording supersession. | |
| Leave unchanged | Preserve obsolete pending rows and their discovery failures. | |
| Other | Supply a different treatment. | |

**User's choice:** Traceable supersession.

### Deferred evidence

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve scoped validation | Validate approved local/deterministic scope while retaining explicit nonclaims. | ✓ |
| Block all validation | Require every deferred environment before any validated status. | |
| Omit deferred classes | Remove unavailable evidence from validation records. | |
| Other | Supply a different policy. | |

**User's choice:** Preserve scoped validation and explicit deferrals.

---

## Phase 3 Evidence

### Validation record shape

| Option | Description | Selected |
|--------|-------------|----------|
| Compact canonical replacement | Replace the obsolete draft with current scope, evidence, nonclaims, and stop conditions. | ✓ |
| Historical appendix | Add a current section while retaining the full obsolete body in the live file. | |
| Leave draft | Continue relying on the milestone audit to translate it. | |
| Other | Supply a different treatment. | |

**User's choice:** Replace it with a compact canonical validation record.

### Current execution boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Finite contract regression | Run current integrity, recovery, cache-over-store, and non-live gates once. | ✓ |
| Exact historical replay | Recreate all original interpreters, fixtures, benchmarks, and detached-checkout evidence. | |
| Later evidence only | Run no current Phase 3-focused tests. | |
| Other | Supply a different boundary. | |

**User's choice:** Finite contract regression only.

### Failure authority

| Option | Description | Selected |
|--------|-------------|----------|
| Classify and stop | Fix obsolete evidence only; a genuine lifecycle defect requires separately authorized ADR-scoped work. | ✓ |
| Repair lifecycle code | Modify production coordination until the gate passes. | |
| Retain historical pass | Normalize as validated despite a current failure. | |
| Other | Supply a different rule. | |

**User's choice:** Classify and stop.

### Original verification provenance

| Option | Description | Selected |
|--------|-------------|----------|
| State it plainly | Preserve the user-approved direct primary-agent qualification and do not invent independent review. | ✓ |
| Treat Phase 11 as independent verification | Retroactively fill the originally planned verifier role. | |
| Omit provenance | Record results without who performed the qualification. | |
| Other | Supply a different treatment. | |

**User's choice:** State it plainly.

---

## Completion Proof

### Required gate

| Option | Description | Selected |
|--------|-------------|----------|
| Layered bounded gate | Combine removal/package, docs, validation discovery, Phase 3, Ruff, wheel, and frozen non-live proof. | ✓ |
| Documentation only | Check docs and schemas without package/regression proof. | |
| Full release qualification | Also require deferred live, Linux, Windows, and publication evidence. | |
| Other | Supply a different gate. | |

**User's choice:** Layered bounded gate.

### Documentation regression ownership

| Option | Description | Selected |
|--------|-------------|----------|
| Canonical-example ownership | Link to the four executable examples; test any additional runnable snippet exactly. | ✓ |
| Execute every Markdown block | Run all code fences, including illustrative fragments. | |
| Manual review only | Add no durable regression ownership. | |
| Other | Supply a different policy. | |

**User's choice:** Canonical-example ownership.

### Milestone audit closure

| Option | Description | Selected |
|--------|-------------|----------|
| Refresh existing audit | Resolve debts in the current audit, retain deferrals, and issue a current verdict. | ✓ |
| Add closure addendum | Preserve the original audit unchanged and add a second result. | |
| No audit update | Rely only on Phase 11 artifacts. | |
| Other | Supply a different treatment. | |

**User's choice:** Refresh the existing milestone audit.

### TensorFlow seed

| Option | Description | Selected |
|--------|-------------|----------|
| Mark fulfilled and retain | Preserve rationale, link Phase 11 resolution, and prevent re-promotion. | ✓ |
| Delete seed | Remove the seed and references. | |
| Leave open | Keep completed work in the active backlog. | |
| Other | Supply a different treatment. | |

**User's choice:** Mark it fulfilled and retain it as history.

## the agent's Discretion

- Exact canonical destination or deletion outcome for unique pandas and custom-metadata content.
- Exact bounded test selectors, supersession mappings, and internal edit sequence.

## Deferred Ideas

- Real PostgreSQL/Amazon-S3 qualification and immutable publication.
- Controlled-Linux performance qualification.
- Native Windows lifecycle qualification.
- Existing Narwhals, XXH3, and handler developer-kit backlog work.
