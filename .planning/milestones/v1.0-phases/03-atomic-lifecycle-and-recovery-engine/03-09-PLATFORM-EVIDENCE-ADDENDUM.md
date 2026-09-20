---
contract_version: 1
repository-runtime-evidence:
  command: ".venv/bin/python verify_platform.py --phase3"
  role: "Records the completed Plan 03-09 repository-interpreter and current-host runner behavior."
  source_plan: "03-09"
qualification-attestation:
  command: "uv run --python 3.11 --frozen python verify_platform.py --phase3 --require-system Windows --require-python 3.11"
  non_overridable: true
  expected_status: UNAVAILABLE
  expected_exit_code: 2
  milestone_status: NOT_QUALIFIED
  native_evidence: false
  backlog_phase: 999.1
  future_required_status: PASS
  future_required_exit_code: 0
source-binding:
  source_summary_path: ".planning/phases/03-atomic-lifecycle-and-recovery-engine/03-09-SUMMARY.md"
  source_summary_sha256: 426189ec064f6beb333d41d9adc8c90b83affcfaaf4495f5113a773a2ed7f177
  completion_commit: 947f76d
  implementation_commits:
    - 9abeb10
    - f9e64e7
    - 5735e1a
    - 5165dc1
    - d74ec66
    - 5488478
    - f05e6c1
    - 7888814
    - a846c10
    - 7d59ff4
    - d0dcc73
---

# Plan 03-09 Platform Evidence Addendum

This additive contract gives downstream plans separate, machine-readable roles for
the two Phase 3 runners. It does not amend, reinterpret, or replace the completed
Plan 03-09 summary or any of its implementation commits.

## Source binding

The frontmatter binds this artifact to the exact completed Plan 03-09 summary bytes,
its close-out commit, and every existing Task 1 and Task 2 implementation commit.
The binding is provenance only: it neither replays nor replaces that completed work.

## Command roles

- `repository-runtime-evidence` records the already-completed repository interpreter
  and current-host runner behavior from Plan 03-09.
- `qualification-attestation` is the only native-qualification attestation role. Its
  argv is fixed and non-overridable; callers must neither add overrides nor substitute
  the repository-runtime role.

## Qualification truth

For this milestone, `UNAVAILABLE` with exit code 2 is `NOT_QUALIFIED` and has
`native_evidence: false`. It is not a passing result and must not be cited as native
Windows support.

Backlog Phase 999.1 must run the designated qualification role in an eligible native
Windows Python 3.11 environment and produce `PASS` with exit code 0 before any
Windows-qualified release. That later run additionally supplies the protected NTFS
root and the distinct session or service-token evidence required by D-22 and D-31.
