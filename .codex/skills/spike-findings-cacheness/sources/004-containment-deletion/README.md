---
spike: 004
idea: obstore-payload-participant
name: containment-deletion
type: standard
validates: "Given immutable generations and hostile locator or suffix inputs, when reads, publication, and deletion cross an obstore wrapper, then containment fails closed and only the exact named generation is deleted."
verdict: PARTIAL
related: [001, 002, 003]
tags: [obstore, deletion, path-security, suffix, s3]
---

# Spike 004: Containment and Exact Deletion

## What This Validates

Obstore accepts object keys, but the Cacheness participant must own a narrower
managed namespace. The experiment rejects traversal, absolute paths, URI-like
keys, alternate prefixes, separators, and unsafe handler suffixes before any
backend call. It then deletes one immutable generation while preserving its
sibling on LocalStore, MemoryStore, and mocked S3.

## Research

| Approach | Benefit | Limitation | Status |
|---|---|---|---|
| Pass raw locators to obstore | Minimal code | Does not encode Cacheness namespace policy | Rejected |
| Conditional delete by e-tag/version | Protects reused names | Not exposed by obstore's delete API | Unavailable |
| Validate namespace + never reuse immutable locator | Exact, backend-neutral target | Authority must retain cleanup debt and locator identity | Chosen |

Obstore 0.11.1 exposes `delete(paths)` with no version or e-tag condition. Its
documentation also states that deleting an absent object is backend-specific:
LocalStore can raise while MemoryStore and S3 can succeed. The participant must
normalize both outcomes only after validating the exact locator.

Source: <https://developmentseed.org/obstore/latest/api/delete/>

## How to Run

```bash
uv run --with obstore==0.11.1 --with 'moto[server]' --with boto3 \
  python .planning/spikes/004-containment-deletion/experiment.py
```

## What to Expect

Every backend rejects all nine hostile locators, deletes generation 001,
preserves generation 002, and normalizes repeated deletion to proven absence.
Seven unsafe suffixes are rejected. The overall verdict remains partial because
delete has no conditional version primitive.

## Investigation Trail

- Applied validation before every participant operation rather than relying on
  backend-specific path parsing.
- Retained an outside sentinel and both generation objects through all hostile
  delete attempts.
- Inspected each backend's live delete signature to verify the absence of an
  e-tag/version precondition.

## Results

**PARTIAL.** All three participants rejected nine hostile locators before
performing an operation. Five expected suffix forms were accepted and seven
unsafe forms were rejected. Exact deletion removed generation 001, preserved
generation 002, and normalized a repeated delete to proven absence. The
outside-filesystem sentinel was unchanged.

The partial verdict is intentional: obstore cannot condition deletion on an
observed e-tag or version. This is safe for Cacheness only under the stronger
immutable-generation rule that a locator is never reused. The authority must
persist the exact locator as cleanup debt; validation must precede deletion;
and successful/absent backend responses must be followed by an absence proof
when the lifecycle requires one.
