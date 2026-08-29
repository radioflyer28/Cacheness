# Compatibility Fixture Corpus

This directory is an immutable, test-only compatibility index. It is not a
runtime canonical manifest, a production data store, or a source for generated
payloads. The production readers are owned by Plans 01-04 and 01-12; this corpus
only establishes exact historical evidence that those readers must handle.

## Writer-only provenance

Every fixture is generated from its pinned full Git commit in a disposable
detached worktree. The generator uses only the historical writer and the fixed
non-object input:

```python
np.arange(6, dtype=np.int32).reshape(2, 3)
```

Historical readers are never invoked during generation. Before each file is
copied into this directory, the source SHA-256 is recorded; the copied SHA-256
must be equal. `provenance.json` records those paired hashes, the exact commit
and version, the historical toolchain, the semantic input, and the complete
format discriminator. `manifest.json` independently records the current hash
of every fixture file, including its provenance document.

All file names recorded in the manifest and provenance must be portable,
normalized relative paths. Absolute paths, traversal, drive-qualified paths,
UNC paths, backslashes, empty paths, and duplicate paths are rejected.

## Current staged corpus

| Fixture | Historical writer | Raw-frame API |
| --- | --- | --- |
| `array-raw-v035-compress` | `041c930fb66c7aa23f53d1f9f524e9fafdd20e68` (`0.3.5`) | `blosc2.compress` |
| `array-raw-v037-compress2` | `a756d70c858cec13ff1c885e2316c0fe725c4949` (`0.3.7`) | `blosc2.compress2` |

Plans 01-09 through 01-11 append the remaining resolved JSON, SQLite, and
decorator-key records in the exact order defined by `validate_corpus.py`.

## Validation

Run the validator after every fixture-generation task. It is
production-independent: it imports neither Cacheness nor a historical reader.
It requires the exact accumulated matrix prefix, validates the manifest and
provenance schemas, checks all SHA-256 records before inspection, and snapshots
fixture bytes before and after inspection.

```bash
uv run python tests/fixtures/compat/validate_corpus.py \
  --expected-through array-raw-v035-compress

uv run python tests/fixtures/compat/validate_corpus.py \
  --expected-through array-raw-v037-compress2
```

For raw `.b2nd` evidence, validation bounds both little-endian header lengths,
requires the exact UTF-8 `(2, 3)`/`int32` discriminators, and compares
decompressed bytes directly with the fixed input. It neither evaluates header
text nor reconstructs an array. Later JSON, NPZ, and SQLite stages extend the
same pre-read gate with complete schema discrimination, pickle-disabled NPZ
inspection, and read-only copied-database inspection.
