# Spike Conventions

Patterns and stack choices established across spike sessions. New spikes follow
these unless the question requires otherwise.

## Stack

- Python 3.11+ experiments matching the project runtime.
- Temporary dependencies are pinned per command with `uv run --with`; production
  dependency manifests remain unchanged during architectural spikes.
- Obstore experiments currently target `obstore==0.11.1`.
- S3 behavior is exercised through obstore's real Rust HTTP client against a
  localhost Moto server, not by mocking obstore methods.

## Structure

- Each experiment lives in `.planning/spikes/NNN-name/` with an executable
  `experiment.py` and evidence-focused `README.md`.
- Experiments emit machine-readable JSON containing their verdict and measured
  outcomes.

## Patterns

- Reuse Cacheness's private handler staging and path-based handler API; pass the
  validated open descriptor into payload participants and give readers a
  suffix-preserving private snapshot.
- Use deterministic immutable generation locators plus create-if-absent.
- Normalize only documented typed absence/collision outcomes; do not parse error
  strings or convert arbitrary failures to misses.
- Keep metadata authority visibility separate from payload presence and verify
  digest plus size during ambiguous-effect recovery.
- Measure native-library memory with externally sampled process RSS rather than
  Python `tracemalloc`.

## Tools & Libraries

- `obstore==0.11.1` for LocalStore, MemoryStore, and S3Store experiments.
- `mcap==1.4.0` for a realistic user-defined native-container handler.
- `moto[server]` and `boto3` for HTTP-level mocked S3.
- `psutil` for child-process RSS sampling.
