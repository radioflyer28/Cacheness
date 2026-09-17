# Handler Integration

## Requirements

- Preserve the existing path-based `FormatHandler.put` and `FormatHandler.get`
  signatures and store-local custom registration.
- Keep `store.handlers.register_handler(...)` as the one custom-format
  extension point; it does not create another storage lifecycle.
- Never give a handler a managed filesystem path, S3 key, or obstore instance.
- Validate the handler artifact and its native suffix before publication.
- Support materializing single-file handlers; reject directory, multi-file, or
  lazy path-retaining results explicitly.

## How to Build It

1. Keep `GuardedHandlerIO` as the handler-facing boundary. Its private staging,
   symlink checks, descriptor/inode identity, suffix validation, and result
   normalization are security responsibilities, not backend mechanics.
2. Make publication backend-neutral by passing the retained, validated open
   descriptor to the obstore participant:

   ```python
   with handler_io.stage(handler, value, config) as staged:
       locator = managed_locator(generation, staged.suffix)
       with staged.open() as (source, byte_size):
           store.put(locator, source, mode="create")
   ```

3. For reads, stream the object into a private, suffix-preserving snapshot and
   keep that snapshot alive only for the synchronous handler call:

   ```python
   with private_snapshot(suffix) as path:
       with path.open("xb") as output:
           for chunk in store.get(locator):
               output.write(chunk)
       return handler.get(path, metadata)
   ```

4. Resolve write handlers through the existing `HandlerRegistry`; resolve read
   handlers from the persisted handler type and payload contract. Obstore does
   not participate in format selection. Optional dataframe and other format
   integrations stay request-bound rather than announcing availability at base
   import time.
5. Add contract tests with at least one built-in native format and one separately
   registered third-party handler. Spike 001's MCAP handler is the reference
   example.

## What to Avoid

- Do not change handlers to accept streams merely because obstore supports them.
- Do not expose LocalStore paths to handlers; that design cannot generalize to
  MemoryStore or S3Store.
- Do not pass only the staged path to obstore after validating a descriptor;
  doing so reopens a path race. Pass the retained open source.
- Do not return lazy objects that still need the private snapshot after
  `FormatHandler.get` returns.

## Constraints

- Validated with obstore 0.11.1, Cacheness's NumPy `.npz` handler, and a custom
  MCAP 1.4.0 handler.
- The path API necessarily uses temporary disk even for MemoryStore and S3Store.
- Handler output remains exactly one ordinary regular file with a bounded safe
  suffix.

## Origin

Synthesized from spike: 001
Source files available in: `sources/001-handler-boundary/`
