# Deferred Items

- The legacy facade lifecycle suites still construct `UnifiedCache` through the
  removed implicit `cacheness(config)` constructor and therefore fail before
  exercising their updated removal-report assertions. Reconcile or retire those
  helpers as part of the Phase 6 explicit public-surface cutover; do not restore
  an implicit cache/store compatibility path.
  status: acknowledged
