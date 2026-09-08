# V2.2 causal-history infrastructure

This package is separate from `research.data` and every V2.1 scientific path.
Its contract pins the accepted V2.2 specification, seal and full 799-station
static manifest. It does not rebuild a panel or execute scientific work on import.

`Contract` loads only the three pinned governance JSON files. `StatusIndex`
accepts explicit station/timestamp evidence, uses deterministic set semantics,
and ignores stations outside the fixed roster. Every history call still uses the
entire frozen city denominator. Time is represented as integer UTC microseconds;
naive, floating-point and sub-microsecond timestamp inputs are rejected rather
than rounded across a causal boundary.

`IdealizedCountFeed` requires an explicitly exhaustive, finalized count ledger
for its declared station/time domain. Absent records inside that domain mean
zero; outside-domain reads fail rather than silently becoming zeros. Availability
is exactly hour_end <= cutoff. The feed accepts no retrospective observation
mask. Static infrastructure is the supplied benchmark conditioning, not historical
publication-time evidence. Calendar conversion uses the installed IANA timezone
database; its runtime version is recorded with the implementation test report.

`CausalHistory.at_origin` is the single implementation called by training and
inference wrappers. Its immutable output contains an explicit V2.2 artifact
identity, X_hist/M_hist, X_week, static/calendar values and prefix evidence lineage.
`arrays()` creates model-compatible float32/Boolean arrays without invoking a
model. As in the registered architecture, **storage is lag 1..24**; the existing
recurrent code flips that axis to **t-24,...,t-1**. The `chronological` view exposes
that recurrent order for inspection; do not feed it to an adapter that flips again.
There is no retrospective Y read or import in the active history function.

`FitRequest.registered` fixes source/target membership and zero/1/7/30/full elapsed
windows at HD or HF. `build_fit_snapshot` implements cutoff-C label membership but
fingerprints predictors at each original origin. Supplying explicit candidate
hours marks the result NON_SCIENTIFIC_TEST_ONLY. No full-dataset call is made by
the tests. `seal_training_keys` returns the sorted-key and snapshot content seal;
`require_training_keys` is the mandatory gate for future V2.2 fitting consumers.
The HA row accessor uses exactly this same gate. Non-scientific fixtures cannot
authorize scientific fitting. Zero-budget reuse cannot start a target fit.
No optimizer or HA fitting implementation exists in this package.

Retrospective evaluation labels have a separate immutable descriptor in
`labels.py`; it is not accepted as an idealized count feed. No final-label reader,
evaluator or Stage-4 code is implemented here.

`ArtifactIdentity` binds protocol/specification/seal/cohort, function version,
cutoff, city scope, roster and graph identity where applicable. `ArtifactLoader`
requires an externally hash-bound V2.2 index, rejects legacy/final-label paths,
checks file/content hashes and supports explicit JSON envelopes only. It does not
deserialize model code or pickle files. Graph-independent history may omit an
adjacency hash; graph artifacts and Graph-GRU checkpoint identities may not.
Future panel/cache/checkpoint orchestration must use these identities and guards;
this package does not silently retrofit V2.1 runners.

## Non-scientific verification

Run `python -B research/scripts/test_causal_history_implementation_v2_2.py` from
the repository. It loads only the named new unit tests, then three small
development integration tests after unit success. It emits JSON to stdout and
writes no files. Existing local dependencies are discovered from
`tmp/neural_build/pydeps`; no packages are downloaded.

Integration uses Dortmund's 76 frozen station IDs, a short status window plus
earlier lifetime evidence, two forecast origins, and one candidate fit hour.
Counts are explicitly synthetic. No scientific panel, actual demand labels,
scientific predictions, metrics or model fitting are involved.

The separate `verify_causal_history_implementation_v2_2.py` reads the implementation
manifest and saved test report, checks hashes and preservation, and does not
rerun tests or touch raw scientific data. Full V2.2 panel construction and every
scientific execution remain unperformed and require separate authorization.
