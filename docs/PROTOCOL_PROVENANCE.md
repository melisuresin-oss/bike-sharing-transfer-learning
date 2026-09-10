# Protocol provenance

`RESEARCH_PROTOCOL_V2_2_AMENDMENT.md` is the highest scientific authority in
this repository. The retained V2/V2.1 documents remain because the amendment
and sealed contracts carry forward or hash-bind parts of them. Where documents
conflict, V2.2 governs; inherited documents are provenance, not permission to
revert the experiment.

The final evaluation resolution is recorded in
[`FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md`](../FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md)
and
[`final_evaluation_protocol_v2_2.json`](../final_evaluation_protocol_v2_2.json).
The GRL source-domain-cardinality clarification is recorded in the adjacent
Markdown and JSON files. Compact machine-readable authority under
`research/results/` includes the causal-history specification and implementation
closure, frozen Stage 1/2/2B decisions, Stage 3 policy, Stage 4 selection, and
final execution seals.

Some retained protocol artifacts describe the prospective state at the time
they were sealed—for example, that final execution or final-target access had
not yet begun. Those statements are immutable historical controls, not the
current project status. The final result artifacts and successful final
validation layer document the subsequently completed evaluation.

## Immutable path-heavy exception

`research/results/causal_history_implementation_v2_2/implementation_manifest.json`
is retained byte-for-byte because development-data verification, Stage 1,
Stage 2, and Stage 4 source modules directly depend on it, and downstream frozen
artifacts bind its SHA-256 (`a1fe0ee486c55b1348e92d91edd42289dd49ce982129cdaa544691eafd605a9d`).
It in turn hash-binds `test_report.json`, which is retained to preserve that
scientific artifact closure.

Those two immutable files contain a historical absolute local data path inside
a recorded DuckDB query. The path is execution provenance only; it is not a
portable runtime requirement or an instruction to recreate that directory.
Unrelated path-heavy operational reports and deployment bundles are excluded.

## Published and omitted artifacts

The repository contains no raw dataset arrays, processed panels, target-label
arrays, cached tensors, fitted checkpoints, predictions, bootstrap-replicate
arrays, live worker/controller outputs, or Phase-A/Phase-B archives. The files
under `processed/protocol_v2_2/` and `research/results/` are compact protocol and
identity metadata. Public frozen result summaries are indexed by the new,
path-neutral
[`results/final_v2_2/MANIFEST.json`](../results/final_v2_2/MANIFEST.json).
