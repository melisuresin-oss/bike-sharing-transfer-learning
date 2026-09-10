# Reproducibility guide

## Scope

This repository publishes the source, specifications, tests, compact manifests,
and frozen summary results needed to inspect and reproduce the **scientific
workflow**. It does **not** reproduce the exact historical governed execution by
itself. That execution depended on immutable raw snapshots, environment and
package bindings, authorization and label-access state, fitted checkpoints,
committed predictions, materialized target arrays, bootstrap replicates, and
large Phase-A/Phase-B archives that are intentionally not public here.

There is therefore no honest one-command replay from the public tree. The map
below identifies the actual implementation layers and the order in which the
historical workflow operated.

## Environment and data

Use Python with the dependencies in [`requirements.txt`](../requirements.txt):
DuckDB 1.5.5, NumPy 2.x, PyTorch 2.13.0, pandas, PyArrow, PyYAML, and pytest.
The experiment used the
[European Bike-Sharing Dataset](https://huggingface.co/datasets/PellelNitram/european-bike-sharing-dataset).
Raw data are deliberately excluded from Git.

The frozen cohort identity and hashes are recorded in
[`processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json`](../processed/protocol_v2_2/FINAL_DATASET_MANIFEST.json).
The final data-gate implementation is in
[`research/final_v2_2/data_gate.py`](../research/final_v2_2/data_gate.py), with
build entry points in
[`research/scripts/build_final_data_v2_2.py`](../research/scripts/build_final_data_v2_2.py)
and
[`research/scripts/build_final_data_v2_2_r2.py`](../research/scripts/build_final_data_v2_2_r2.py).

## Fixed city and temporal design

Development/source systems are Dortmund, Heidelberg, Marburg, Gießen, Cardiff,
Bilbao, Freiburg, and Göteborg. Held-out final targets are Mannheim, Innsbruck,
Glasgow, and Split.

All intervals are half-open and in UTC:

| Boundary/window | Value |
|---|---|
| Observation start, H0 | 2022-08-29 05:00:00Z |
| Source/adaptation cutoff, HF | 2023-04-16 22:00:00Z |
| Evaluation start after embargo, HT | 2023-05-11 15:00:00Z |
| Evaluation end, HE | 2023-07-15 20:00:00Z |
| Source refit window | [H0, HF) |
| Embargo window | [HF, HT) |
| Evaluation window | [HT, HE) |

The 1d, 7d, 30d, and Full target-adaptation windows end at HF. The 0d condition
performs no target fit and reuses the frozen source checkpoint. No parameter or
fitted-statistic update is permitted in [HF, HE).

## Causal coverage and preprocessing

The V2.2 data layer distinguishes verified zero demand from missing or
unobservable demand. Forecast features use strictly causal histories and fixed
eligibility/status coverage; unknown values are never silently imputed as zero.
Station cohorts, graph inputs, valid masks, and temporal boundaries are frozen
before final evaluation. The embargo separates fitting/adaptation from the
evaluation interval. See
[`COVERAGE_AND_TARGET_DEFINITION_V2_1.md`](../COVERAGE_AND_TARGET_DEFINITION_V2_1.md),
[`TEMPORAL_SPLIT_PROTOCOL_V2.md`](../TEMPORAL_SPLIT_PROTOCOL_V2.md), and
[`research/v2_2/`](../research/v2_2/) for the enforcement code and contracts.

Parameter zero-shot does not mean history-free cold start: causal target demand
histories may be model inputs, but target labels cannot be used to fit, adapt,
or select model parameters.

## Model selection and frozen settings

Development selection was completed before final-target scoring:

1. Stage 1 selected the LOG1P target transform and loss policy.
2. Stage 2 selected `GGRU_K04_H032_D00` (Graph-GRU, k=4, hidden width 32, dropout 0).
3. Stage 2B fixed `VGRU_H032_D00` as the capacity-matched Vanilla-GRU comparator.
4. Stage 3 fixed full-network target adaptation with no target-specific early stopping.
5. Stage 4 selected `grl_l50_constant`, a constant λ=0.5 source-domain GRL specification.

The frozen settings are:

| Setting | Value |
|---|---|
| Learned-model seeds | 17, 29, 43, 71, 101 |
| Graph neighbors | 4 |
| Hidden width | 32 |
| GRL λ | 0.5, constant |
| Optimizer | AdamW |
| Source learning rate | 1e-3 |
| Target learning rate | 2e-4 |
| Weight decay | 1e-4 |
| Batch | 16 eligible forecast hours |
| Global gradient clipping | 1 |
| Source pretraining | 12,000 updates |
| Target adaptation | 1d: 100; 7d: 300; 30d: 600; Full: 1,200 updates |
| Target-specific early stopping | None |
| Primary metric | Count-space MAE |
| Secondary metrics | RMSE, WAPE, station-macro MAE |
| Bootstrap | 2,000 paired temporal replicates |

## Final execution workflow

The validated source is intentionally kept as an explicit revision chain
because each layer corrects or hardens a distinct execution phase.

### 1. Dataset and job-plan sealing

The final data gate and immutable job/evaluation plans are implemented in
[`research/final_v2_2/`](../research/final_v2_2/). Compact execution-seal
manifests are retained under `research/results/`; the large data cache and job
outputs they originally bound are not included.

### 2. Source pretraining, adaptation, and prediction commitment

[`research/final_v2_2_r7_r1/`](../research/final_v2_2_r7_r1/) implements Phase A:
package checks, preflight validation, source pretraining, target-only and pooled
training, fixed-budget adaptation, prediction generation, prediction commitment,
and post-run validation. Its command dispatcher is
[`cli.py`](../research/final_v2_2_r7_r1/cli.py); controller, worker, data,
commitment, prediction, integrity, and validation modules contain the actual
steps.

Historically, the dispatcher exposed `package-check`, `preflight`, `dry-run`,
`launch`, `status`, `postrun-validate`, `predict`, `validate-predictions`,
`commit-predictions`, and `archive-phase-a` operations. Those commands require
the omitted governed inputs and should not be presented as a turnkey public
replay.

### 3. Target materialization and scoring

[`research/final_v2_2_r7_r1_pb1/`](../research/final_v2_2_r7_r1_pb1/) is the
Phase-B base. The R4 layer adds verified-snapshot and time-of-check/time-of-use
protection; R5 records the recovery logic for interrupted materialization; R6
contains the final deterministic-reference scoring correction. The final scorer
is
[`research/final_v2_2_r7_r1_pb1_r6_scoring_fix/scorer.py`](../research/final_v2_2_r7_r1_pb1_r6_scoring_fix/scorer.py).

The historical Phase-B sequence established authority, staged and authenticated
committed predictions, materialized retrospectively valid targets, scored fixed
matched cohorts, validated artifacts, and froze the summaries. Target labels
were inaccessible to fitting, prediction, and selection code.

### 4. Final validation

[`research/final_v2_2_r7_r1_pb1_r7_r1_final_validation_fix/`](../research/final_v2_2_r7_r1_pb1_r7_r1_final_validation_fix/)
contains the successful final archive-validation correction. The earlier
superseded R7 attempt is not published.

The three published summaries are already frozen; do not recompute or
reserialize them. Verify their bytes against
[`results/final_v2_2/MANIFEST.json`](../results/final_v2_2/MANIFEST.json).

## Evaluation semantics

Standalone results use equal-city macro aggregation. Controlled comparisons use
fixed intersections of retrospectively valid targets and valid committed
predictions from both methods. For learned-method contrasts, the gain is
computed within matched seed and then averaged. The paired temporal bootstrap
uses 2,000 replicates and 168-hour blocks while holding cities, methods,
checkpoints, seeds, masks, and the evaluation period fixed.

For interpretation and reported point values, see
[`RESULTS.md`](RESULTS.md). For the frozen specification hierarchy, see
[`PROTOCOL_PROVENANCE.md`](PROTOCOL_PROVENANCE.md).
