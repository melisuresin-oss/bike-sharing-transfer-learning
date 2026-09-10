# Final V2.2 research implementation

This directory contains the scientific implementation for the final 12-city
study. The code is preserved in its validated revision structure; historical
execution layers have not been flattened or rewritten for presentation.

## Core implementation

- `data/`: budget sampling and protocol-bound dataset loaders.
- `graphs/`: fixed geographic station-graph construction and manifests.
- `models/`: Vanilla-GRU and Graph-GRU architectures.
- `training/`: batching, losses, deterministic training, and checkpoint logic.
- `evaluation/`: count-space evaluation metrics.
- `v2_2/`: causal-history, label-isolation, contract, and artifact primitives.
- `final_v2_2/`: final data-gate and immutable job-plan construction used by
  the published final-data manifest.

## Development selection stages

- `stage1_v2_2/`: target transform and loss selection.
- `stage2_v2_2/`: Graph-GRU development selection.
- `governance/`: Stage 2B, Stage 3, and Stage 4 frozen-policy checks.
- `stage4_v2_2/`: GRL-based multi-source domain-invariant pretraining.

The compact frozen selection and execution manifests needed by the final code
are retained under `research/results/`. Large job outputs, checkpoints,
predictions, raw snapshots, and deployment bundles are intentionally absent.

## Final execution revision chain

The final workflow is represented by the following dependency-preserving
layers:

1. `final_v2_2_r7_r1/` — final Phase-A training and prediction implementation.
2. `final_v2_2_r7_r1_pb1/` — Phase-B materialization and scoring base.
3. `final_v2_2_r7_r1_pb1_r4/` — verified-snapshot/TOCTOU correction.
4. `final_v2_2_r7_r1_pb1_r5_recovery/` — recovery from the interrupted
   materialization caused by a missing dependency.
5. `final_v2_2_r7_r1_pb1_r6_scoring_fix/` — deterministic zero-reference
   scoring correction.
6. `final_v2_2_r7_r1_pb1_r7_r1_final_validation_fix/` — successful final
   archive validation correction.

The superseded R7 validation attempt is not published. See
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the scientific
workflow and the distinction between source-level reproduction and the exact
historical governed execution.

## Protocols and results

The final evaluation and GRL clarification documents are stored at repository
root because the frozen implementation binds those exact paths and hashes.
The public, compact frozen result summaries are under
[`results/final_v2_2/`](../results/final_v2_2/), with interpretation in
[`docs/RESULTS.md`](../docs/RESULTS.md).
