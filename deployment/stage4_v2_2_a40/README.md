# Stage 4 V2.2 — UNIVERSITY_A40

This package runs only the prospectively sealed Stage-4 development selection:
72 source fits, 72 seven-day adaptations, 72 post-adaptation evaluations.
It never starts the final held-out experiment. Stage 1, 2, 2B and 3 remain frozen.

Use the accompanying delivery runbook and its external ZIP/manifest hashes.
Required TS9 executable: `$env:USERPROFILE\bike_env\Scripts\python.exe`.
Do not substitute a plain `python` command or a local CPU checkpoint.
The strict preflight requires one actual NVIDIA A40, the validated bike_env
runtime, deterministic algorithms, CuBLAS `:4096:8`, both TF32 switches off,
cuDNN benchmark off and cuDNN deterministic on. Preflight uses synthetic
forward passes only; it does not perform a scientific optimizer update.

The launcher takes `-Mode package-check`, `preflight`, `validate-existing`,
`launch`, `validate`, or `freeze`, with `-ManifestSha256` and `-Workers 4`.
`launch` detaches a coordinator, which repeats strict preflight and starts
separate immutable job processes. It stops dispatch on a genuine failure.
Success requires exit code zero and a validated hash-committed completion.
The coordinator finishes after all 144 jobs; then run read-only validation,
freeze the decision, and STOP. It does not launch another scientific stage.

Recovery uses the same launcher with `-Mode launch -Resume`. Valid completed
jobs are verified and skipped. Partial attempts remain append-only and restart
in a fresh attempt directory at update zero. Corrupt or half-committed
completions fail closed; preserve them and investigate rather than deleting
them or silently treating them as successful.

All recurrent predictors come from the accepted V2.2 origin-frozen cache.
The sole legacy-data exception is hash-reconciled raw geographic adjacency.
The source domain mask is the current training example's permitted target
mask from the sealed fit snapshot at HD. Historical observation masks and
demand values are never pooling weights. Source checkpoint probes stay in
source cities. Only the matching source forecasting weights initialize its
target adaptation; the discriminator and optimizer state are discarded.

Selection follows the unchanged prospective clarification: seed means within
each fold, equal mean of four folds, primary equality at four decimal places
using Decimal ROUND_HALF_UP, then unrounded population SD (N=4), smaller lambda,
and constant before linear. All unrounded values are retained. Secondary
metrics and discriminator diagnostics cannot affect selection.

`research.stage4_v2_2.tests` is a synthetic implementation suite. Its explicit
detached-startup fixture mode cannot carry production authorization and writes
only under `tmp/stage4_v2_2_synthetic_tests`. Fixture checkpoints and metrics
are not scientific artifacts. Never transfer them into production outputs.

Open downstream issues remain `FINAL_SEED_COUNT_UNRESOLVED_3_VS_5` and
`FINAL_SEED_BOOTSTRAP_AGGREGATION_UNRESOLVED`.
