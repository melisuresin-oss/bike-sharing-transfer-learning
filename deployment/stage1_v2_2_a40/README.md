# Stage-1 V2.2 on UNIVERSITY_A40

Extract the ZIP into a **new dedicated folder**, preserving its paths. Run the
commands in COMMANDS.txt from that extraction root. Do not overlay an old
repository or copy local checkpoints into it. The package contains the accepted
Stage-1 fitting cache, not a panel rebuild. Original predictor and fitting NPZ
files are referenced by their accepted manifest hashes; their execution-ready
cache bytes are included. Four retrospective development-label files are
included only for the registered pseudo-target evaluations. Final-target labels,
local checkpoint payloads, and unrelated historical outputs are excluded.

The execution authorization supersedes only the local execution environment,
precision device wording, and runner implementation fields of the original
scientific manifest. The original scientific manifest, job map, selection rule,
model, trainer, and RNG definitions remain byte-for-byte unchanged. The
`stage1-v2.1` seed-domain literal is the frozen RNG formula, not a V2.1 input load.
The reconciled static adjacency is included inside the V2.2 fitting cache.

Required pre-existing TS9 environment: `%USERPROFILE%\bike_env\Scripts\python.exe`,
Python 3.10, torch 2.0.1+cu118, NumPy 1.26.4, one NVIDIA A40. No plain Python,
vendored CPU dependencies, installation, mixed precision, TF32, compilation,
architecture rewrite, or budget changes. No other Python dependencies are needed
by this Stage-1 adapter. The model/trainer and V2.2 binding modules are included.

Preflight verifies every bundle member, the immutable job map, development and
causal specification bindings, scientific fitting seals, workload, device,
Python environment, and deterministic settings. It runs synthetic CUDA forwards
only: zero optimizer steps, no scientific evaluation, no final-label access.
It records a separate immutable PASS report. Launch rechecks preflight on the
actual execution machine before dispatching jobs.

Launch runs detached from the PowerShell window. It prints its process ID and
stdout/stderr log paths. Keep TS9 powered on and its Windows session available;
detaching does not survive a reboot or logoff. The coordinator launches one
Python process per immutable fit. A restarted launch validates and skips only
completed jobs. OS locks prevent overlapping coordinators or duplicate jobs.
Each interrupted attempt remains in its UUID directory. It is never resumed:
a source restarts at its registered initialization seed and optimizer step 0;
an adaptation restarts from its completed A40 source checkpoint with the frozen
fine-tune seed and a fresh optimizer at step 0. No local CPU checkpoint is used.

There are exactly 24 source fits (12,000 updates each), 24 seven-day adaptations
(300 updates each), and 48 evaluation records. RAW/LOG1P, four folds, seeds
17/29/43, full graphs, batch 16, AdamW and final-step checkpoint rules are frozen.
Every full prediction grid is persisted before retrospective evaluation labels
are read. No loss, metric, wall-clock timing, or partial result changes a budget.

After all 48 completions, a separate read-only validator verifies checkpoints,
optimizer step counts, dependencies, prediction keys/hashes, recomputed metrics,
and runtime/input bindings. Only after PASS does the frozen Stage-1 selection
function create `stage1_decision_manifest.json` with status `FROZEN_VALIDATED`.
The coordinator prints `STAGE1_FROZEN_STOP` and exits. It cannot start Stage 2 or
Stage 2B. The standalone `validate` command also checks the frozen decision.

An A40 PASS is not claimed by the local package check. Device-specific preflight
must pass on TS9. CPU/CUDA and library releases need not produce identical
numerical trajectories; all authoritative candidates run exclusively on the
same validated A40 runtime, starting fresh, under the frozen scientific design.
