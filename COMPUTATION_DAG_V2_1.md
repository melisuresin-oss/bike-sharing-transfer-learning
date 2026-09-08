# Computation DAG V2.1

## Counting rule

A unique training job changes parameters from an initialization or parent
checkpoint. Re-evaluating identical parameters under another budget label is
not a fit. A deterministic baseline is a prediction/statistic job, not neural
training. Source-frozen checkpoints are fitted once per seed and reused across
all targets and budget columns.

Development uses seeds 17, 29, and 43. Final learned models use five seeds: 17,
29, 43, 71, and 101.

## Checkpoint graph

```text
Stage 1 reference source fit (fold × scale × seed)
├── zero-shot pseudo-target evaluation [reuse only]
└── 7-day pseudo-target fine-tune

Stage 2 ordinary grid source fit (configuration × fold × seed)
└── zero-shot pseudo-target evaluation [reuse only]

Stage 4 DANN source fit (setting × fold × seed)
└── 7-day pseudo-target fine-tune

Final ordinary source fit (seed)
├── frozen prediction for each target [one pass, reused across five budgets]
└── target fine-tune (target × nonzero budget)

Final DANN source fit (seed)
├── frozen prediction for each target [one pass, reused across five budgets]
└── target fine-tune (target × nonzero budget)

Independent final branches
├── vanilla GRU target-only (target × nonzero budget × seed)
├── Graph-GRU target-only (target × nonzero budget × seed)
└── pooled Graph-GRU (target × nonzero budget × seed)
```

## Unique-fit totals

| Phase/category | Unique fits |
|---|---:|
| Stage 1 ordinary source pretraining | 24 |
| Stage 1 7-day fine-tuning | 24 |
| Stage 2 ordinary grid source pretraining | 144 |
| Stage 4 DANN source pretraining | 72 |
| Stage 4 DANN 7-day fine-tuning | 72 |
| **Development subtotal** | **336** |
| Final vanilla-GRU target-only | 80 |
| Final Graph-GRU target-only | 80 |
| Final ordinary source pretraining | 5 |
| Final ordinary fine-tuning | 80 |
| Final pooled fitting | 80 |
| Final DANN source pretraining | 5 |
| Final DANN fine-tuning | 80 |
| **Final subtotal** | **410** |
| **Registered learned-fit total** | **746** |

Across development and final phases, ordinary source-pretraining jobs total 173
(24 + 144 + 5), target-only jobs 160, fine-tuning jobs 256
(24 + 72 + 80 + 80), pooled jobs 80, and all DANN parameter-changing jobs 229
(72 + 72 + 5 + 80). Categories overlap intentionally for DANN fine-tunes.

## Evaluation and deterministic reuse

- Five ordinary source checkpoints produce 20 unique target prediction passes;
  each pass is reused across five budget labels, creating 80 additional
  evaluation-cell reuses and no extra fit.
- DANN frozen checkpoints follow the same 20-pass/80-reuse pattern.
- Persistence and seasonal naive each require four target prediction passes,
  reused across five budget labels: 16 additional reuses per baseline.
- Historical hour-of-week averages depend on the target budget and therefore
  require 16 deterministic target-budget statistic/prediction jobs; they are
  not neural fits.
- The parameter-zero `HA_SOURCE` baseline requires four source-statistic/target-
  prediction passes, one per target, and uses no target demand label.

The final matrix therefore has 68 unique evaluation/statistic passes in these
reuse branches and 192 additional budget-cell reuses. Predictions from a
fine-tuned, target-only, or pooled checkpoint are not reused across budgets
because their parameters differ.

## Five-seed amendment

Moving final learned results from three to five seeds raises final unique fits
from 246 to 410: **164 additional fits, a 66.7% increase**. It also raises frozen
source prediction passes from 24 to 40 across ordinary and DANN branches.
Development remains at three seeds because it already spans four folds and
bounded grids; using five would raise the 336 development fits to 560 without
changing the registered decision hierarchy. Five final seeds improve stability
reporting where the substantive claims are made and are adopted for V2.1.

`unique_training_jobs_v2_1.csv` is the machine-readable group ledger. Runtime,
memory, emissions, and storage must be measured from pilot jobs before execution;
this document counts jobs and reuse only. No job has been run.
