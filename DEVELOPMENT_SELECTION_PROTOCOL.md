# Development selection protocol

## Purpose and invariant

All selection uses the four pseudo-target folds—Bilbao, Cardiff, Freiburg, and
Göteborg—and only their registered `[HD,HF)` development evaluation labels.
Within a fold the pseudo-target is absent from source fitting. Final forecasting
performance and final evaluation labels remain sealed. Stages execute once, in
order; a later stage cannot reopen an earlier decision.

## Stage 0 — freeze the scale-reference configuration

Before raw versus log1p is compared, freeze this reference Graph-GRU:

| Component | Fixed value |
|---|---|
| graph | symmetric union geographic 8-NN, Gaussian distance weights, unit self-loops, symmetric normalization |
| recurrent backbone | one Graph-GRU layer, hidden size 64 |
| dropout | 0.10 on encoded inputs and recurrent output before the head; no recurrent-gate dropout |
| history and horizon | 24 elapsed hourly inputs plus the `t-168h` feature; one-hour forecast |
| batch | one city per update, 16 uniformly sampled eligible anchor hours, full station graph |
| source sampling | each available source city equally likely |
| optimizer | AdamW, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4` |
| source learning rate | `1e-3` |
| source schedule | exactly 12,000 optimizer steps, no target-controlled stopping |
| gradient clipping | global norm 1.0 |
| 7-day fine-tuning | all parameters, AdamW state reset, learning rate `2e-4`, exactly 300 target-only steps |
| zero-shot | source checkpoint with no target update |
| seeds | 17, 29, 43 |

The final-step checkpoint is used; there is no pseudo-target-specific early
stopping. Data-order and initialization seeds are recorded separately but
derived deterministically from the registered seed.

## Stage 1 — freeze scale and loss

Run only the registered raw-count masked-MAE and log1p-target masked-MAE
comparison with the Stage 0 configuration, at parameter zero-shot and 7-day
fine-tuning. The primary selection statistic is count-space MAE averaged equally
over the four pseudo-target-by-regime cells after seed averaging. A tie to four
decimals chooses lower across-cell standard deviation, then raw counts. Freeze
the winner. No other loss, normalization, or transformation enters Stage 1.

## Stage 2 — select the ordinary backbone

With scale/loss frozen, exhaust this 12-configuration grid:

- geographic `k in {4,8,16}`;
- hidden size in `{32,64}`; and
- dropout in `{0.00,0.10}`.

Optimizer, learning rate, batch, 12,000-step source schedule, features, graph
weighting, and seeds remain as Stage 0. Each configuration is fitted in all four
pseudo-target folds and evaluated parameter zero-shot; no fine-tuning decision
is involved yet. Select equal-fold count-space MAE after seed averaging. Ties to
four decimals choose lower fold dispersion, then smaller hidden size, smaller
`k`, and lower dropout. Freeze the ordinary Graph-GRU configuration.

## Stage 3 — freeze target fine-tuning

V2.1 does not performance-select a fine-tuning grid. Freeze full-network
fine-tuning with reset AdamW state, learning rate `2e-4`, weight decay `1e-4`,
batch size 16, gradient clip 1.0, and no early stopping. Exact target-only update
counts are:

| Target-label budget | Updates |
|---|---:|
| 0 days | 0 |
| 1 day | 100 |
| 7 days | 300 |
| 30 days | 600 |
| full pre-cutoff history | 1,200 |

Labels are sampled with replacement only within the elapsed-time budget; causal
pre-budget history may supply inputs but never extra adaptation targets.

## Stage 4 — select domain-adversarial settings

Keep the ordinary backbone, scale/loss, features, optimizer, source steps, and
Stage 3 fine-tuning frozen. The discriminator is a two-layer MLP with hidden size
64, ReLU, dropout 0.10, and source-city softmax. Exhaust only:

- maximum gradient-reversal coefficient `lambda in {0.01,0.10,0.50}`; and
- schedule in `{constant, linear warm-up over the first 25% of source steps}`.

Each of six settings is source-fitted and then 7-day fine-tuned in every
pseudo-target fold using seeds 17, 29, and 43. Select equal-fold count-space MAE;
ties choose lower dispersion, smaller `lambda`, then constant schedule. The
domain discriminator is removed during target fine-tuning and inference.

## Seal record

At each stage record the candidate set, seed-level development table, selection
statistic, tie-break path, chosen manifest hash, code revision, and timestamp.
After Stage 4, freeze all development decisions before final model fitting. Any
change requires a dated amendment made without final forecasting results.
