# Temporal split protocol V2

## UTC index and frozen boundaries

All intervals are UTC, left-closed, and right-open. A forecast for hour `h` may
use only inputs strictly earlier than `h`.

| Symbol | Boundary | Purpose |
|---|---|---|
| `H0` | `2022-08-29T05:00:00Z` | common fitting-history start |
| `HD` | `2023-02-15T22:00:00Z` | pseudo-target adaptation/evaluation boundary |
| `HF` | `2023-04-16T22:00:00Z` | final adaptation cutoff and seal |
| `HT` | `2023-05-11T15:00:00Z` | common final evaluation start |
| `HE` | `2023-07-15T20:00:00Z` | common final evaluation end |

`H0`, `HD`, and `HF` remain supported by all 12 cities. The old
`HE=2023-07-15T21:00:00Z` is not retained: over `[HF, old HE)`, every city has
only 85.27%--85.97% observable 12h city-hours, below the registered 90% check.

Roles were frozen first. Using status only, enumerate all contiguous subintervals
of `[HF, old HE)` and choose the longest interval on which every sealed final
target has at least 90% observable 12h city-hours; ties choose the earliest
start. This yields `[HT,HE)`, **1,565 hours** (65 days 5 hours), with final-target
coverage 90.03%--90.29%. No trip count or forecast enters this boundary rule.

The **593-hour embargo** `[HF,HT)` is not a fitting period. It prevents boundary
selection from quietly enlarging a target budget and supplies causal history
before evaluation.

## Development phase

The pseudo-targets are Bilbao, Cardiff, Freiburg, and Göteborg. For each fold:

1. exclude the pseudo-target from source fitting;
2. fit on the other seven development cities using `[H0,HD)`;
3. adapt to the pseudo-target only within its registered budget ending at `HD`;
4. evaluate on `[HD,HF)` (1,440 hours); and
5. aggregate selection MAE with equal pseudo-target weight.

Architecture, scale/loss formulation, graph `k`, optimizer, step/epoch schedule,
fine-tuning policy, adversarial settings, features, and station threshold are
frozen before final-target labels are available.

## Final phase

After development is frozen:

1. refit source models on all eight development cities using `[H0,HF)`;
2. build each final graph from its frozen eligible station roster and coordinates;
3. adapt only on the declared pre-`HF` target budget;
4. perform no weight or fitted-statistic update in `[HF,HE)`; and
5. evaluate Mannheim, Innsbruck, Glasgow, and Split on identical keys in
   `[HT,HE)` for every budget.

Final-target labels in `[HT,HE)` are read only after predictions for their keys
are committed. Labels in the embargo may be used as strictly causal lag inputs
after their hour ends, identically for all budgets, but never for fitting,
selection, normalization, or early stopping.

## Nested target-label budgets

| Budget | Pseudo-target window ending `HD` | Final-target window ending `HF` |
|---|---|---|
| parameter zero-shot | none | none |
| 1 day | `[2023-02-14 22:00, HD)` | `[2023-04-15 22:00, HF)` |
| 7 days | `[2023-02-08 22:00, HD)` | `[2023-04-09 22:00, HF)` |
| 30 days | `[2023-01-16 22:00, HD)` | `[2023-03-17 22:00, HF)` |
| full history | `[H0,HD)` | `[H0,HF)` |

Budgets are elapsed time and are never extended to replace masked labels. Report
the actual retained station-hour count per city and budget. No final-target
validation tail or target-specific stopping rule is allowed.

## Coverage diagnostics and failures

After applying the frozen 0.50 station rule, all four final targets pass the
`[HT,HE)` 12h check. Two development cities do not: **Bilbao** has 89.84% and
**Gießen** 89.78% observable city-hours, although their median station coverage
exceeds 90%. This does not alter roles because development decisions use
`[HD,HF)`, where both passed the pretest gate. It must be reported if those
cities are shown descriptively over the final interval.

## Leakage invariants

- The city and station populations are frozen from pre-`HF` evidence.
- Final demand labels do not choose `HT` or `HE`; status coverage alone does.
- Every lag timestamp is earlier than its prediction timestamp.
- All budgets share `[HT,HE)` and the same method-common keys.
- Final targets never control stopping or schedule length.
- No result-driven rerun, city replacement, or boundary change is permitted.
