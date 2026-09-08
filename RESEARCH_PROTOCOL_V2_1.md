# When Does Cross-City Graph Transfer Help?

## V2.1 pre-panel research protocol

### Objective and scope

The study evaluates when cross-city graph-temporal transfer helps or hurts
hourly station-level departure forecasting under 0/1/7/30/full target-label
budgets. V2.1 is a methodological amendment completed before panel construction:
no forecasting model has been trained and no forecasting performance has been
inspected.

The contribution is a controlled multi-city empirical study of graph structure,
ordinary source pretraining, target fine-tuning, pooling, and simple domain
alignment under a label-independent coverage protocol. Graph-GRU and
domain-adversarial training are established mechanism classes, not architectural
novelty claims.

### Population and roles

The primary population remains Bilbao, Glasgow, Freiburg, Mannheim, Dortmund,
Cardiff, Split, Göteborg, Innsbruck, Heidelberg, Marburg, and Gießen. The frozen
development sources are Dortmund, Heidelberg, Marburg, Gießen, Cardiff, Bilbao,
Freiburg, and Göteborg. Pseudo-target folds are Bilbao, Cardiff, Freiburg, and
Göteborg. Sealed final targets are Mannheim, Innsbruck, Glasgow, and Split.

These cities were characterized before modelling. Pretest metadata/status and
pretest demand summaries were used for eligibility and deterministic diversity
stratification. Final-period status availability—not final demand or forecasting
performance—was used to define the common cohort/window. “Sealed” means final
forecasting performance and evaluation labels cannot influence modelling
decisions. The study is few-shot target-label adaptation, not true new-city cold
start.

### Coverage amendment

All 799 finite-coordinate stations with pretest 12-hour bracketed-lifetime
coverage `q_s >= 0.50` remain eligible. The pre-`HF` audit compared:

- A: 12h station bracket + 50% city bracketing + same-hour status event;
- B: 12h station bracket + 50% city bracketing; and
- C: Rule B + network status evidence within six hours on both sides.

Rule B is primary. Relative to A it retained 576 additional city-hours and
25,172 station-hours, of which 25,162 were zero/no-trip and only 10 positive.
Equal-city nighttime retention increased from 95.19% to 98.25%. The same-hour
condition therefore selected against quiet hours in a change-triggered table.
Rules A and C remain sensitivities.

The target is the matched departure count for a station and UTC hour. A no-trip
key passing station and city bracketing is a “coverage-qualified zero-demand
station-hour”; an uncovered key remains null. Positive departures outside the
mask are excluded symmetrically and reported. The 24-hour bracket and symmetric
no-maintenance analyses remain required. No demand value makes a coverage key
active.

### Temporal design

UTC, end-exclusive boundaries remain:

- `H0 = 2022-08-29T05:00:00Z`;
- `HD = 2023-02-15T22:00:00Z`;
- `HF = 2023-04-16T22:00:00Z`;
- `HT = 2023-05-11T15:00:00Z`; and
- `HE = 2023-07-15T20:00:00Z`.

Pseudo-target evaluation is `[HD,HF)`. Final evaluation is the unchanged common
`[HT,HE)` cohort. Final adaptation labels end at `HF`; the 593-hour embargo is
not fitted. No final-target validation or early stopping is allowed, and every
budget uses the same evaluation interval.

### Frozen primary features

Every learned model receives the fixed V2.1 set: `log1p` demand lags 1–24 with
observation indicators, a `t-168h` `log1p` lag with indicator, transformed rack
capacity with validity, and deterministic local hour/weekday/weekend/UTC-offset
calendar terms. Coordinates construct the graph but are not node features.

Weather, holidays, events, realized future covariates, status values, city IDs,
and target-fitted normalization statistics are excluded. No weather source with
forecast-issue timestamps and a leakage audit exists; realized future weather
is prohibited.

### Ordered development selection

Development proceeds once in five stages:

0. freeze the reference Graph-GRU (`k=8`, hidden 64, dropout 0.10, AdamW,
   source learning rate `1e-3`, 12,000 source steps, 7-day fine-tuning learning
   rate `2e-4` for 300 steps, seeds 17/29/43);
1. compare only raw-count versus log1p target/loss at zero-shot and 7-day
   pseudo-target regimes and freeze the winner;
2. with scale/loss frozen, exhaust `k={4,8,16}`, hidden `{32,64}`, and dropout
   `{0,0.1}` using zero-shot pseudo-target folds, then freeze the ordinary
   backbone;
3. freeze all-parameter target fine-tuning at 0/100/300/600/1,200 updates for
   0/1/7/30/full budgets; and
4. with ordinary settings frozen, select only DANN `lambda={0.01,0.10,0.50}`
   and constant versus 25%-warm-up schedules in 7-day pseudo-target folds.

Tie-breakers and seal records are specified in
`DEVELOPMENT_SELECTION_PROTOCOL.md`. A later stage cannot reopen an earlier one.

### Backbone and controlled comparisons

The shared backbone is a one-step Graph-GRU with symmetric geographic-kNN,
Gaussian distance weights, GCN-style symmetric normalization, and propagation
inside GRU gates. It is explicitly distinguished from canonical T-GCN and from
DCRNN's directed diffusion/sequence-to-sequence design. A separate vanilla GRU
provides the genuine non-graph control.

Required families remain historical hour-of-week average, persistence,
seasonal naive, vanilla GRU target-only, Graph-GRU target-only, frozen ordinary
source transfer, ordinary source plus fine-tuning, pooled source-plus-target,
frozen DANN transfer, and DANN plus fine-tuning.

The historical-average family is frozen by
`HISTORICAL_AVERAGE_BASELINE_AMENDMENT.md`. `HA_TARGET` uses only permitted
target-budget labels and the registered six-level station/target/source
fallback hierarchy. `HA_SOURCE` is the distinct parameter-zero baseline and
uses no target demand label. Target-city fallbacks are equal-station means;
source fallbacks are equal-station within city and equal-city across sources.
All bins use validated local—not UTC—hour-of-week.

### Pooled versus sequential exposure

Both mechanisms receive 12,000 source updates and exactly 0/100/300/600/1,200
target updates for 0/1/7/30/full budgets. Pooled schedules contain the exact
source and target token counts in seeded shuffled order; sequential training
places all source steps before target steps and resets optimizer state. Batch
size, masks, target-label set, source-city balance, total steps, learning rates,
and seeds are matched. Target oversampling in pooled training is deliberate and
disclosed; update order and optimizer-state history remain the mechanisms that
differ.

### Evaluation and seeds

MAE remains primary; RMSE, WAPE, and station-macro MAE are required. Report each
final target, equal-city macro, station macro, then pooled station-hours as
secondary. Final learned results use seeds **17, 29, 43, 71, and 101**.
Development remains at three seeds because four folds and bounded grids already
yield 336 development fits; five would raise that to 560. The five-seed final
amendment raises final fits from 246 to 410.

Use paired seven-day temporal block-bootstrap intervals within each observed
city-period. They quantify temporal uncertainty, not inference over a population
of cities. Do not run iid tests across four targets.

Positive/negative transfer uses matched target, budget, station-hour cohort, and
seed comparisons. Domain-shift demand-profile diagnostics are budget-limited:
parameter zero-shot has no target demand profile; 1-day and 7-day budgets may use
only a normalized 24-bin local-hour profile; 30-day and full budgets may use a
168-bin local hour-of-week profile. Other registered metadata/coverage/scale
distances remain descriptive, with no complex four-city meta-model.

### Computation

The unique-fit DAG contains 336 development fits and 410 final fits, for 746
future learned fits. Frozen ordinary and DANN source checkpoints are fitted once
per seed, not once per target-budget column. Their prediction branches plus
deterministic baselines contain 64 unique evaluation/statistic passes and 192
additional budget-cell reuses. These are counts, not completed executions.

## A. Frozen decisions

- 12-city population, eight development sources, four pseudo-target folds, and
  four sealed final targets;
- `q_s >= 0.50`, finite coordinates, 799-station roster;
- 12-hour primary station bracket and bracket-only 50% city rule;
- UTC boundaries and common final interval;
- fixed no-weather primary feature set;
- Stage 0 reference settings, bounded Stage 2 grid, fixed fine-tuning updates,
  and bounded Stage 4 DANN grid;
- pooled/sequential update accounting;
- metrics, reporting hierarchy, bootstrap method; and
- three development seeds and five final seeds.

## B. Still-unresolved decisions

Only registered development outcomes remain unresolved:

1. raw-count versus log1p target/loss winner at Stage 1;
2. one of the 12 ordinary Graph-GRU configurations at Stage 2; and
3. one of six DANN coefficient/schedule settings at Stage 4.

These are intentionally unresolved before the model-ready panel exists. No
city, station threshold, feature subset, temporal boundary, target budget,
fine-tuning schedule, or mask rule remains open to performance selection.

## C. Remaining approval gates

Before panel construction:

1. approve the V2.1 bracket-only amendment and frozen feature manifest;
2. hash and reconcile `city_roles_v2.csv`, the 799-station roster, UTC
   boundaries, and coverage-rule configuration; and
3. preserve the observability audit JSON/CSV provenance and pinned Parquet
   revision.

Before any model fit:

4. validate unique panel keys, trip reconciliation, null-versus-zero behavior,
   DST conversion, causal lag timestamps, and mask symmetry;
5. validate graph/feature tensor contracts and exact optimizer-token schedules;
   and
6. seal Stage 0 inputs, code revision, seed derivation, and development output
   schema.

Before final evaluation labels are opened:

7. complete Stages 1–4 on pseudo-targets, freeze their manifests, refit source
   checkpoints, and commit final prediction keys and hashes.

## D. Panel-construction permission

**Panel construction is now methodologically permitted after acceptance of this
V2.1 amendment.** The observability problem was serious but is resolved by
removing the activity-selective same-hour condition; it is not an outstanding
blocker. Construction must implement Rule B and the fixed feature schema, then
pass gates 2–5 before any training. Panel construction does not authorize model
fitting or inspection of forecasting performance.

Permitted claims remain limited to this empirical population, periods, budgets,
and evaluated methods. Prohibited claims include novelty of graph-inside-GRU or
DANN, first-method status, true cold-start generalization, universal European or
causal conclusions, superiority over unevaluated methods, and complete latent
demand ground truth.
