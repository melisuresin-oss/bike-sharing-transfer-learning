# Evaluation protocol V2

## Frozen evaluation unit

Final evaluation uses Mannheim, Innsbruck, Glasgow, and Split over the identical
UTC interval `[2023-05-11T15:00:00Z, 2023-07-15T20:00:00Z)`. For each target
budget, methods are compared only on the intersection of station-hours for
which the frozen primary mask supplies a label and every method in the stated
contrast supplies a prediction. Cohort size, stations, hours, and excluded
positive departures are reported beside each result.

The final target data are not used for architecture, graph, scale/loss,
features, optimizer, schedules, stopping, or coverage thresholds. Predictions
for final evaluation keys are committed before their labels are opened.

## Metrics

For valid key set `K`, the primary metric is

\[
MAE=|K|^{-1}\sum_{(s,t)\in K}|y_{s,t}-\hat y_{s,t}|.
\]

Required secondary metrics are RMSE,
`sqrt(mean((y-yhat)^2))`; WAPE,
`sum(|y-yhat|)/sum(y)`; and station-macro MAE, the arithmetic mean of station
MAEs after requiring at least 24 valid evaluation hours for a station. A WAPE
denominator of zero is reported as undefined, never zero. Predictions from a
log1p model are inverse transformed before every metric.

The reporting hierarchy is:

1. each final target separately;
2. equal-city macro mean across the four targets;
3. station-macro results; and
4. pooled station-hour metrics as secondary descriptive output only.

No trip-weighted or station-count-weighted aggregate replaces the equal-city
primary summary.

## Replication and uncertainty

Learned models use fixed seeds 17, 29, and 43. Report the seed mean and standard
deviation for every city-method-budget cell; deterministic baselines appear
once. Primary method differences are paired within city, seed, target budget,
and matched key.

For each city, form contiguous non-overlapping seven-day UTC blocks, with any
short terminal block retained. Draw 2,000 paired block-bootstrap replicates
with replacement, recompute the method difference, and report the percentile
95% interval. This estimates temporal uncertainty for the observed city-period.
It is not inference over a population of European cities. No conventional iid
test treats four targets as independent population replicates, and no
multiple-testing-adjusted “significance” language substitutes for effect sizes.

## Transfer definitions and controlled contrasts

Transfer gain is

\[
G_{m,c,b}=MAE_{\text{matched non-transfer reference},c,b}
          -MAE_{m,c,b}.
\]

Positive values indicate lower MAE; negative values indicate negative transfer.
The reference is target-only Graph-GRU for 1/7/30/full budgets and the strongest
registered deterministic no-target-fit baseline for parameter zero-shot.
Interpretation always names the exact target, budget, cohort, scale formulation,
seed aggregation, and reference.

The ablation matrix provides the controlled contrasts: vanilla GRU versus
target-only Graph-GRU for graph contribution; target-only versus source
pretraining plus fine-tuning for source contribution; source zero-shot versus
fine-tuning for adaptation; pooled versus sequential source pretraining at a
matched budget; and ordinary versus domain-adversarial source pretraining under
the same fine-tuning schedule.

## Registered domain-shift diagnostics

Before final results are inspected, compute for every source-target pair:

- absolute difference in `log1p` pretest demand per station-active-day;
- absolute difference in `log1p` stations with departures;
- great-circle distance between station-coordinate centroids;
- absolute differences in 12h city and median-station coverage;
- development-country represented versus unseen;
- return-policy equality; and
- Jensen-Shannon distance between normalized 168-bin hour-of-week departure
  profiles, using sources' permitted fitting history and only the target's
  registered adaptation labels.

Aggregate source-target distances by the source mixture weights actually used.
Show descriptive scatterplots and Spearman rank correlations with transfer gain,
separately by budget where possible. With four final cities, fit no complex
meta-model and make no causal claim. Profile distance is unavailable for
parameter zero-shot and remains null rather than being calculated from final
labels.

## Claim rule

A finding is emphasized only when its direction is reasonably stable across
seeds, visible at individual-city level, and not reversed by the 24h or symmetric
no-maintenance sensitivities. Heterogeneous effects are reported as such. Pooled
improvement cannot establish universal European generalization.
