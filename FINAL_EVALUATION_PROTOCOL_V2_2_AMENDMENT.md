# Final Evaluation Protocol V2.2 Amendment

## Status and scope

Status: PROSPECTIVE_PROTOCOL_RESOLUTION_SEALED_FOR_IMPLEMENTATION.

This append-only amendment resolves the two final-evaluation ambiguities recorded
by RESEARCH_PROTOCOL_V2_2_AMENDMENT.md: the final learned-model seed count and
the aggregation of training-seed replication with the paired seven-day temporal
block bootstrap. It was written after Stage 4 development selection was frozen
and before any final-target evaluation label was accessed, any final prediction
was generated, or any final optimizer update occurred. It does not authorize
final execution; prediction-commitment, implementation, data/hash, package, and
runtime firewalls remain separate downstream gates.

No Stage-1, Stage-2, Stage-2B, Stage-3, or Stage-4 frozen decision is changed.
No method, hyperparameter, budget, city, temporal boundary, mask, feature, or
selection criterion is added in response to Stage-4 performance.

## Bound authority and chronology

The following SHA-256 bindings are authoritative for this amendment:

| Artifact | SHA-256 |
|---|---|
| RESEARCH_PROTOCOL_V2_2_AMENDMENT.md | 02a44e3aa8bf1d656d2f9de833b45072d564052e2ab3c1426bda2a6456c558d0 |
| RESEARCH_PROTOCOL_V2_1.md | 6bbf5bc39292ef9e1bf8321b90006269217b43fd0df59da25687712fc418c35c |
| COMPUTATION_DAG_V2_1.md | 48f38bc97505d622316008b8ac155fb09e2b9a9efce6f36dad7e15b35d9f53cf |
| EVALUATION_PROTOCOL_V2.md | ebde12ca004508645e876e66d1899eb23f36410751ef0d102029c602874549ee |
| TRAINING_BUDGET_PROTOCOL.md | 479d87f18095611b1d488a7f1b932c160e688d11a8776c1b0c6d85b21af81940 |
| MODEL_SPECIFICATION_V2_1.md | 430fd12c6322a102687c2a546be394bb7ef344656c1a34edf05a86beadca88bf |
| MODEL_SPECIFICATION_V2.md | cd81675bbb83386b006c28f4fc279c1e9367dc1645e68a581963d72cd840db88 |
| HISTORICAL_AVERAGE_BASELINE_AMENDMENT.md | f490478bacbc6b10493ac3b58111ce953f7d37c03113e0ab965ad18860dcc774 |
| V2.2 causal-history machine specification | c85c8fdbcab26e7239bfb4528b570b720e7d31ea5933934d9258b63ad90f9277 |
| Stage-1 V2.2 decision manifest | 8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed |
| Stage-2 V2.2 decision manifest | 9cca604ec3496a28f4440d89a9d6d1184311fc507c35f27b428a3ca768a6eeca |
| Stage-2B V2.2 decision manifest | 9d85d6df6a509013f1b5411002246aee81c254fbde078f639293976674b4f36d |
| Stage-3 V2.2 policy manifest | f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5 |
| Stage-4 prospective selection clarification | 852d8c32dd9004a8dfaa17633e285877828672d79414cf5fb67b8550c8ea7d4d |
| Returned Stage-4 frozen decision manifest | b8ab61745465fcb3c2db3ce811effaba091111a57eefe89f70d5fa5df3e1aae0 |

EVALUATION_PROTOCOL_V2.md predates the V2.1 pre-panel amendment and names
seeds 17, 29, and 43 without distinguishing development from final replication.
RESEARCH_PROTOCOL_V2_1.md later distinguishes the scopes explicitly: the same
three seeds remain for development, while final learned results use five seeds.
COMPUTATION_DAG_V2_1.md implements that later five-seed amendment, gives the
exact five-seed list, and recalculates final fits from 246 to 410. V2.2 carries
forward unchanged V2.1 design unless V2.2 explicitly supersedes it; V2.2 keeps
the three development seeds and does not replace the later final five-seed
amendment. Therefore three seeds govern development selection and five seeds
govern final learned-model fitting and evaluation.

## Frozen Stage-4 development decision

The returned Stage-4 decision is FROZEN_VALIDATED. The frozen winner is
grl_l50_constant: gradient-reversal source-domain-invariant Graph-GRU
pretraining with lambda 0.5 and a constant schedule, using the frozen
GGRU_K04_H032_D00 backbone. The selection statistic is the arithmetic mean of
the four pseudo-target fold seed-mean post-7-day count-space MAEs.

- overall MAE: 1.2941647654804875;
- population SD across the four fold seed-means, denominator 4:
  1.27787072713976;
- Bilbao: 3.503722389559748;
- Cardiff: 0.4880558047479122;
- Freiburg: 0.679111522117663;
- Göteborg: 0.5057693454966269;
- runner-up: grl_l10_constant, MAE 1.3274240522132068;
- unrounded winning margin: 0.0332592867327193.

Primary values rounded with Decimal ROUND_HALF_UP to four decimals were 1.2942
and 1.3274; they were not equal, so no tie-breaker was invoked. The sealed
ranking did not use RMSE, WAPE, station-macro MAE, discriminator accuracy,
domain loss, or any final-target quantity. Post-run validation records 72 source
fits, 72 seven-day adaptation fits, 144 completed fits, 72 post-adaptation
evaluations, zero zero-shot selection evaluations, zero verifier forward calls,
zero verifier optimizer updates, and no final-label access.

## Frozen final population and time design

Final held-out cities, in fixed reporting order, are Mannheim (195), Innsbruck
(199), Glasgow (237), and Split (617). All eight development-source cities are
used for each final source refit. The UTC, end-exclusive boundaries remain:

- H0 = 2022-08-29T05:00:00Z;
- HF = 2023-04-16T22:00:00Z;
- HT = 2023-05-11T15:00:00Z;
- HE = 2023-07-15T20:00:00Z.

Final source refits use [H0, HF). Final target adaptation-label windows end at
HF and are:

| Budget | Half-open adaptation-label window | Updates |
|---|---|---:|
| 0 | [HF, HF) | 0 |
| 1 day | [2023-04-15T22:00:00Z, HF) | 100 |
| 7 days | [2023-04-09T22:00:00Z, HF) | 300 |
| 30 days | [2023-03-17T22:00:00Z, HF) | 600 |
| full | [H0, HF) | 1200 |

The embargo is [HF, HT) and the common final evaluation interval is [HT, HE).
No parameters or fitted statistics may be updated in [HF, HE). Missing coverage
never extends a window. A required positive-budget pool that is empty is
non-estimable and stops that fit; it is not repaired with a longer window.

## Final seeds and optimization policy

Final learned-model seeds are frozen, in order, as 17, 29, 43, 71, 101. They
are training replications, not bootstrap draws. The Stage-3 policy remains
full-network fine-tuning with reset AdamW state, learning rate 2e-4, weight
decay 1e-4, batch size 16, global gradient clipping 1.0, no early stopping,
and update budgets 0, 100, 300, 600, 1200. Budget zero is checkpoint reuse,
not a fine-tuning fit. No final-target validation split, early stopping, or
performance-based retuning is permitted.

## Metrics, retained unit, and seed aggregation

The retained evaluation record is one row for each (city_id, station_id,
UTC_hour, method_id, budget_id, training_seed) with committed prediction,
retrospective observed target, and explicit validity masks. Deterministic
methods use training_seed = null. Prediction rows and keys are committed and
hashed before the retrospective final labels are joined.

Count-space MAE is primary. Required secondary metrics are RMSE, WAPE, and
station-macro MAE, exactly as defined by EVALUATION_PROTOCOL_V2.md; inverse
transformation precedes all metrics. WAPE with a zero denominator is undefined.
A station enters station-macro MAE only if its original, unresampled matched
cohort contains at least 24 distinct valid evaluation hours.

For every learned city-method-budget cell, compute each metric separately for
each of the five training seeds, then take the arithmetic mean of the five
metrics. Do not average predictions before computing a nonlinear metric. Report
the population standard deviation of the five seed-specific point metrics with
denominator 5 as replication dispersion. Deterministic methods are computed
once and are not given artificial seed variation.

## Exact paired temporal-block bootstrap

The final interval contains 1,565 UTC hours. For each city, partition [HT, HE)
into ten consecutive, non-overlapping blocks anchored at HT: blocks 0 through
8 contain 168 hours each and block 9 is the retained 53-hour interval
[2023-07-13T15:00:00Z, 2023-07-15T20:00:00Z). The short terminal block is a
full sampling unit: it is never dropped, padded, split, or lengthened.

Generate 2,000 replicates per city. Each replicate draws exactly ten block IDs
independently with replacement from {0,...,9}. A selected block contributes all
of its matched rows, with multiplicity. Because block 9 is shorter, replicate
row counts may vary; no reweighting or trimming corrects that registered fact.

Bootstrap randomness is only temporal. Training seeds remain outside the
resampling distribution. The RNG contract is:

1. namespace string
   FINAL_EVALUATION_V2_2|TEMPORAL_BLOCK_BOOTSTRAP|20260908;
2. for each city, append |<city_id>, encode as ASCII, compute SHA-256, take the
   first eight digest bytes as an unsigned big-endian integer;
3. initialize Python standard-library random.Random(city_seed) independently
   for each city;
4. generate 20,000 calls to randrange(10), consumed row-major as 2,000 rows
   of ten block IDs.

The derived city seeds are Mannheim 195: 12648532355080271797; Innsbruck 199:
2634071097790620903; Glasgow 237: 14234234470032652229; and Split 617:
6032700276159629083.

For a learned method and each city/bootstrap replicate, compute the metric
separately for each training seed on the resampled rows and then arithmetic-mean
the five seed-specific metrics. For a deterministic method compute one metric.
For an equal-city macro replicate, arithmetic-mean the four city replicate
statistics with weight 1/4 each. The same replicate number combines the four
independently seeded city streams. The primary summary is never weighted by
station count, valid-row count, or trip volume.

The 95% interval is the empirical percentile interval from the 2,000 replicate
statistics: quantiles 0.025 and 0.975 using linear interpolation equivalent to
numpy.quantile(values, [0.025, 0.975], method="linear"). It represents temporal
block-resampling uncertainty conditional on the four fixed cities, registered
training seeds/checkpoints, methods, budgets, masks, and evaluation period. It
does not represent training-seed uncertainty or inference to a population of
cities. Seed dispersion is reported separately.

## Pairing, missingness, and transfer contrasts

For every stated pairwise contrast, first form the exact intersection of keys
whose retrospective final target is valid and for which both compared methods
have valid committed predictions. Use this same fixed matched cohort for both
point estimates and all replicates. Within a city and replicate, the identical
sampled block-ID vector is reused for both methods, every training seed, and all
reported metrics. Across contrasts, the pre-generated city/replicate block
vectors are also reused. Never replace unknown labels, unavailable lag
predictions, or missing predictions with zero.

For error metric L, define the paired gain of method A over reference B as
G(A,B) = L(B) - L(A) on the matched cohort. Positive values favor A. With two
learned methods, compute the paired gain separately for each matched training
seed and then average the five gains. With a deterministic reference, reuse its
single resampled metric against each learned seed before seed averaging. The
equal-city macro paired gain is the arithmetic mean of the four city gains.

If a point cohort is empty, the contrast is unavailable. If any of the 2,000
replicates has no valid matched row, or a requested metric is undefined (for
example WAPE denominator zero), that cell's bootstrap interval is reported as
undefined with its reason; replicates are not silently removed, redrawn, or
replaced.

Persistence is exactly the observed elapsed-UTC y[t-1h]; seasonal naive is
exactly observed y[t-168h]. Under V2.2, either prediction is unavailable, not
zero, wherever its causal as-of lag mask is false. Availability counts and
matched-cohort sizes must accompany their metrics. HA_SOURCE remains the
registered parameter-zero historical baseline; the earlier evaluation rule
that describes the parameter-zero transfer reference as the strongest
registered deterministic no-target-fit baseline remains a prospectively fixed
evaluation rule, not permission to alter methods. It is selected only among
HA_SOURCE, persistence, and seasonal naive by the lowest equal-city macro
count-space MAE after prediction commitment; an exact numerical tie is resolved
lexicographically by method ID. The selected reference is used consistently for
all parameter-zero transfer-gain reporting and is not a modelling decision.

## Frozen final method set and fit DAG

The final method set is limited to the already registered families:

1. HA_SOURCE at budget zero and HA_TARGET at 1-day, 7-day, 30-day, and full;
2. persistence and seasonal naive at all five budget labels by prediction reuse;
3. frozen VGRU_H032_D00 target-only Vanilla-GRU at four nonzero budgets;
4. frozen GGRU_K04_H032_D00 target-only Graph-GRU at four nonzero budgets;
5. ordinary eight-source pretrained GGRU_K04_H032_D00, frozen prediction reused
   across all budgets;
6. ordinary source-pretrained Graph-GRU plus Stage-3 fine-tuning at four
   nonzero budgets;
7. pooled source-plus-target Graph-GRU from scratch at four nonzero budgets;
8. winning grl_l50_constant source-domain-invariant Graph-GRU, frozen prediction
   reused across all budgets; and
9. the same winning source-domain-invariant checkpoint plus Stage-3 fine-tuning
   at four nonzero budgets.

The GRL method is a transfer/negative-transfer mitigation mechanism and ablation,
not an architecture-novelty claim and not a claim of canonical DANN reproduction.

With 5 seeds, 4 final targets, and 4 nonzero budgets, final neural fits are:

| Category | Arithmetic | Unique fits |
|---|---:|---:|
| target-only Vanilla-GRU | 5*4*4 | 80 |
| target-only Graph-GRU | 5*4*4 | 80 |
| ordinary source pretraining | 5 | 5 |
| ordinary source adaptation | 5*4*4 | 80 |
| pooled Graph-GRU | 5*4*4 | 80 |
| GRL source pretraining | 5 | 5 |
| GRL source adaptation | 5*4*4 | 80 |
| total | | 410 |

Equivalently: 10 source fits, 160 adaptation fits, 160 target-only fits, and 80
pooled fits. Ordinary transfer accounts for 85 fits and GRL transfer for 85.
Deterministic baselines are not neural fits.

There are 400 unique evaluation passes from target-specific learned checkpoints,
40 frozen-source prediction passes (20 ordinary and 20 GRL), and 28 deterministic
statistic/prediction passes (4 persistence, 4 seasonal, 16 HA_TARGET, and 4
HA_SOURCE): 468 unique evaluation/statistic passes. Frozen ordinary and GRL
predictions each create 80 additional budget-label reuse cells; persistence and
seasonal each create 16, for 192 reuses and 660 budget-labelled reporting cells.

## Firewall and limitation

At seal creation:

- final_target_labels_accessed = false;
- final_experiment_started = false;
- optimizer_updates = 0;
- model_forward_calls_on_final_labels = 0.

The eventual implementation must commit sorted prediction keys, prediction
hashes, checkpoint/code/data hashes, and the evaluation plan before any final
retrospective label join. The final-label reader must be physically unavailable
to fitting, prediction generation, method selection, or hyperparameter code.
No final result can reopen Stages 1-4 or this amendment.

The benchmark retains the V2.2 IDEALIZED_COUNT_FEED_BENCHMARK limitation:
reconstructed hourly departure counts are assumed available at hour end, raw
prefix replay/ingestion history is not validated, and claims must be conditioned
on that assumed feed plus the frozen cohort/static inputs. This is not measured
real-time ingestion performance or universal deployment validity.

## Downstream gate

This amendment resolves the seed-count and seed-by-bootstrap blockers. The next
authorized work is implementation and read-only verification of the final
execution plan, prediction-commitment mechanism, data/cache/hash bindings, and
strict final-label firewall. It must create no final prediction, perform no fit,
and open no final label until a separate execution authorization is sealed.

