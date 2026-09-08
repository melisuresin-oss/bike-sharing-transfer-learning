# V2.2 causal-history protocol amendment

Status: authoritative specification; scientific implementation and execution have not occurred.
Count policy: **IDEALIZED_COUNT_FEED_BENCHMARK**.

This append-only amendment supersedes conflicting V2.1 predictor availability,
fit-membership, eligibility-recomputation and upstream-selection bindings. It
does not rewrite V2.1, invalidate its execution provenance, or authorize data
construction, training, evaluation, predictions, Stage 4 implementation, or final
execution. The accepted forensic finding is LOOKAHEAD_FINDING = CONFIRMED.
The V2.1 experiments faithfully executed their then-registered specification;
the causal-input validity issue was discovered afterward.

The companion authority is
`research/results/causal_history_spec_v2_2/v2_2_specification.json`.
The prose and machine-readable contract must agree; any contradiction is a hard
stop, not permission to select whichever definition is convenient.

## 1. Fixed cohort and infrastructure conditioning

Preserve exactly the frozen 799 station identities and their city/node ordering.
The cohort is retrospectively selected research-design conditioning, not
prospective station discovery. The identity list is authoritative; recomputed
eligibility scores are descriptive and must never add or remove stations.
Do not claim that every retained identity satisfies a re-evaluated exact-boundary
eligibility threshold. Historical q scores remain provenance, not predictors.

The fixed-cohort/static manifest freezes station IDs, coordinates, rack capacity,
city mapping and timezone from the supplied infrastructure metadata. Historical
publication timestamps are unavailable; these fields are assumed supplied and
applicable throughout this benchmark. This is a conditioning assumption, not
verified historical metadata availability. Coordinates construct geographic
edges, capacity uses the registered positive-capacity log1p/validity pair, and
IDs are keys, not target-specific learned embeddings. No demand-derived filling,
future lifetime, q score or future maintenance state may enter static features.
Frozen graph nodes remain present even when historical coverage is unknown.

## 2. Count availability: an explicit idealized feed

For UTC hour h = [s,e), e = s + 1 hour, and evidence cutoff a:

    Q_i(h;a) = 1[e <= a].

The benchmark assumes finalized reconstructed hourly departure counts d_i(h)
are supplied at hour end. This is not measured ingestion latency, validated raw
stream replay, or a claim that inferred trips were available online at time_start.
Observed zero means zero accepted reconstructed departures in a coverage-qualified
completed hour; it does not prove that every real-world trip was captured.

The export enforces duration strictly below 8 hours. Under chronological,
append-only processing with stable mappings, this supports an event-time release
candidate time_start + duration and a sufficient hourly bound e + 8 hours.
The public export lacks the raw observations, predecessor buffers and ingestion
history needed to validate empirical prefix reconstruction. These facts are
limitations only: **8 hours is not an active V2.2 predictor or fitting delay**.
Code evidence: `etl/schema_definition.sql:1429-1483,1675-1694` and
`etl/data_pipeline.py:248-285,345-377`.

## 3. Forecast-origin status coverage

All comparisons use exact UTC timestamps at microsecond precision. For a
predictor at origin t, use only status events with event timestamp <= t. In the
absence of receipt timestamps, assume status availability at its event time;
do not label this measured ingestion availability. Condition invariance claims
on the frozen cohort/static inputs and the assumed count feed.

Let S_i(a) be station i's status timestamps restricted to <= a. For h = [s,e):

    p_i(h;a) = max {u in S_i(a): u <= s}
    n_i(h;a) = min {u in S_i(a): u >= e}
    B_i(h;a) = 1[p and n exist AND 0 <= s-p <= 12h AND 0 <= n-e <= 12h].

Missing p or n makes B false. At prediction, a = t; hence
e <= n <= min(t,e+12h). No next certificate by t means unknown, even if there
is a recent previous status or a positive recorded count. Do not substitute
backward-only, carried-forward or event-only history.

Define a status-evidenced lifetime, not physical installation/removal dates:

    f_i(a) = min S_i(a); l_i(a) = max S_i(a)
    L_i(h;a) = 1[S_i(a) nonempty AND f_i(a) <= s <= l_i(a)].

Both extrema are prefix extrema. No full-history minimum/maximum may be used to
make an earlier predictor decision. A station with no prefix evidence has L=0.
For the frozen roster R_c of city c:

    D_c(h;a) = sum_{j in R_c} L_j(h;a)
    N_c(h;a) = sum_{j in R_c} L_j(h;a) * B_j(h;a)
    G_c(h;a) = 1[D_c > 0 AND 2*N_c >= D_c]
    O_i(h;a) = L_i(h;a) * B_i(h;a) * G_c(h;a).

The numerator and denominator contain only frozen-roster stations evidenced
under the same prefix. D=0 means unobserved. Integer 2*N>=D implements the exact
>=50% threshold, including odd denominators. No other city's evidence participates.
As-of masks may change non-monotonically at later origins; freeze each origin's
representation rather than rewriting it when later evidence arrives.

## 4. One historical-feature function

    M_i(h;t) = O_i(h;t) * Q_i(h;t)
    F_i(h;t) = (log1p(d_i(h)), 1) if M=1; otherwise (0,0).

Observed positive is (log1p(d),1); observed zero is (0,1); unknown is (0,0).
There is no positive-count coverage exception. Placeholder zero is never an
observed-zero claim. Masks are Boolean before tensor casting; zero placeholders
are exact positive numeric zero. Missing events and out-of-history cells produce
unknown, not NaN-valued tensor channels.

For each k=1..24 use h=[t-k hours,t-(k-1) hours). X_hist stores the value and
indicator; M_hist is exactly the indicator channel. X_week uses the same F with
k=168, including its indicator. Retain the registered lag storage order and
chronological recurrent consumption. Hours before H0 remain padded unknown;
earlier status evidence may support hours at/after H0. Pre-budget history at/after
H0 may be an input but never enlarges adaptation-label budgets.

Training, development evaluation and final inference must share this function,
column ordering, dtypes and transformations. The input log1p transform remains
the same whichever Stage-1 output/loss formulation wins. Predictors must never
be generated by shifting retrospective Y, target_12h, target_24h or their masks.
Declare separate predictor and label schemas and prohibit label-field reads by
the active predictor path.

## 5. Retrospective labels versus fitting snapshots

Evaluation adjudication is logically separate and may use later evidence where
the registered evaluation protocol permits retrospective ground truth. Its
pinned evidence snapshot and mask are identified in the eventual data manifest.
Later adjudication cannot alter predictors, prediction keys, or earlier fitted
artifacts. Prediction generation is not gated by eventual target validity;
the evaluator joins permitted labels after prediction commitment.

For a fit at cutoff C, a label hour must lie in its registered fitting window,
have e<=C, and satisfy O_i(h;C)*Q_i(h;C)=1. All station/network conditions are
computed at C, including the city denominator. Do not merely filter an eventual
label table by next_status<=C, or intersect with an eventual retrospective mask.
Neural labels, HA sufficient statistics, adaptation pools and training-time
target masks use this same cutoff-specific membership. A training example at
historical origin t still uses F(h;t), never F(h;C).

No post-C evidence may backfill completed fits. Freeze sorted training-key
manifests with values/mask snapshot hashes, origin-feature hash, permitted city
set, window, budget and C before any optimizer update or HA fitting. Missing
coverage must not extend windows. Empty required positive-budget pools are a
hard stop/non-estimable condition, never an implicit window extension or fallback.

Development fits/adaptation end at HD; final source fits/adaptation end at HF.
There are no parameter or fitted-statistic updates in [HF,HE). Current source
domain pooling in Stage 4 may use the permitted training target mask at C; it
must not feed retrospective evaluation masks into inference.

## 6. Exact endpoints and deterministic timestamps

Hour membership is s <= departure_timestamp < e. A historical hour is complete
when e<=t. A previous event at s is permitted; one exactly 12h before s is
permitted. A next event at e or exactly e+12h is permitted only if <= cutoff.
12h+1 microsecond is rejected on either elapsed comparison. Evidence exactly
at t or C is permitted; evidence strictly after is prohibited. Use exact UTC
microsecond timestamp arithmetic, not minute-rounded date_diff predicates.
Pin timestamp conversion and timezone-database/runtime provenance before builds.
Hourly/weekly lags use elapsed UTC time across DST, not wall-clock offsets.
For coverage, deduplicate identical station/timestamp evidence into a set;
irrelevant status payload conflicts cannot alter coverage. Missing timestamps
cannot certify coverage. Any later payload-dependent feature requires its own
registered deterministic policy and must not be silently introduced.

## 7. Scientific design carried forward

Preserve city roles, pseudo-target folds, H0/HD/HF/HT/HE, half-open phase windows,
elapsed scarcity windows, parameter-zero meaning, architectures, geographic
graph algorithm, metrics and bounded selection rules. Their authoritative
sources and hashes are bound in the machine-readable specification.

Development seeds remain 17,29,43. Stage 1 retains the two raw/log1p masked-MAE
formulations, Stage-0 Graph-GRU reference, four folds, zero-shot and seven-day
adaptation comparison. Stage 2 retains k={4,8,16}, H={32,64}, dropout={0,0.1}.
Stage 2B retains the independent genuine VanillaGRU H={32,64}, dropout={0,0.1}
grid and its registered selection/tie rules. All active V2.2 winners are unset.
Previous winners are historical decisions, not defaults for corrected science.

Stage 3 retains full-network adaptation, reset AdamW, learning rate 0.0002,
weight decay 0.0001, batch size 16, gradient clip 1, no early stopping, and
0/100/300/600/1200 updates for zero/1d/7d/30d/full. Sampling is with replacement
within the permitted elapsed window. Zero budget is checkpoint reuse, not a fit.
Stage 4 must eventually bind the newly selected width/loss; discriminator input
width follows the selected backbone rather than retaining the old width of 32.

## 8. Authority, version separation and supersession

Authority order is: accepted V2.2 amendment and consistent machine contract;
fixed-cohort/static manifest; verified panel and fit snapshots; causal-history
and descriptive gates; baseline records; Stage-1 selection; independent Stage-2
and Stage-2B selections and joint closure; rebound Stage-3 policy; rebound
Stage-4 method. Lower artifacts cannot amend higher rules. Conflicts stop work.
Unchanged V2.1 design sources apply only subject to this amendment. New changes
require another append-only amendment, not edits to sealed V2.2 artifacts.

Every future active artifact must declare protocol_version=2.2, this seal's
SHA256, artifact role, schema version, producer/code hashes and exact upstream
hashes. Reject version/schema/hash mismatches and implicit latest-path fallbacks.
V2.1 predictors, fit rows, checkpoints, predictions and winners cannot be loaded
as active V2.2 artifacts. Only explicitly bound raw/infrastructure/design
provenance may cross versions; copied metadata keeps its original-source hash.

The append-only supersession registry assigns affected V2.1 panels/caches,
Stage-1/2/2B results, HA/baseline results and executable downstream bindings:

    EXECUTION_PROVENANCE_RETAINED_SUPERSEDED_FOR_CAUSAL_INFERENCE.

Preserve all V2.1 bytes, hashes, raw outputs and incident records. Historical
aborted attempts retain their original statuses; they do not become completed
experiments. Stage-3 numerical policy and Stage-4 conceptual method remain
usable design provenance; their old executable upstream bindings are superseded.

## 9. Mandatory pre-scientific gates

The 18 named invariants in the machine contract are mandatory. They cover future
append, physical status-prefix equivalence, evidence lineage, exact boundaries,
train/inference parity, fit-cutoff invariance, target separation, final-label
firewall, denominator zero, odd/even 50%, all three value states, duplicate/missing
events, DST, fixed cohort, metadata hashes, idealized count feed, no retrospective
Y reads, and sealed training-key manifests. Use synthetic tests first and
development-only integration checks before scientific reruns. A specification
verifier PASS does not claim these implementation tests have run.

Before Stage 1, require development-only descriptive reporting by city/lag for
lags 1..24 and 168; positive/zero/unknown proportions; source rows at HD;
pseudo-target rows for 1/7/30/full; baseline coverage and HA fitting pools.
Freeze explicit denominators, counts and missingness reasons. This gate cannot
change budgets, cities, thresholds, grids or model design because availability
is inconvenient. Flag sparse/empty persistence coverage with no invented
fallback. Empty comparison intersections are unavailable contrasts, not scores.
Baseline predictive artifacts and metrics remain future work, not this seal.

## 10. Rerun DAG and execution boundaries

    V2.2 implementation -> invariant tests -> V2.2 panel + fit snapshots
    -> descriptive availability/baseline audit -> Stage 1 rerun/selection
    -> {Stage 2 rerun/selection || Stage 2B rerun/selection}
    -> joint upstream closure -> Stage 3 rebind -> Stage 4 method rebind
    -> Stage 4 implementation/software gates -> Stage 4 development execution.

Stage 2 and Stage 2B may run in parallel only after Stage 1 V2.2 is frozen.
The completed-stage neural reruns comprise 48 Stage-1 fits (24 source plus
24 adaptation), 144 Stage-2 source fits and 48 Stage-2B source fits: 240 total.
Stage 4 development later adds 72 source and 72 adaptation fits. Rebinding
policies/seals performs no fits. Future execution requires its own authorization.

Final execution remains blocked by the final 3-versus-5 seed contradiction,
seed x temporal-bootstrap aggregation, and prediction-commitment/final execution
machinery. These do not block V2.2 implementation, and this seal does not resolve
them or authorize final execution. No final evaluation labels are accessed.

## 11. Read-only specification verification

Run `python -B research/scripts/verify_causal_history_spec_v2_2.py`.
The verifier uses the standard library, reads only allowlisted governance/static
provenance, hashes files, validates the contract and emits JSON to stdout.
It cannot rebuild data, fit models, evaluate, generate predictions or open final
evaluation labels. Its report is a specification/integrity result only.
