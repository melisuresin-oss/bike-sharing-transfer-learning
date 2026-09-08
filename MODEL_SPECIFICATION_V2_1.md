# Model specification V2.1

## Positioning and non-novelty statement

The registered backbone is called **Graph-GRU**. Graph propagation inside a
gated recurrent architecture is established prior work and is not presented as
novel. This lightweight implementation is a shared controlled backbone for an
empirical cross-city transfer study.

It is distinct from two reference families:

- **Canonical T-GCN:** Zhao et al. combine graph convolution for spatial
  dependence with a GRU for temporal dependence. V2.1 does not claim a faithful
  reproduction of that published computation graph or use its name as shorthand
  for unspecified code. See https://arxiv.org/abs/1811.05320 and
  https://doi.org/10.1109/TITS.2019.2935152.
- **DCRNN:** Li et al. model traffic on a directed graph with diffusion
  convolution inside a recurrent sequence-to-sequence architecture and use
  scheduled sampling. V2.1 uses neither a directed random-walk diffusion
  operator nor a sequence-to-sequence decoder. See
  https://arxiv.org/abs/1707.01926 and
  https://openreview.net/forum?id=SJiHXGWAZ.

Our Graph-GRU uses symmetric geographic-kNN GCN-style propagation inside GRU
gates, shared across differently sized city graphs, for one-hour-ahead station
departure forecasting. Its value is tested through controls; the architecture
itself is not a contribution claim.

## Graph and recurrent equations

For each city, nodes are the stations passing the frozen `q_s >= 0.50` rule.
Directed geographic `k`-nearest-neighbour selections are symmetrized by union.
For selected non-self distance `d_ij`,

\[
w_{ij}=\exp[-(d_{ij}/\sigma_c)^2],
\]

where `sigma_c` is the city's median nonzero selected-edge distance. Add unit
self-loops and define

\[
\widehat A=\widetilde D^{-1/2}(A+I)\widetilde D^{-1/2},\qquad
G(U;W,b)=\widehat A U W+b.
\]

The Graph-GRU gates are unchanged from V2:

\[
r_t=\sigma(G([X_t,H_{t-1}];W_r,b_r)),\quad
u_t=\sigma(G([X_t,H_{t-1}];W_u,b_u)),
\]
\[
\widetilde H_t=\tanh(G([X_t,r_t\odot H_{t-1}];W_h,b_h)),\quad
H_t=u_t\odot H_{t-1}+(1-u_t)\odot\widetilde H_t.
\]

A shared node-wise head produces the nonnegative output under the Stage 1
scale/loss choice. Different cities have no edges between them.

## Genuine vanilla GRU control

The non-graph control is a separate vanilla GRU implementation:

\[
r_t=\sigma([X_t,H_{t-1}]W_r+b_r),\quad
u_t=\sigma([X_t,H_{t-1}]W_u+b_u),
\]
\[
\widetilde H_t=\tanh([X_t,r_t\odot H_{t-1}]W_h+b_h),\quad
H_t=u_t\odot H_{t-1}+(1-u_t)\odot\widetilde H_t.
\]

It has no adjacency object or neighbour aggregation and is not implemented by
calling identity adjacency a GRU. Inputs, hidden size, head, output/loss,
training exposure, and seeds are matched to Graph-GRU; parameter-count
differences are reported.

## Inputs and coverage

Inputs are exactly those in `FEATURE_PROTOCOL_V2_1.md`: 24 causal hourly demand
lags with masks, a causal 168-hour lag with mask, rack-capacity value/validity,
and deterministic local calendar features. **Weather is absent.** Status values,
future coverage evidence, city identifiers, and realized future covariates are
not features. Labels use the V2.1 12-hour bracket-only mask.

## Required experiment families

### Deterministic historical averages

`HISTORICAL_AVERAGE_BASELINE_AMENDMENT.md` is authoritative. For nonzero
target budget `b`, `HA_TARGET_b` falls back in this order: station local
hour-of-week mean, station overall mean, equal-station target-city local
hour-of-week mean, equal-station target-city overall mean, equal-city/equal-
station source local hour-of-week mean, then equal-city/equal-station source
global mean. `HA_SOURCE` uses only the last two levels and is the parameter-zero
baseline. Pseudo-target sources exclude the fold target and use `[H0,HD)`;
final sources use all eight development cities over `[H0,HF)`. Target statistics
use only the registered budget ending at the applicable cutoff. There is no
smoothing, shrinkage, interpolation, minimum-bin count, or fitted weighting.

Persistence is exactly `y[t-1]` when that elapsed UTC lag is observed. Seasonal
naive is exactly `y[t-168h]` when that elapsed UTC lag is observed. Either is
unavailable—not zero—when its lag mask is false.

- historical local hour-of-week average, persistence `y[t-1]`, and seasonal
  naive `y[t-168]`;
- vanilla GRU target-only;
- Graph-GRU target-only;
- ordinary source-pretrained Graph-GRU, frozen and fine-tuned;
- pooled source-plus-target Graph-GRU from scratch under the matched exposure
  protocol; and
- domain-adversarial source pretraining, frozen and fine-tuned.

For domain-adversarial pretraining, a discriminator predicts source city from a
mask-weighted mean of node hidden states through gradient reversal. It is
removed during target fine-tuning and inference. This is a transfer-mechanism
ablation, not a novel architecture.

## Final-target meaning

Final forecasting performance is sealed, and final evaluation labels cannot
influence coverage choice, features, development selection, training schedules,
or model settings. Final cities are not metadata-unseen: pretest status and
demand summaries support eligibility and deterministic diversity strata, and
final-period status availability defines the common cohort/window. This is
few-shot target-label adaptation, not true new-city cold start.

## Reproducibility checks

Before any fit: validate graph symmetry, self-loops, finite weights, roster
hashes, tensor shapes, strict lag causality, V2.1 mask invariants, nonnegative
outputs, exact optimizer exposures, deterministic seed construction, and the
absence of final evaluation labels in all configuration paths. This document
specifies future work; no forecasting model has been run.
