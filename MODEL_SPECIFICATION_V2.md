# Model specification V2

## Scientific role

The learned backbone is named **Graph-GRU**: a graph-convolutional recurrent
model in which neighborhood aggregation occurs inside GRU gates. It is not
labelled a canonical T-GCN reproduction. Zhao et al.'s published T-GCN combines
a graph convolutional network with a gated recurrent unit; the registered model
below shares that broad motivation but uses explicitly stated gate equations so
the implementation, rather than a borrowed label, defines the method.

Reference: Zhao, L. et al. (2020), “T-GCN: A Temporal Graph Convolutional
Network for Traffic Prediction,” *IEEE Transactions on Intelligent
Transportation Systems*, 21(9), 3848–3858,
https://doi.org/10.1109/TITS.2019.2935152 (preprint:
https://arxiv.org/abs/1811.05320).

## Frozen graph and inputs

For each city, nodes are the stations passing the 0.50 pretest coverage rule.
Edges are the symmetric union of geographic `k`-nearest-neighbor links, with
great-circle distance and self-loops. Candidate `k` values and all feature,
hidden-size, optimizer, dropout, and schedule choices are selected exclusively
through pseudo-target development and then frozen. Edge construction never uses
trip flows or final demand labels.

Inputs at hour `t` contain permitted lagged demand plus registered calendar,
weather, static station, and missingness features. A lag is usable only if its
timestamp is earlier than `t`; missing lags are not filled as observed zeros.

Let `Atilde=A+I`, `Dtilde_ii=sum_j Atilde_ij`, and
`Ahat=Dtilde^(-1/2) Atilde Dtilde^(-1/2)`. For a node-feature matrix `U`, define
`G(U;W,b)=Ahat U W+b`. With concatenation denoted by brackets, the Graph-GRU is

\[
r_t=\sigma(G([X_t,H_{t-1}];W_r,b_r)),\quad
u_t=\sigma(G([X_t,H_{t-1}];W_u,b_u)),
\]
\[
\widetilde H_t=\tanh(G([X_t,r_t\odot H_{t-1}];W_h,b_h)),\quad
H_t=u_t\odot H_{t-1}+(1-u_t)\odot\widetilde H_t.
\]

A shared node-wise output head maps `H_t` to the nonnegative prediction under
the development-selected scale/loss formulation. Graph topology is fixed within
a run.

## Genuine vanilla GRU control

The vanilla GRU is a separate recurrent module with no adjacency object or
neighborhood aggregation:

\[
r_t=\sigma([X_t,H_{t-1}]W_r+b_r),\quad
u_t=\sigma([X_t,H_{t-1}]W_u+b_u),
\]
\[
\widetilde H_t=\tanh([X_t,r_t\odot H_{t-1}]W_h+b_h),\quad
H_t=u_t\odot H_{t-1}+(1-u_t)\odot\widetilde H_t.
\]

Weights are shared across stations and the output head matches Graph-GRU. It is
not implemented by substituting identity adjacency. Parameter counts, inputs,
history, loss, batches, and optimization budgets are matched as closely as
possible and any residual difference is reported.

## Required comparisons

The experiment families are:

- historical hour-of-week average, persistence `y[t-1]`, and seasonal naive
  `y[t-168]`;
- vanilla GRU trained only on the target budget;
- Graph-GRU trained only on the target budget;
- ordinary source-pretrained Graph-GRU evaluated parameter zero-shot;
- ordinary source-pretrained Graph-GRU followed by target fine-tuning;
- pooled Graph-GRU trained jointly from scratch on sources and the permitted
  target budget under a matched optimization budget; and
- domain-adversarial source pretraining followed by the same target
  fine-tuning.

For domain-adversarial pretraining, the discriminator receives a mask-weighted
mean of station hidden states and predicts the source-city identity. A gradient
reversal layer applies the registered coefficient to the encoder while demand
loss remains primary. The discriminator is removed during target fine-tuning.
The coefficient and schedule are selected only in pseudo-target folds.
Domain-adversarial training is a candidate mechanism and ablation, not an
architectural novelty claim.

Target-only learned models are undefined at the parameter-zero budget and are
not replaced with source weights. Deterministic baselines use the history
actually permitted at each key and budget; unavailable lags produce null
predictions excluded from the matched comparison cohort.

## Reproducibility checks

Before any forecasting experiment, tests must confirm graph symmetry and
self-loops, dimensional consistency, nonnegative outputs, strict lag causality,
mask-invariant loss denominators, deterministic seed behavior, and the absence
of source-city labels during fine-tuning and inference. This file specifies a
future implementation; no model has been trained for V2.
