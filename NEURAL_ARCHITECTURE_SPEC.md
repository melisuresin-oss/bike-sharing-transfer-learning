# Neural architecture specification

## Scope and inheritance

This implementation is a software realization of the frozen V2.1 vanilla GRU
and Graph-GRU only. It does not implement DANN or execute development selection.

`MODEL_SPECIFICATION.md` explicitly places the masked weekly lag, static station
features, and target-hour calendar features after recurrent encoding. V2 and
V2.1 retain the gate equations and do not revoke that placement. Accordingly,
only the two-channel hourly history enters the recurrent cell. The final hidden
state is concatenated with the ten registered contextual channels before one
shared linear/Softplus head. This inheritance should be restated directly in a
future consolidated protocol edition; it is not an architectural change here.

## Tensor flow

| Stage | Vanilla GRU | Graph-GRU |
|---|---|---|
| Loader history | `X_hist [B,24,2]` | `X_hist [B,24,N,2]` |
| Recurrent order | loader lag `1..24` is reversed to chronological `t-24..t-1` | same |
| Recurrent input | `[value * observed, observed]`, 2 channels | `[value * observed, observed]`, 2 channels per node |
| Redundant mask | `M_hist [B,24]` validates the embedded indicator; it is not appended again | `M_hist [B,24,N]`, same rule |
| Hidden state | `[B,H]` | `[B,N,H]` |
| Context | weekly pair + static pair + 6 calendar channels = `[B,10]` | weekly `[B,N,2]` + broadcast static `[B,N,2]` + broadcast calendar `[B,N,6]` = `[B,N,10]` |
| Shared head input | `[B,H+10]` | `[B,N,H+10]` |
| Output | `[B]` | `[B,N]` |

An unavailable numeric history placeholder is multiplied by its zero indicator
inside the model. An observed zero is `[0,1]`; a missing value is `[0,0]`.
Dropout is applied to the sanitized recurrent input and to the final recurrent
state before the head. There is no recurrent-gate dropout.

## Explicit vanilla GRU

The implementation uses an explicit cell rather than `torch.nn.GRU`:

\[
r_t=\sigma([x_t,h_{t-1}]W_r+b_r),\quad
u_t=\sigma([x_t,h_{t-1}]W_u+b_u),
\]
\[
\widetilde h_t=\tanh([x_t,r_t\odot h_{t-1}]W_h+b_h),\quad
h_t=u_t\odot h_{t-1}+(1-u_t)\odot\widetilde h_t.
\]

Weights are shared across stations. The module contains no adjacency argument,
graph object, neighborhood aggregation, station embedding, or city embedding.

## Explicit Graph-GRU

For the registered normalized geographic adjacency,

\[
G(U;W,b)=\widehat A U W+b.
\]

Every reset, update, and candidate gate uses this propagation exactly as frozen:

\[
r_t=\sigma(G([X_t,H_{t-1}];W_r,b_r)),\quad
u_t=\sigma(G([X_t,H_{t-1}];W_u,b_u)),
\]
\[
\widetilde H_t=\tanh(G([X_t,r_t\odot H_{t-1}];W_h,b_h)),\quad
H_t=u_t\odot H_{t-1}+(1-u_t)\odot\widetilde H_t.
\]

T-GCN motivates graph-plus-recurrent forecasting and DCRNN is another
graph-recurrent precedent. This registered backbone instead uses symmetric
geographic-kNN propagation inside GRU gates. No novelty claim is made for the
recurrent construction.

## Coordinate-only graph

The graph utility projects only `station_id`, `latitude`, and `longitude` in
frozen node order. It computes haversine distances, caller-supplied `k` nearest
neighbors, symmetric union, Gaussian weights using the median nonzero selected
edge distance, unit self-loops, and symmetric degree normalization. It has no
demand/trip argument and does not tune `k`. Architecture validation uses the
registered Stage-0 reference `k=8` only.

## Prediction head and losses

One shared `Linear(H+10,1) -> Softplus` head supports both still-unresolved
registered modes:

- `raw_count`: Softplus output is the count prediction and masked raw-count MAE
  is used;
- `log1p_target`: Softplus output is the transformed prediction, masked log1p
  MAE is used, and reporting space is `max(0,expm1(output))`.

This implementation does not select between the modes. No other activation,
loss, fitted normalization, or output clipping is added.

## Parameter counts

| Model | Hidden size | Trainable parameters |
|---|---:|---:|
| Vanilla GRU | 32 | 3,403 |
| Vanilla GRU | 64 | 12,939 |
| Graph-GRU | 32 | 3,403 |
| Graph-GRU | 64 | 12,939 |

Both explicit cells use the same three `(input+hidden) x hidden` matrices, three
bias vectors, and shared contextual head. Graph propagation has no trainable
parameter, so the parameter counts are exactly equal without altering either
registered architecture.
