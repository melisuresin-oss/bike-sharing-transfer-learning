# Feature protocol V2.1

## Frozen primary feature set

One feature set is fixed for every deterministic split, city, learned model,
target budget, and seed. It is not selected by forecasting performance and no
arbitrary subset search is permitted.

| Tensor | Shape | Frozen contents |
|---|---:|---|
| `X_hist` | `B x 24 x N x 2` | `log1p` demand for elapsed UTC lags 1–24 and one observation indicator per lag |
| `X_week` | `B x N x 2` | `log1p` demand at exactly `t-168h` and its observation indicator |
| `X_static` | `N x 2` | `log1p(bike_racks)` when positive and a rack-validity indicator |
| `X_calendar` | `B x 6` | local-hour sine/cosine, local-weekday sine/cosine, weekend, and UTC offset/12 |
| `Y` | `B x N` | nullable next-hour matched departure count |
| `M` | `B x N` | V2.1 bracket-only label mask |

`B` is a batch of forecast anchor hours and `N` is the frozen eligible station
roster for the current city. The Graph-GRU and vanilla GRU receive exactly the
same values; only the Graph-GRU uses adjacency.

## Dynamic demand history

For lag `l`, the value is `log1p(y_t-l)` only when that prior station-hour was
coverage-qualified. An unavailable lag uses numeric zero as a neutral tensor
placeholder and observation indicator zero. The placeholder is never interpreted
as observed zero demand. The weekly lag is an elapsed 168 UTC hours, which may
differ from the same wall-clock hour across DST.

At rolling prediction time, an observed target-city label may become a later lag
only after its hour ends. Target history may affect predictions causally but does
not update a parameter-zero checkpoint.

## Calendar and static inputs

For IANA-timezone local hour `u` and Monday-based weekday `d`, calendar inputs are
`sin(2*pi*u/24)`, `cos(2*pi*u/24)`, `sin(2*pi*d/7)`,
`cos(2*pi*d/7)`, a Saturday/Sunday indicator, and UTC offset hours divided by
12. These are deterministic and need no fitted normalizer.

Rack capacity is `log1p(bike_racks)` only when `bike_racks > 0`; otherwise the
value and validity flag are zero. It is neither a target denominator nor a
city-specific scale. Coordinates construct the graph but are not node features.
Station ID/name, city/country ID, terminal type, place type, special racks,
absolute coordinates, return policy, and status values are excluded from the
primary model.

## Weather exclusion

V2.1 uses **no weather feature**. The repository has no separately audited,
historical forecast-weather source with issue timestamps and a leakage-safe
join protocol. Realized future weather is prohibited. Adding weather later
requires a pre-result protocol amendment, source/version manifest, forecast
vintage semantics, missingness audit, and a separate sensitivity—not a silent
primary-feature expansion.

Public holiday and event features are also excluded because no harmonized local
source is frozen. The primary set relies only on repository demand history,
audited station metadata, timezone, and deterministic calendar transforms.

## Scale and availability

Feature transforms contain no fitted target mean, variance, quantile, or
capacity normalization. The `log1p` lag transform remains identical in the raw
and log1p output/loss comparison so Stage 1 isolates the target/loss choice.
Every feature must be constructible for all V2 cities under the same code path.

## Leakage checks

- every dynamic feature timestamp is strictly earlier than its target;
- the future status event used to certify coverage is not a feature;
- no final evaluation label enters a feature before its hour ends;
- no full-period lifetime, future maintenance state, target test statistic, or
  target-specific embedding is exposed; and
- the serialized feature manifest has an ordered column list, dtype, transform,
  and source timestamp for every tensor channel.
