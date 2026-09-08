# Coverage and target definition V2.1

## Superseding amendment

This file supersedes the primary city-hour condition in
`COVERAGE_AND_TARGET_DEFINITION_V2.md`. City roles, temporal boundaries, the
12-hour station bracket, and the pretest `q_s >= 0.50` station roster remain
unchanged.

## Target

For eligible station `s` and UTC hour `h`, `y_s,h` is the number of valid,
city-matched trips whose departure falls in `[h,h+1h)`. It is an observed
departure count, not latent demand or complete transaction ground truth. The
eligible station-by-hour index is constructed before counts are joined.

## Primary bracket-only mask

Let `p_s,h` be the latest status event at or before `h` and `n_s,h` the earliest
event at or after `h+1h`. Station `s` is bracketed when both exist, `h-p_s,h <=
12h`, and `n_s,h-(h+1h) <= 12h`.

A station-hour enters the primary panel exactly when:

1. the station belongs to a frozen V2 city and has finite coordinates;
2. its pre-`HF` 12-hour bracketed-lifetime fraction is `q_s >= 0.50`;
3. the hour is within its observed status lifetime;
4. the station is 12-hour bracketed; and
5. at least 50% of eligible stations whose lifetimes cover the city-hour are
   also 12-hour bracketed.

There is **no same-hour status-event requirement**. `station_status` is
change-triggered; absence of a change during an exact hour is not evidence of
inactivity. The pre-model audit showed that the old condition disproportionately
removed quiet nighttime zero/no-trip hours.

When the mask is true, no matched trip produces a **coverage-qualified
zero-demand station-hour**. When it is false, the target is null even when the
raw trip count is zero. Positive departures outside the mask are also excluded
from loss and metrics and reported as a separate diagnostic. Demand never
determines coverage.

## Frozen sensitivities

- **Same-hour event:** add at least one eligible-station event in the exact hour.
- **Network-recent 6h:** add at least one eligible-station network event within
  six hours before and six hours after the hour.
- **24-hour bracket:** replace each 12-hour station bound with 24 hours while
  retaining the same roster, roles, Rule B city semantics, and time windows.
- **No-maintenance:** symmetrically remove station-hours whose most recent prior
  status is maintenance from labels, predictions, loss, and metrics.

No sensitivity selects a model or changes a cohort after performance is seen.
If a substantive conclusion reverses, it is reported as mask- or
maintenance-dependent.

## Causality and final-target seal

Coverage may use the next status event because it retrospectively determines
label observability; that future event is never a model input. Features and
demand lags use only information available strictly before the prediction hour.
The common final window remains `[2023-05-11T15:00:00Z,
2023-07-15T20:00:00Z)`, conservatively preserving the status-only V2 cohort.

“Sealed final target” means forecasting performance and final evaluation labels
cannot influence any modelling decision. It does not mean metadata-free or true
new-city cold start: pretest metadata/status, pretest demand summaries, and
status-only final-period availability were used before modelling for eligibility,
diversity stratification, and the common evaluation cohort. The experiment is
few-shot target-label adaptation.

## Required panel checks

- unique `(city_id, station_id, timestamp_utc)` keys;
- exactly the frozen station roster and UTC intervals;
- no retained station-hour without its station bracket and city threshold;
- no null target converted to zero;
- no negative count and exact reconciliation to matched trips;
- local-time features derived after the UTC key is fixed; and
- hashes for role, station-roster, mask-rule, and temporal manifests.
