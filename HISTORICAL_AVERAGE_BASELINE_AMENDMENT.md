# Historical-average baseline amendment to V2.1

## Status

This pre-model amendment is authoritative for `RESEARCH_PROTOCOL_V2_1`. It
closes the deterministic historical-average ambiguity before baseline code or
forecasting performance exists. It changes no city role, station roster,
coverage mask, feature, temporal boundary, target budget, or evaluation rule.

## Registered baselines

Two distinct deterministic baselines are registered:

- `HA_TARGET`, defined only for the 1-day, 7-day, 30-day, and full target-label
  budgets; and
- `HA_SOURCE`, the parameter-zero no-target-label historical baseline.

For target city `c`, station `s`, prediction time `t`, and its validated local
hour-of-week bin `b(t)`, `HA_TARGET` uses this fixed fallback hierarchy:

1. station × local-hour-of-week mean;
2. station overall mean;
3. target-city local-hour-of-week mean;
4. target-city overall mean;
5. source-only local-hour-of-week mean; and
6. source-only global mean.

`HA_SOURCE` uses source-only local-hour-of-week mean, then source-only global
mean. It uses no target station mean, target-city statistic, adaptation label,
or final-evaluation label.

## Equal-station and equal-city aggregation

Fallback statistics never pool all station-hours.

- A target-city hour-of-week value is the equal mean of available station-level
  means for that bin.
- A target-city global value is the equal mean of station overall means.
- A source hour-of-week value first averages station-bin means equally within
  each source city, then averages available city-bin means equally across
  source cities.
- A source global value first averages station overall means equally within
  each source city, then averages the resulting city means equally.

One legally permitted observation makes a statistic available. There is no
minimum count, shrinkage, interpolation, neighboring-bin smoothing, learned or
similarity weighting, or baseline-specific demand normalization.

## Frozen fitting windows

For pseudo-target development, source statistics exclude the pseudo-target and
use the other seven development cities over `[H0,HD)`. Target statistics use
only the target's registered budget ending at `HD`.

For final inference, source statistics use all eight development cities over
`[H0,HF)`. Target statistics use only the registered final-target budget ending
at `HF`. No budget is extended to replace unavailable observations.

## Local calendar and DST

All bins are Monday-based local hour-of-week under the validated IANA timezone
mapping. A target bin is matched to the same local bin in every source city.
UTC hour-of-week is not used. DST follows `FEATURE_PROTOCOL_V2_1`; the fixed
UTC key is converted to local calendar time before binning.

## Output identifiers

Use `HA_SOURCE`, `HA_TARGET_1D`, `HA_TARGET_7D`, `HA_TARGET_30D`, and
`HA_TARGET_FULL`. `HA_SOURCE` must never be labelled as a target historical
average.
