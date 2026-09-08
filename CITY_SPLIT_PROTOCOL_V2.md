# City split protocol V2

## Authority and pre-model gate

`eligible_city_statistics_full.csv` is the authoritative city audit. City
selection uses only fields computed before `HF = 2023-04-16T22:00:00Z`; no
forecast, model residual, fitted representation, or final-period demand label is
permitted.

The primary high-confidence population satisfies all three frozen conditions:

1. `coverage_12h_gate_pass == true`;
2. `coverage_12h_observable_city_hour_fraction >= 0.90`; and
3. `coverage_12h_median_station_bracketed_lifetime_fraction >= 0.80`.

Exactly 12 cities pass: **Bilbao, Glasgow, Freiburg, Mannheim, Dortmund,
Cardiff, Split, Göteborg, Innsbruck, Heidelberg, Marburg, and Gießen**. The
thresholds are not adjusted to retain Wien or any previously preferred city.

The other eight corrected eligible cities fail as follows:

| City | 12h observable city-hours | 12h median station coverage | Primary reason |
|---|---:|---:|---|
| Warszawa | 59.87% | 59.67% | both thresholds |
| Wien | 65.58% | 59.43% | both thresholds |
| Palma de Mallorca | 42.91% | 94.54% | city-hour threshold |
| Ostrava | 36.75% | 39.03% | both thresholds |
| Praha | 2.67% | 16.76% | both thresholds |
| Las Palmas de Gran Canaria | 79.27% | 96.42% | city-hour threshold |
| Duisburg | 76.94% | 65.03% | city-hour threshold |
| KielRegion | 50.79% | 80.65% | city-hour threshold |

## Frozen diversity strata

Demand-scale and station-count low/middle/high strata are the rank-based
terciles already stored in the authoritative 20-city audit. They are not
recomputed after applying the high-confidence gate. The two continuous
selection coordinates are:

\[
z_d=z\{\log(1+\text{departures per station-active-day})\},\qquad
z_n=z\{\log(1+\text{stations with departures})\},
\]

where each mean and population standard deviation is computed over the 12
high-confidence cities.

## Deterministic four-target rule

Before enumerating targets, retain city 532 (Bilbao) in development because it
is the maximum demand-scale city. This makes the development-only scale/loss
comparison confront the most severe observed scale difference rather than
discovering it after final targets are opened.

Enumerate every four-city target subset and retain it only when:

- all four countries are distinct;
- exactly two target countries remain represented among development cities and
  exactly two target countries are unseen in development;
- both target and development sets contain low, middle, and high demand strata;
- both sets contain low, middle, and high station-count strata; and
- Bilbao is not a target.

For every retained subset compute all six pairwise Euclidean distances in
`(z_d,z_n)`. Sort subsets lexicographically by:

1. minimum pairwise distance descending;
2. minimum of the eight target coverage quantities (12h city and median-station
   coverage for four targets) descending;
3. mean of those eight coverage quantities descending;
4. pairwise-distance sum descending; and
5. sorted city-ID tuple ascending.

The first subset is frozen. Its minimum pairwise distance is **0.979608**, its
coverage floor is **0.893586**, and its ID tuple is `(195, 199, 237, 617)`.

## Frozen V2 roles

| Role | City (ID) | Country | Matched departures | Stations | Active days | Demand/station-day | 12h city | 12h station median | Stratum |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| sealed final target | Mannheim (195) | DE | 366,870 | 87 | 233 | 18.0983 | 97.24% | 96.01% | high / middle |
| sealed final target | Innsbruck (199) | AT | 171,866 | 45 | 233 | 16.3916 | 95.54% | 96.95% | high / low |
| sealed final target | Glasgow (237) | GB | 239,358 | 104 | 233 | 9.8778 | 98.32% | 95.84% | middle / high |
| sealed final target | Split (617) | HR | 125,506 | 68 | 233 | 7.9214 | 94.98% | 89.36% | low / middle |
| development source + pseudo-target | Bilbao (532) | ES | 1,031,013 | 44 | 230 | 101.8788 | 97.76% | 98.57% | high / low |
| development source + pseudo-target | Cardiff (476) | GB | 196,086 | 100 | 233 | 8.4157 | 97.53% | 85.10% | low / high |
| development source + pseudo-target | Freiburg (619) | DE | 333,232 | 99 | 233 | 14.4463 | 98.41% | 95.99% | middle / middle |
| development source + pseudo-target | Göteborg (658) | SE | 366,862 | 146 | 233 | 10.7843 | 94.29% | 88.82% | middle / high |
| development source | Dortmund (129) | DE | 207,266 | 96 | 233 | 9.2662 | 96.91% | 84.75% | low / middle |
| development source | Heidelberg (194) | DE | 208,652 | 50 | 233 | 17.9100 | 97.47% | 96.74% | high / low |
| development source | Marburg (438) | DE | 113,496 | 46 | 233 | 10.5893 | 97.04% | 92.44% | middle / low |
| development source | Gießen (467) | DE | 124,254 | 30 | 233 | 17.7760 | 95.76% | 92.96% | high / low |

Mannheim and Glasgow have countries represented in development. Innsbruck and
Split have countries unseen in development.

## Pseudo-target development

Use four leave-one-city-out pseudo-target folds, one per development country.
Within each development country, select the city with the highest corrected
pretest score, then city ID ascending. This gives **Bilbao, Cardiff, Freiburg,
and Göteborg**. In a fold, the pseudo-target is excluded from source fitting;
the other seven development cities are available as sources. The four remaining
German development cities improve source diversity but do not create additional
hyperparameter folds.

## Suggested candidate set

The suggested Freiburg–Cardiff–Göteborg–Innsbruck set satisfies the constraints
but ranks seventh. Its minimum pairwise distance is **0.789576** and coverage
floor **0.850957**, both below the selected set. It is therefore not substituted
after inspection.

## Seal and later diagnostics

Final-target demand labels may not select architecture, graph, features, loss,
training schedule, stopping epoch, station threshold, or target roles. A later
label-independent status check may shorten the common evaluation period but may
not replace a target. `city_roles_v2.csv` is the machine-readable role manifest.
