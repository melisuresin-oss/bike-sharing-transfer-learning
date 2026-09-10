# Final V2.2 results

All values below are read from the frozen JSON summaries in
[`results/final_v2_2/`](../results/final_v2_2/). MAE is measured in count space;
lower standalone MAE is better.

## Standalone equal-city macro MAE

| Method | 0d | 1d | 7d | 30d | Full |
|---|---:|---:|---:|---:|---:|
| Source historical average | 1.072289 | — | — | — | — |
| Target historical average | — | 0.976402 | 0.941287 | 0.835273 | 0.807435 |
| Seasonal naive | 0.878195 | 0.878195 | 0.878195 | 0.878195 | 0.878195 |
| Persistence | unavailable | unavailable | unavailable | unavailable | unavailable |
| Target-only Vanilla-GRU | — | 1.111835 | 0.996551 | 0.970196 | 0.956180 |
| Target-only Graph-GRU | — | 1.084970 | 0.996926 | 0.992346 | 0.991063 |
| Pooled Graph-GRU | — | 0.888162 | 0.869285 | 0.879580 | 0.906724 |
| Ordinary source-pretrained Graph-GRU | 0.874405 | 0.889892 | 0.873627 | 0.872275 | 0.871797 |
| GRL source-pretrained Graph-GRU | 0.940250 | 0.961752 | 0.943779 | 0.936276 | 0.936954 |

Persistence is unavailable throughout because the strictly causal one-hour lag
did not satisfy the final as-of validity requirement on the fixed evaluation
cohort; unavailable predictions were not replaced with zeros.

## How to read controlled gains

For every controlled comparison:

> **Controlled gain = MAE(reference) − MAE(method). Positive favors the evaluated method.**

Controlled paired gains use fixed matched prediction intersections. They are
therefore **not** simply differences between independently aggregated standalone
table cells. Learned-method comparisons are formed within matched training seed
before averaging over the five seeds.

## Graph-GRU versus Vanilla-GRU

The evaluated method is target-only Graph-GRU; the reference is target-only
Vanilla-GRU.

| Budget | Controlled gain |
|---|---:|
| 1d | +0.026866 |
| 7d | −0.000375 |
| 30d | −0.022149 |
| Full | −0.034883 |

Graph propagation helps at 1d, is effectively neutral at 7d, and trails the
Vanilla-GRU under the two larger supervision budgets.

## Ordinary transfer versus target-only Graph-GRU

The evaluated method is ordinary source-pretrained Graph-GRU after the fixed
adaptation policy; the reference is target-only Graph-GRU.

| Budget | Controlled gain |
|---|---:|
| 1d | +0.195077 |
| 7d | +0.123300 |
| 30d | +0.120071 |
| Full | +0.119266 |

Ordinary transfer improves on target-only Graph-GRU at every nonzero budget.

## GRL transfer versus ordinary transfer

The evaluated method is GRL-based source-pretrained Graph-GRU; the reference is
ordinary source-pretrained Graph-GRU.

| Budget | Controlled gain |
|---|---:|
| 0d | −0.065845 |
| 1d | −0.071859 |
| 7d | −0.070153 |
| 30d | −0.064002 |
| Full | −0.065156 |

The selected constant-λ GRL specification underperforms ordinary pretraining at
every budget in this final benchmark.

## Pooled versus sequential ordinary transfer

The evaluated method is pooled Graph-GRU; the reference is sequential ordinary
source pretraining followed by target adaptation.

| Budget | Controlled gain |
|---|---:|
| 1d | +0.001730 |
| 7d | +0.004342 |
| 30d | −0.007305 |
| Full | −0.034927 |

The two strategies are close at scarce budgets; pooled training has a small
paired advantage at 1d and 7d, while sequential transfer is favored at 30d and
Full.

## City heterogeneity

Equal-city macro values conceal meaningful variation:

- At 30d, Graph-GRU versus Vanilla-GRU gains are −0.037455 in Mannheim and −0.051921 in Innsbruck, but +0.000588 in Glasgow and +0.000191 in Split.
- At 7d, pooled versus sequential ordinary-transfer gains are +0.044787 in Mannheim, −0.004880 in Innsbruck, −0.017428 in Glasgow, and −0.005111 in Split.
- At Full, ordinary transfer versus target-only Graph-GRU gains remain positive in all four targets: +0.149950 in Mannheim, +0.193265 in Innsbruck, +0.047367 in Glasgow, and +0.086482 in Split.

These examples support conclusions tied to this fixed set of systems and
budgets, not a claim of universal significance across European bike-sharing
systems.

## Artifact provenance

The standalone, paired, and bootstrap summaries are byte-preserved artifacts.
Their SHA-256 values are recorded in
[`results/final_v2_2/MANIFEST.json`](../results/final_v2_2/MANIFEST.json). The
manifest itself is a newly created, path-neutral public index and is not a
frozen scientific result.
