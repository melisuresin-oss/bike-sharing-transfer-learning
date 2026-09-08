# Training budget protocol

## Fair comparison objective

The pooled and sequential Graph-GRU comparisons receive exactly the same number
of source-only updates, target-containing updates, total optimizer steps, batch
size, eligible target labels, features, graph, loss, and seed. They differ in
update order: sequential completes source pretraining before target-only
fine-tuning; pooled interleaves the same exposures from initialization.

## Common update unit

One optimizer update uses one city, 16 eligible forecast-anchor hours sampled
with replacement, and that city's full station graph. A source update selects a
source city uniformly: probability `1/7` inside a pseudo-target fold and `1/8`
for the final refit. Target updates select only anchors inside the registered
target-label budget. A city-balanced loss is averaged over valid station-hours.

The source exposure is fixed at `S = 12,000` updates. Target exposure `F_b` is
frozen by budget:

| Budget | Source updates | Target updates `F_b` | Total | Pooled source probability | Pooled target probability |
|---|---:|---:|---:|---:|---:|
| 0 days | 12,000 | 0 | 12,000 | 100.000% | 0.000% |
| 1 day | 12,000 | 100 | 12,100 | 99.174% | 0.826% |
| 7 days | 12,000 | 300 | 12,300 | 97.561% | 2.439% |
| 30 days | 12,000 | 600 | 12,600 | 95.238% | 4.762% |
| full | 12,000 | 1,200 | 13,200 | 90.909% | 9.091% |

Probabilities describe proportions, not uncontrolled binomial draws. For each
seed, construct a deterministic schedule containing exactly `S` source tokens
and `F_b` target tokens, shuffle it with the registered seed, and consume every
token. Conditional source-city draws are balanced to differ by at most one
update per city.

## Sequential source-pretraining plus fine-tuning

1. Run 12,000 source-only steps at the frozen source learning rate.
2. Save one source checkpoint per seed.
3. Reset AdamW state and run exactly `F_b` target-only steps at `2e-4`.

Thus total steps are `12,000 + F_b`, and every one of the final `F_b` updates
contains target labels. The single source checkpoint is reused across all four
targets and budgets for the same seed; it is not retrained per budget.

## Pooled training from scratch

Initialize once per target, budget, and seed. Process the exact shuffled token
schedule. Source-token steps use source learning rate `1e-3`; target-token steps
use `2e-4`; one continuous AdamW moment state is retained. This method receives
exactly `F_b` target-containing updates and 12,000 source updates, matching the
sequential method. It does not start from the sequential source checkpoint.

Target sampling is intentionally oversampled relative to the natural pooled
station-hour frequency whenever `F_b/(S+F_b)` exceeds that frequency. The
oversampling is disclosed and applied only to achieve matched target exposure;
it never enlarges the eligible target-label set.

## What is and is not matched

Matched: source and target update counts, total steps, batch size, source-city
balance, target budget, station-hour mask, features, graph, loss, initialization
distribution, seed list, gradient clipping, and per-domain learning rates.

Not matched by design: update order and optimizer-state history. Sequential
resets optimizer state at the source/target boundary; pooled has one continuous
state. These are part of the mechanisms being compared and must be stated when
interpreting pooling versus pretraining.

Report realized per-city source updates, target updates, valid station-hours,
and cumulative examples for every run. A deviation from the exact counts
invalidates the contrast rather than being corrected after results are seen.
