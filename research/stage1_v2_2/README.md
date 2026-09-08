# Stage 1 V2.2

This implementation independently compares RAW and LOG1P using corrected V2.2
development inputs. It never starts Stage 2, Stage 2B or Stage 4.

The scientific manifest and immutable job map are written before training. The
preflight report gates launch. The map contains 24 source fits (two candidates,
four pseudo-targets, three seeds) and their 24 seven-day adaptations. Six CPU
workers execute the map with two PyTorch threads each. Runtime and scientific
source hashes are frozen in the manifest.

The registered Graph-GRU, head, masked losses, AdamW trainer and metric functions
are reused unchanged as software definitions. No V2.1 scientific predictors,
fitting pools, checkpoints or predictions are consumed. The historical
`stage1-v2.1` string in seed derivation is retained solely to preserve the frozen
RNG mapping; it does not identify the input or output artifact version.

Source-city exposure is exactly balanced to within one update across seven
cities. Anchors are sampled uniformly with replacement from the sealed HD
fitting masks. Adaptation resets AdamW and uses only the sealed seven-day mask.
The training cache contains fit-cutoff counts and masks, never retrospective
evaluation labels. Features, including unknown lag 1, are copied without repair,
imputation or availability changes.

Every source fit takes exactly 12,000 updates; every adaptation takes 300.
Final-step checkpoints alone supply evaluation and selection. Intermediate
checkpoints at 2,000-update intervals preserve model/optimizer/RNG state and
exposure counters as recovery evidence; they are not candidate checkpoints.
The launcher refuses to silently repeat tasks with existing execution records.

For each zero-shot and adapted model, all immutable [HD,HF) development
predictions are saved before retrospective labels are opened by the evaluation
function. Count-space MAE is calculated on observed keys. RMSE, WAPE and
station-macro MAE are secondary. Three seeds are averaged within each of eight
fold-by-regime cells, then those eight means receive equal weight. Four-decimal
ROUND_HALF_UP ties use unrounded population SD across the eight cells, then RAW.
The historical V2.1 decision is opened for descriptive comparison only after the
independent V2.2 decision has been written.

The separate postrun verifier instantiates no model and performs no forward,
backward or optimization call. It checks stored checkpoint tensors and actual
AdamW step counters, saved predictions, reconstructed metrics, exact job
membership, input hashes, the independent selection calculation and historical
preservation. Its report and closure seal are appended after scientific output
creation without changing the pre-execution manifest or job map.
