# V2.2 development scientific data

This package reads the separately authorized development build. It does not
change the sealed `research.v2_2` implementation, V2.1 runners, or any scientific
policy. Construction calls the sealed causal-history function at every original
forecast origin and uses its exact coverage and training-key seal functions.

The eight development cities contain 523 members of the frozen 799-station
cohort. Every predictor partition spans [H0,HF) with full frozen node order.
Storage is origin, lag 1..24, station, channel; registered recurrent consumption
remains chronological. Weekly history uses exact elapsed 168-hour offsets.
The partition identity's cutoff is its exclusive end HF; each row's origin and
feature fingerprint bind its actual as-of cutoff. Unknown feature values are
zero placeholders with a false indicator. Predictors contain no label fields.

Development retrospective labels are separate NPZ files. Their observation
mask is explicit and count -1 is the unknown sentinel, never a demand value.
Each partition records its retrospective status adjudication cutoff. Evaluation
key files include only observed pseudo-target labels in [HD,HF), without metrics.

HD source-city NPZ partitions store sorted (city, station, origin) eligible keys,
counts, true observation masks and original-origin feature hashes. Seven-city
source-fold manifests bind these partitions without duplicating them. Each
adaptation budget has its own NPZ and training-key seal. The sealed core computes
the same snapshot payload/key hashes for future fitting consumers. HA manifests
reference exactly these permitted fitting pools; no means or lookups were fitted.

`DevelopmentRegistry` requires an externally supplied manifest SHA256. Predictor,
retrospective-label and fit access are separate methods. It rejects legacy data,
wrong identities, final cities and label paths in predictor access. The only
legacy data exception is `graph(city,k)`, which reads the exact reconciled static
adjacency named in the V2.2 graph manifest and returns a V2.2 identity. No graph
hyperparameter is selected. `sealed_fit` checks the sealed core training-key gate;
zero-budget snapshots cannot start a fitting operation.

The read-only panel verifier checks stored data, hashes, original-origin feature
fingerprints, fitting subsets/seals, descriptive counts, graph bindings and
historical preservation. Its report binds this reader's code hash as well as the
data-manifest hash. It writes only to stdout and performs no scientific model
operation. The earlier implementation verifier remains historical evidence: its
pre-build absence checks are intentionally not a post-build gate.

Stage 1 and all subsequent model execution require separate user authorization.
