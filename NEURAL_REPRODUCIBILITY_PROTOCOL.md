# Neural reproducibility protocol

## Seed and deterministic execution

`set_seed(seed)` initializes Python `random`, NumPy, PyTorch CPU, and all CUDA
generators when CUDA exists. Deterministic PyTorch algorithms are enabled;
cuDNN benchmarking is disabled and deterministic cuDNN behavior requested.
Data-order and initialization seeds can be recorded separately by later
registered experiment runners while deriving both from the registered seed.

The software-validation run used PyTorch `2.13.0+cpu`; CUDA was unavailable.
CPU reruns under this environment are deterministic. PyTorch does not promise
bitwise identity across library releases, hardware types, or every CUDA kernel.

## Checkpoint contract

Every checkpoint stores:

- model state and AdamW optimizer state;
- complete architecture configuration;
- graph method, `k`, sigma, node count, coordinate hash, roster hash, and
  adjacency hash where applicable;
- protocol version;
- feature-manifest SHA-256;
- city-roster SHA-256;
- registered seed, epoch, and optimizer step;
- raw-count or log1p target/loss mode; and
- neural source-code SHA-256.

Loading fails before weights are accepted when protocol, feature schema,
station roster, or expected graph hashes disagree. Both validation checkpoints
reproduced predictions with maximum absolute difference exactly `0.0`.

## Training invariants

- optimizer: explicitly configured AdamW only;
- registered defaults: betas `(0.9,0.999)`, epsilon `1e-8`, weight decay
  `1e-4`;
- global gradient clipping: norm `1.0`;
- fixed-step/final checkpoint is the normal mode;
- an explicit best-fixed-validation mode may be configured later only where the
  governing development protocol authorizes it;
- there is no automatic scheduler search or target-specific early stopping;
- structured step logs record step, masked loss, valid targets, pre-clip
  gradient norm, and learning rate;
- city-balanced loss averages within city and then equally across valid cities.

## Final-label firewall

Neural model, graph, and training modules contain no sealed-label path. Their
data boundary is the existing hash-verified registry. Final inference supplies
features and keys with `Y=None`, and the trainer refuses a label-free batch.
The validation manifest records:

- final labels accessed: `false`;
- final-target predictions created: `false`;
- pseudo-target validation compared: `false`; and
- Stage 1 started: `false`.

## Reproduction commands

```powershell
python -m pip install -r research/requirements.txt
python -m unittest discover -s research/tests -v
python research/scripts/run_neural_validation_amended.py --overwrite
```

The amended runner verifies the original failed-evidence hash before writing to
a separate output directory. It executes the predeclared teacher-student gate,
two fixed-seed geographic trajectories, the identity diagnostic, equivalence
audit, checkpoint roundtrip, synthetic overfit, and microbenchmarks. The current
amended run exits successfully; its duplicate trajectories have maximum
absolute loss difference `0.0` against tolerance `1e-7`.
