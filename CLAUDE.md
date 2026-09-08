# V2.2 Scientific Source of Truth

This branch is the collaboration snapshot of the research design that passed the V2.2 scientific and engineering gates. Files under `research/` are authoritative for V2.2. The pre-existing `src/` tree is retained as historical team work and must not be treated as the current scientific implementation.

Frozen decisions must never be silently changed or replaced:

- Stage 1: `LOG1P`; decision SHA-256 `8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed`.
- Stage 2: `GGRU_K04_H032_D00` (`k=4`, hidden size 32, dropout 0); decision SHA-256 `9cca604ec3496a28f4440d89a9d6d1184311fc507c35f27b428a3ca768a6eeca`.
- Stage 2B: `VGRU_H032_D00` (hidden size 32, dropout 0); decision SHA-256 `9d85d6df6a509013f1b5411002246aee81c254fbde078f639293976674b4f36d`.
- Stage 3: rebound target-adaptation policy; contract SHA-256 `f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5`.
- Stage 4: scientific design and prospective selection clarification; contract SHA-256 `94e586624d18ac9047c6b0a71922139ad00795da54e20ca05d8ef04008470e15`.

## Current execution state

**Stage 4 is currently running on the university NVIDIA A40. No Stage-4 result or winner is frozen.** Do not infer, fabricate, aggregate, or publish a Stage-4 result from this snapshot.

Final targets are sealed: Mannheim 195, Innsbruck 199, Glasgow 237, and Split 617. Do not access final labels, create final predictions, or start final evaluation.

The following downstream gates remain unresolved and must not be silently resolved:

- `FINAL_SEED_COUNT_UNRESOLVED_3_VS_5`
- `FINAL_SEED_BOOTSTRAP_AGGREGATION_UNRESOLVED`

## Permitted AI-assisted work

AI assistants may explain code, improve documentation, draft paper sections from verified artifacts, prepare figure or table templates from frozen results, write unit tests, review implementations, identify possible bugs, and clearly propose changes without implementing scientific changes.

Without explicit team approval, AI assistants must not change the target transform, city splits, causal observability rules, features, architecture, graph construction, adaptation budgets, Stage-4 grid, or Stage-4 selection rule. They must not tune against final targets, open final labels, invent missing results, replace frozen manifests, or claim Stage-4 results before the frozen execution completes.

Potential contradictions or bugs must be reported explicitly and fail closed; they do not authorize a scientific redesign.
