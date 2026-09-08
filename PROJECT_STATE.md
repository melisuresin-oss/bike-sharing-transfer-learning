# V2.2 Current State — 2026-09-08

This branch records the current scientifically authoritative V2.2 project state. The repository's pre-existing `main`/`feature` code and `src/` implementation represent earlier work; `research/...` on this branch is the V2.2 scientific source of truth.

## Done and frozen

- Causal V2.2 redesign and invariance audit
- V2.2 development data and sealed metadata
- Stage 1 scale/loss selection: `LOG1P`
- Stage 2 Graph-GRU selection: `GGRU_K04_H032_D00`
- Stage 2B Vanilla-GRU selection: `VGRU_H032_D00`
- Stage 3 target-adaptation policy rebind
- Stage 4 method design and prospective selection clarification

## Running

- Stage 4 on the university NVIDIA A40
- 72 source-pretraining fits plus 72 post-pretraining adaptation fits (144 total)
- Results are not included here and no Stage-4 winner exists yet

## Do not start

- Final held-out evaluation
- Any final-target label access or final prediction generation

## Open gates

- `FINAL_SEED_COUNT_UNRESOLVED_3_VS_5`
- `FINAL_SEED_BOOTSTRAP_AGGREGATION_UNRESOLVED`

These gates remain explicit blockers for the later final execution plan. This snapshot does not resolve them.
