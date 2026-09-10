# Final GRL Source-Domain Cardinality Clarification V2.2

**Status:** `FROZEN_PROSPECTIVE_CLARIFICATION`  
**Scope:** final V2.2 GRL source refitting only  
**Seal date:** 2026-09-09  
**Verdict:** `FINAL_GRL_CARDINALITY_V2_2_RESOLVED_AND_SEALED_READY_FOR_R6`

## Purpose and authority

This append-only clarification resolves one narrow ambiguity discovered before R6 implementation and before any final experiment began. It does not modify or supersede a frozen Stage-1, Stage-2, Stage-2B, Stage-3, or Stage-4 artifact. It interprets the already-frozen Stage-4 method under the already-frozen final rule that all eight registered development cities are used in every final source refit.

The resolution was made prospectively from authoritative protocol lineage. No final-target label, prediction, metric, or model outcome was accessed or generated.

## Lineage finding

Stage 4 used four leave-one-pseudo-target development folds. In each fold, the pseudo-target was excluded from source pretraining, source batches, domain labels, and reload probes. The active source pool was therefore the eight registered development cities minus the one held-out pseudo-target: exactly seven source cities.

The discriminator label denotes the identity of the active source city. The frozen development implementation consequently used:

`Linear(32,64) -> ReLU -> Dropout(0.10) -> Linear(64,7)`

The literal value 7 was not a candidate in the Stage-4 selection grid, was not compared with another domain count, and was not used by the selection rule. The complete registered grid varied only `lambda_max` in `{0.01, 0.10, 0.50}` and schedule in `{constant, linear}`. Domain loss and domain accuracy were diagnostic only. Therefore 7 was structurally induced by each leave-one-pseudo-target fold; it was not an independently selected scientific hyperparameter.

## Candidate-resolution analysis

1. **Eight outputs for the final all-source refit — accepted.** This applies the unchanged source-city classification semantics to the eight active registered source cities. It is mechanically induced by the final source pool and is consistent with the required final refit.
2. **Keep seven outputs by excluding one source city — rejected.** This contradicts the frozen rule that all eight development-source cities participate in every final source refit, silently changes source membership, and introduces an unregistered city-exclusion choice.
3. **Keep seven outputs by merging two source cities — rejected.** This breaks the frozen source-city-identity meaning of the labels and introduces an unregistered, non-reproducible merge rule that changes the scientific question.
4. **Reuse a Stage-4 seven-source checkpoint — rejected.** Stage-4 checkpoints are fold-specific development artifacts with one pseudo-target excluded. Reuse would violate the registered final all-source refit and its one-refit-per-final-seed design.
5. **Other interpretations — none supported.** No authoritative document registers a literal-seven final classifier, a city exclusion, a merged domain, or checkpoint substitution.

## Frozen final GRL rule

The discriminator output dimension is defined as:

`discriminator_output_dim = number_of_active_source_domains`

- Stage-4 development fold: `8 - 1 = 7` active source domains and seven outputs.
- Final all-source refit: `8` active source domains and eight outputs.

Each final GRL source refit uses exactly these cities and this deterministic class map, in frozen development-source order:

| Domain class | City ID | City |
|---:|---:|---|
| 0 | 129 | Dortmund |
| 1 | 194 | Heidelberg |
| 2 | 438 | Marburg |
| 3 | 467 | Gießen |
| 4 | 476 | Cardiff |
| 5 | 532 | Bilbao |
| 6 | 619 | Freiburg |
| 7 | 658 | Göteborg |

The exact final discriminator is:

`Linear(32,64) -> ReLU -> Dropout(0.10) -> Linear(64,8)`

It retains ordinary unweighted cross-entropy over active source-city identities, unit forward domain-loss weight, and the same gradient-reversal mechanism. The selected backbone remains `GGRU_K04_H032_D00`; `lambda_max` remains `0.5`; the schedule remains `constant`. The registered optimizer, update, sampling, initialization, clipping, and checkpoint-provenance rules remain unchanged. There is no additional tuning, no city exclusion, no city merge, and no Stage-4 checkpoint substitution.

Changing the terminal layer from seven to eight outputs for the eight-domain final task is not a new selected hyperparameter. It is the mechanical adaptation of a categorical head to the cardinality of its registered label set. The selected Stage-4 method and scientific question are preserved.

## Final computation DAG impact

The clarification changes no job identity or count. The five final GRL source refits remain one fit for each final seed `17`, `29`, `43`, `71`, and `101`, each using the same eight-source pool.

- Unique neural fits: `410`
- Unique evaluation/statistic passes: `468`
- Budget-labelled reporting cells: `660`
- Final GRL source refits: `5`

## Firewall state at seal

```text
phase_a_execution_data_constructed = true
phase_b_targets_materialized = false
final_predictions_generated = false
final_metrics_computed = false
final_experiment_started = false
optimizer_updates = 0
model_forward_calls_on_final_labels = 0
```

This clarification does not authorize execution by itself. R6 may be implemented only as a separate downstream software/package step that faithfully realizes this sealed rule and passes the existing final-label firewall and pre-execution gates.

## Authoritative lineage bindings

| Artifact | SHA-256 |
|---|---|
| `RESEARCH_PROTOCOL_V2_1.md` | `6bbf5bc39292ef9e1bf8321b90006269217b43fd0df59da25687712fc418c35c` |
| `RESEARCH_PROTOCOL_V2_2_AMENDMENT.md` | `02a44e3aa8bf1d656d2f9de833b45072d564052e2ab3c1426bda2a6456c558d0` |
| `COMPUTATION_DAG_V2_1.md` | `48f38bc97505d622316008b8ac155fb09e2b9a9efce6f36dad7e15b35d9f53cf` |
| `MODEL_SPECIFICATION_V2_1.md` | `430fd12c6322a102687c2a546be394bb7ef344656c1a34edf05a86beadca88bf` |
| `FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md` | `c4a8fb1cc314cd93336269b8b1ee035771ce766c823e82476daac97e3e565f6c` |
| `final_evaluation_protocol_v2_2.json` | `16617ca602e2d70cc621005a838dfb84d308b5f33488bac8b92d38fccea1d054` |
| `research/results/stage4_adversarial_method_v2_1/STAGE4_ADVERSARIAL_METHOD_SEAL.md` | `fa3e82673dd0e5196729966953fe2a099f58190a3f33821eacf2f827b4000ee1` |
| `research/results/stage4_adversarial_method_v2_1/stage4_adversarial_method_manifest.json` | `a830e155f0f4052a4b8f3a7d9492bc27877513b11cee885dc93ab6fa1cee7d92` |
| `research/results/stage4_source_invariance_v2_2/scientific_contract.json` | `94e586624d18ac9047c6b0a71922139ad00795da54e20ca05d8ef04008470e15` |
| `research/results/stage4_source_invariance_v2_2/job_map.json` | `ce84b658100d49a313fa102ea6b07d0319d197f1e23d610af53b420a8492458c` |
| `research/results/stage4_selection_clarification_v2_2/selection_clarification.json` | `852d8c32dd9004a8dfaa17633e285877828672d79414cf5fb67b8550c8ea7d4d` |
| `research/results/stage4_v2_2_returned_20260908/stage4_source_invariance_v2_2_a40/stage4_decision_manifest.json` | `b8ab61745465fcb3c2db3ce811effaba091111a57eefe89f70d5fa5df3e1aae0` |
| `research/stage4_v2_2/method.py` | `2579d5af8f79e3abd49a16a654f2ca1116fb0fc178628aed52f1354cd1c26782` |
| `research/stage4_v2_2/core.py` | `4eda3dcad1d9cf97cf1f03374e9ccb3eef45716a8e0999b8ea1d080a7c247f03` |
| `research/stage4_v2_2/selection.py` | `654eb08e9a1a9165a1dd504bc00c204ac6a5203a180bbb8d2f120c704dbee535` |

