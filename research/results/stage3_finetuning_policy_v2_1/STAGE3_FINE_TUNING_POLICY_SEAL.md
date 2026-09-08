# Stage 3 Fine-Tuning Policy Seal

**Status:** `FROZEN_VERIFIED`  
**Seal type:** append-only Stage 3 fine-tuning policy  
**Created:** 2026-09-03T16:13:39.8691202Z

## Authorization boundary

This seal freezes policy only. Its creation performed no model training, fine-tuning, inference, prediction evaluation, or performance comparison. It did not access final-target labels. It does not authorize Stage 3 execution or Stage 4.

Stages 1 and 2 remain permanently frozen. Their decision-manifest SHA-256 digests are pinned in the machine-readable manifest and verified before this seal is accepted.

## Frozen upstream decisions

- Scale/loss: `LOG1P_TARGET_MAE`.
- Ordinary backbone: `GGRU_K04_H032_D00`.
- Graph neighbours: `k = 4`.
- Hidden size: `32`.
- Dropout: `0.0`.

## Frozen Stage 3 policy

- Fine-tune the full network.
- Reset AdamW optimizer state before adaptation.
- Learning rate: `2e-4`.
- Weight decay: `1e-4`.
- Batch size: `16` eligible forecast-anchor hours from one city per update.
- Global gradient clipping: norm `1.0`.
- Use the final fixed-step checkpoint; early stopping is prohibited.
- Target-only update budgets: `0`, `100`, `300`, `600`, and `1200`, corresponding respectively to 0-day, 1-day, 7-day, 30-day, and full-pre-cutoff target-label budgets.
- Budget `0` is the unchanged source checkpoint evaluated as the zero-update/parameter-zero-shot reference. It is not a fine-tuning fit.
- Sample adaptation labels with replacement only from the registered elapsed-time budget. A masked label does not extend the budget.
- Causal pre-budget target history may supply inputs, but it cannot supply additional adaptation targets.

No candidate set, metric, tie-breaker, or performance-based selection exists in Stage 3.

## Consistency result

The policy above is fully consistent with `RESEARCH_PROTOCOL_V2_1.md`, `COMPUTATION_DAG_V2_1.md`, `TRAINING_BUDGET_PROTOCOL.md`, `DEVELOPMENT_SELECTION_PROTOCOL.md`, `MODEL_SPECIFICATION_V2_1.md`, `TEMPORAL_SPLIT_PROTOCOL_V2.md`, and `NEURAL_REPRODUCIBILITY_PROTOCOL.md`. `EVALUATION_PROTOCOL_V2.md` adds the required prediction-before-label-opening firewall and does not contradict these Stage 3 policy constants.

One adjacent contradiction exists outside the policy constants: `EVALUATION_PROTOCOL_V2.md` registers final learned seeds 17, 29, and 43, whereas the later `RESEARCH_PROTOCOL_V2_1.md` and `COMPUTATION_DAG_V2_1.md` register final seeds 17, 29, 43, 71, and 101. This seal does not silently choose between them. The inconsistency is non-blocking for freezing the Stage 3 fine-tuning policy because the seal neither selects seeds nor executes fits, but it is blocking for the downstream Stage 3 execution plan.

## Next gate—not executed

The next downstream action is to produce an append-only Stage 3 execution-plan/firewall manifest. Before any final-target label is opened or any fit begins, that plan must reconcile the final-seed contradiction explicitly; bind the frozen backbone and source-checkpoint provenance; enumerate target, budget, method, and seed jobs according to the computation DAG; separate budget `0` reuse from nonzero fine-tuning fits; pin eligible adaptation-label windows and hashes; prove that `[HT,HE)` evaluation labels are absent from every training path; commit final prediction keys and hashes before label opening; and pass a preflight that triggers no training.

This next gate is described here for auditability only. It has not been created or executed.
