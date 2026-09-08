# V2.2 Stage-3 rebind / Stage-4 scientific gate

Date: 2026-09-08. This is a governance and implementation-readiness audit. No
scientific training, new scientific evaluation, or final-target data access ran.

## A. Upstream verification

All supplied hashes match local files. All three decisions are
`FROZEN_VALIDATED`, executed on `UNIVERSITY_A40`.

| Artifact | Repository-relative path | SHA-256 | Result |
|---|---|---|---|
| Stage 1 | research/results/stage1_scale_loss_v2_2_a40/stage1_decision_manifest.json | 8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed | PASS; LOG1P |
| Stage 2 | research/results/stage2_graphgru_selection_v2_2_a40/stage2_decision_manifest.json | 9cca604ec3496a28f4440d89a9d6d1184311fc507c35f27b428a3ca768a6eeca | PASS; GGRU_K04_H032_D00 |
| Stage 2B | research/results/stage2b_vanilla_fairness_v2_2_a40/stage2b_decision_manifest.json | 9d85d6df6a509013f1b5411002246aee81c254fbde078f639293976674b4f36d | PASS; VGRU_H032_D00 |
| V2.2 panel manifest | processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json | 140dd8c53cf51f48be5cf66b70d162c9f2cc6a9fa0b1b7039496633e385a6ab6 | PASS |
| Stage-2 scientific contract | research/results/stage2_graphgru_selection_v2_2/scientific_contract.json | 55ebfa06ca49d77e5ae0b7b2e531d6bb14c26a2b93f478da7754c75fb1aae52f | PASS |
| Stage-2 job map | research/results/stage2_graphgru_selection_v2_2/job_map.json | fd34f71b46eb6f9cfc9adb5ebfd44001a2e26df049b9bb197aff1d272e7a074a | PASS |
| Stage-2B scientific contract | research/results/stage2b_vanilla_fairness_v2_2/scientific_contract.json | 2eb1a2a14dd1412f1286248c547814a2c7e04fad94edaace2084c7d78be0cad3 | PASS |
| Stage-2B job map | research/results/stage2b_vanilla_fairness_v2_2/job_map.json | 0cf72a4f66797bcc5625bd438046562c5fd444677676971e2697e1f9a3d0f23d | PASS |

The decision-bound validation reports, 240 completion records (48/144/48), and
675 referenced checkpoints/predictions/prediction manifests/preflight artifacts
are synchronized and hash-consistent. Checkpoints were hashed, not deserialized;
predictions were hashed, not evaluated. Selection was not rerun.

The retained development aggregate MAEs are Graph-GRU 1.5529057885454507 and
Vanilla-GRU 1.5384051585782383. The vanilla result is lower. Both selected
configurations remain unchanged; historical V2.1 winners were not substituted.

The new joint closure is
`research/results/upstream_closure_v2_2/joint_upstream_closure_manifest.json`,
SHA-256 `d7a211de6627098bc1642d3eeb7efd0d7386a61513b1bd696a7407c809eef834`.

## B. Stage-3 rebind

The numerical adaptation policy is unambiguous and consistent across the old
policy seal, governing documents, V2.2 amendment, and existing FitRequest API.
The old executable binding pointed to V2.1; the new append-only contract binds
all three corrected decisions, V2.2 data, specification seal, and data APIs.

| Budget | Target-only updates | Fitting interval ending at C |
|---|---:|---|
| 0 | 0 | No fit; unchanged source checkpoint reuse |
| 1 day | 100 | [C-24 elapsed hours,C) |
| 7 days | 300 | [C-168 elapsed hours,C) |
| 30 days | 600 | [C-720 elapsed hours,C) |
| full | 1200 | [H0,C) |

Fresh AdamW: learning rate 0.0002, weight decay 0.0001, betas (0.9,0.999),
epsilon 1e-8. Batch 16 eligible anchor hours from one city, replacement sampling,
global gradient clipping 1.0, full-network adaptation, final fixed-step
checkpoint, no early stopping. Empty positive-budget pools stop as non-estimable;
missing rows never extend a window.

Development C=HD (2023-02-15T22:00:00Z); documentary final C=HF
(2023-04-16T22:00:00Z). H0=2022-08-29T05:00:00Z. Final embargo [HF,HT) is 593
hours, HT=2023-05-11T15:00:00Z. No parameter/statistic updates in [HF,HE), with
HE=2023-07-15T20:00:00Z. Final evaluation remains [HT,HE), only after future
authorization and prediction commitment. No final partition was opened or built.

Predictors retain their original forecast origin t. Fit eligibility and target
masks use the sealed cutoff C without reconstructing predictors at C. The
idealized count feed, exact UTC microseconds, frozen cohort/static metadata,
no-backfill rule, and no retrospective-Y predictor construction remain intact.

Contract:
`research/results/stage3_finetuning_policy_v2_2/stage3_finetuning_policy_manifest.json`

SHA-256: `f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5`

Status: `FROZEN_REBOUND_POLICY_ONLY`. Policy contradictions: none. No new
Stage-3 development search or execution is required by the frozen documents.
Final execution remains unauthorized.

## C. Stage-4 scientific audit

Scientific identity: GRL-based multi-source domain-invariant pretraining. The
pseudo-target is absent from source fitting and from the seven discriminator
classes. The backbone remains GGRU_K04_H032_D00, LOG1P, k=4, H=32, dropout=0.

The historical method seal establishes the supplied GRL gradient semantics,
unweighted CE with no separate alpha, masked hidden-state pooling, discriminator
architecture/dropout, one joint optimizer and global clipping, EqualCitySampler,
12,000 source updates, and constant/linear schedules including denominator 2999.
After source fitting, remove the discriminator and use the fixed fresh-optimizer
7-day/300-update adaptation. In V2.2, the permitted current training-example
target mask is M_i(t;C) from the sealed fit snapshot. It is not a historical
mask, a demand value, or a retrospective evaluation mask supplied to inference.

Confirmed workload: 3 coefficients x 2 schedules x 4 folds x 3 seeds = 72 source
fits plus 72 adaptations = 144 fits. There are 72 post-adaptation selection
evaluations and **zero zero-shot selection evaluations** in the historical seal.

Confirmed primary: post-7-day/300-update adaptation count-space MAE, arithmetic
mean of seeds 17/29/43 within each fold, then equal mean across four folds.
Confirmed tie-break order: lower fold dispersion, smaller lambda_max, constant.

Two precommitments are missing:

1. **Primary tie definition/quantization.** DEVELOPMENT_SELECTION_PROTOCOL.md
   lines 77-89 says only "ties" for Stage 4. The Stage-4 SELECTION_RULE in
   research/governance/stage4_adversarial_method_seal.py lines 126-134 and the
   sealed manifest do not specify exact equality versus quantization, decimal
   precision, or rounding mode. Four-decimal/ROUND_HALF_UP rules elsewhere are
   registered for other stages; no Stage-4 authority explicitly imports them.
2. **Exact dispersion statistic and comparison.** The same authorities specify
   only `lower_fold_dispersion`. They do not define SD versus another statistic,
   the SD convention if applicable, or dispersion comparison quantization.
   Population and sample SD rank equal-sized four-fold vectors identically,
   but this observation does not constitute an executable precommitment.

No historical executable Stage-4 selector resolving these omissions was found.
The inspected method-seal code enumerates a workload and verifies the method;
it does not implement selection. These are omissions, not a changed metric or
a zero-shot/post-adaptation conflict. No resolution was inferred from Stage 2.

Stage-4 execution is not scientifically authorized. Its deployment condition
(fully specified scientific design) is false. Required next input is an explicit
append-only clarification of the primary tie rule and exact dispersion
statistic/comparison, without reopening the established objective, grid, or
upstream results and without consulting any Stage-4 outcomes.

Audit:
`research/results/stage3_stage4_gate_v2_2/scientific_audit.json`

SHA-256: `f24afb39a460fc4dc2813ff669230f0fd802475f3af73e782ae5af4d5dffb812`

## D. Implementation and preservation

Created only:

- research/scripts/rebind_stage3_audit_stage4_v2_2.py
- research/results/upstream_closure_v2_2/joint_upstream_closure_manifest.json
- research/results/stage3_finetuning_policy_v2_2/stage3_finetuning_policy_manifest.json
- research/results/stage3_stage4_gate_v2_2/scientific_audit.json
- research/results/stage3_stage4_gate_v2_2/verification_report.json
- research/results/stage3_stage4_gate_v2_2/SCIENTIFIC_ENGINEERING_REPORT.md
- research/results/stage3_stage4_gate_v2_2/DELIVERY_MANIFEST.json

Existing files modified: none. V2.1 files and V2.2 upstream decisions are
untouched. Historical policy/method source integrity was checked against their
sealed hashes; old predictor payloads were not used. The new governance script
uses existing pure V2.2 Contract and FitRequest definitions, not duplicated
causal construction. It has no scientific training/evaluation entry point.

No Stage-4 training implementation, controller, selector, preflight, job map,
scientific execution contract, or package was created across the failed gate.

## E. Verification

20 governance/rebind checks: PASS. The report is
`research/results/stage3_stage4_gate_v2_2/verification_report.json`, SHA-256
`389e20c19851710cfcdef479f69784002aaf3b212485fa6513439c2fb1807094`.

These checks cover upstream pins, producer/dependency hashes, closure binding,
exact policy/budget mapping, five FitRequest windows, embargo, origin causality,
no window extension, rejected final paths, no new search/execution, preserved
historical selection wording, 144-fit accounting, closed deployment gate,
downstream seed conflict, and preservation of inspected governance sources.

Stage-4 implementation tests run: 0. Controller-startup integration: not run.
Metadata-canonicalization implementation/test: not run. All are deferred behind
the scientific gate; the 20 governance checks do not claim to satisfy the 33
requested Stage-4 implementation requirements.

REAL scientific optimizer updates locally = 0.
REAL scientific evaluations locally = 0.

## F. Package

ZIP path/hash/size: not created. Deployment manifest: not created. Stage-4
scientific contract hash and job-map hash: not created. Stage-3 contract hash is
given above. DELIVERY_MANIFEST.json inventories governance outputs only and is
not an execution authorization or an A40 deployment manifest.

## G. TS9 commands

The requested copy, extraction, package-check, preflight, detached Workers=4
launch, PID/GPU/progress/stderr/exit checks, 144/144 validator, freeze, and result
sync commands are not issued because no Stage-4 execution package exists and
the scientific gate failed. A future authorized package must retain the
required bike_env Python and deterministic A40 settings and default Workers=4.

The completed governance artifacts can be verified locally without training:

```powershell
& 'C:\Users\sefak\AppData\Local\Programs\Python\Python312\python.exe' -X utf8 -B research/scripts/rebind_stage3_audit_stage4_v2_2.py verify
```

## H. Firewall

final_target_labels_accessed = false

final_experiment_started = false

## I. Downstream blockers

FINAL_SEED_COUNT_UNRESOLVED_3_VS_5

The separate seed-by-temporal-bootstrap aggregation convention also remains
unresolved. Neither issue was imported into Stage-4 development selection,
which retains seeds 17,29,43. No final-target execution is authorized.

## Final verdict

STAGE4_BLOCKED_BY_SCIENTIFIC_AMBIGUITY
