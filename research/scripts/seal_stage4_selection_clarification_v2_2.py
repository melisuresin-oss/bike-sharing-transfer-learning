"""Prospective governance writes only, before Stage-4 scientific execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.scripts.rebind_stage3_audit_stage4_v2_2 import UPSTREAM

BASE="research/results/stage4_selection_clarification_v2_2/"
STAGE3="research/results/stage3_finetuning_policy_v2_2/stage3_finetuning_policy_manifest.json"
STAGE3_SHA="f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5"
def sha(rel):return hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
def write(name,value):
    p=ROOT/BASE/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
    return {"path":p.relative_to(ROOT).as_posix(),"sha256":sha(p.relative_to(ROOT))}

# Inventory file names throughout the repository without opening payloads.
# Exclude only Git/runtime internals and sealed final-data directories.
inventory=[];n=0
for parent,dirs,files in os.walk(ROOT):
    dirs[:]=[d for d in dirs if d not in (".git","__pycache__") and not any(t in d.lower() for t in ("final_labels","sealed_final","final_adaptation","final_features"))]
    for name in files:
        rel=(Path(parent)/name).relative_to(ROOT).as_posix();n+=1
        if re.search(r"stage[_-]?4|adversarial|dann",rel,re.I):inventory.append(rel)
unexpected=[]
for rel in inventory:
    low=rel.lower()
    if "v2_1" in low or "stage2b_ts9/" in low:continue
    if low.startswith(("research/stage4_v2_2/","research/results/stage3_stage4_gate_v2_2/")):continue
    if low.startswith(("research/governance/","research/scripts/","research/tests/")):continue
    unexpected.append(rel)
if unexpected:raise RuntimeError("STOP: unexpected Stage-4 candidate/output path before clarification: "+repr(unexpected))
for p,h in {**UPSTREAM,STAGE3:STAGE3_SHA}.items():
    if sha(p)!=h:raise RuntimeError("Authoritative hash mismatch: "+p)
stamp=datetime.now(timezone.utc).isoformat()
evidence=write("pre_execution_firewall.json",{
    "schema_version":"1.0","protocol_version":"2.2","status":"PASS_PRE_EXECUTION",
    "timestamp_utc":stamp,"scope":"Local authoritative repository inventory plus prior conversation execution history",
    "files_inventoried":n,"stage4_related_paths":sorted(inventory),"unexpected_v2_2_output_paths":unexpected,
    "historical_v2_1_artifacts":"method/protocol/audit provenance only; not V2.2 candidate results",
    "prior_v2_2_stage4_artifacts":"stage3_stage4_gate_v2_2 blocked audit only; no scientific results",
    "new_stage4_code_before_seal":"unexecuted deterministic selection implementation only; no fit, prediction, metric, ranking or winner output",
    "completed_stage4_v2_2_scientific_fits":0,"stage4_v2_2_predictions":0,"stage4_v2_2_evaluations":0,
    "stage4_v2_2_candidate_rankings":0,"stage4_v2_2_winner_manifests":0,
    "stage4_v2_2_scientific_results_inspected":False,"final_target_labels_accessed":False,"final_experiment_started":False})
clarification=write("selection_clarification.json",{
    "schema_version":"1.0","protocol_version":"2.2","artifact_role":"prospective_stage4_selection_clarification",
    "status":"FROZEN_PROSPECTIVE_CLARIFICATION","timestamp_utc":stamp,
    "historically_defined":False,"created_before_stage4_scientific_execution":True,
    "reason":"Historical Stage-4 seal omitted primary tie equality/quantization/rounding and the exact fold-dispersion statistic/comparison.",
    "authority":"User explicitly resolves the two omissions prospectively before Stage-4 scientific execution or result inspection.",
    "pre_execution_evidence":evidence,"upstream_sha256":{**UPSTREAM,STAGE3:STAGE3_SHA},
    "dataset_manifest_sha256":UPSTREAM["processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json"],
    "primary":{"regime":"post_7_day_300_update_adaptation","metric":"count_space_MAE",
        "seeds":[17,29,43],"folds":[532,476,619,658],
        "seed_aggregation":"arithmetic_mean_within_fold_in_registered_seed_order",
        "fold_aggregation":"unweighted_arithmetic_mean_of_four_seed_mean_fold_MAEs","lower_is_better":True},
    "primary_tie":{"operation":"Decimal(str(primary_unrounded)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)",
        "precision_decimal_places":4,"rounding":"ROUND_HALF_UP","tie_iff":"quantized_primary_values_equal",
        "quantized_value_use":"primary_equality_only; compare unrounded primary when quantized values differ",
        "retain_unrounded":True,"rationale":"Prospectively matches frozen Stage-2 convention; not chosen after Stage-4 results"},
    "dispersion":{"statistic":"population_standard_deviation","inputs":"four_fold_MAEs_after_seed_averaging",
        "denominator":4,"comparison":"unrounded_full_precision; no tolerance and no quantization",
        "equality":"exact_equality_of_deterministic_stored_numeric_representation"},
    "numerical_implementation":{"aggregation":"statistics.fmean in registered seed/fold order",
        "dispersion":"statistics.pstdev over four seed-mean fold MAEs","storage":"finite binary64 Python float, lossless JSON roundtrip"},
    "hierarchy":["lower_primary_with_4dp_HALF_UP_equality","lower_unrounded_population_SD","smaller_lambda_max","constant_before_linear"],
    "lambda_order":[0.01,0.10,0.50],"schedule_order":["constant","linear"],
    "excluded_selection_inputs":["zero_shot_MAE","RMSE","WAPE","station_macro_MAE","pooled_station_hours","discriminator_accuracy","domain_loss","subjective_stability","final_target_performance"],
    "selection_code":{"path":"research/stage4_v2_2/selection.py","sha256":sha("research/stage4_v2_2/selection.py")},
    "stage4_scientific_results_inspected":False,"real_scientific_optimizer_updates_locally":0,"real_scientific_evaluations_locally":0,
    "final_target_labels_accessed":False,"final_experiment_started":False,
    "downstream_blockers":["FINAL_SEED_COUNT_UNRESOLVED_3_VS_5","FINAL_SEED_BOOTSTRAP_AGGREGATION_UNRESOLVED"]})
write("CLARIFICATION_SHA256.json",clarification)
print(json.dumps(clarification,indent=2))
