"""Governance-only Stage-3 rebind and Stage-4 scientific ambiguity gate.

Standard-library and existing pure V2.2 contract APIs only. No model, optimizer,
evaluation engine, prediction loader, label reader, or deployment launcher.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.v2_2.contract import Contract, HOUR, SPEC_SHA256, SEAL_SHA256, COHORT_SHA256
from research.v2_2.snapshots import FitRequest

UPSTREAM={
 "research/results/stage1_scale_loss_v2_2_a40/stage1_decision_manifest.json":"8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed",
 "research/results/stage2_graphgru_selection_v2_2_a40/stage2_decision_manifest.json":"9cca604ec3496a28f4440d89a9d6d1184311fc507c35f27b428a3ca768a6eeca",
 "research/results/stage2b_vanilla_fairness_v2_2_a40/stage2b_decision_manifest.json":"9d85d6df6a509013f1b5411002246aee81c254fbde078f639293976674b4f36d",
 "processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json":"140dd8c53cf51f48be5cf66b70d162c9f2cc6a9fa0b1b7039496633e385a6ab6",
 "research/results/stage2_graphgru_selection_v2_2/scientific_contract.json":"55ebfa06ca49d77e5ae0b7b2e531d6bb14c26a2b93f478da7754c75fb1aae52f",
 "research/results/stage2_graphgru_selection_v2_2/job_map.json":"fd34f71b46eb6f9cfc9adb5ebfd44001a2e26df049b9bb197aff1d272e7a074a",
 "research/results/stage2b_vanilla_fairness_v2_2/scientific_contract.json":"2eb1a2a14dd1412f1286248c547814a2c7e04fad94edaace2084c7d78be0cad3",
 "research/results/stage2b_vanilla_fairness_v2_2/job_map.json":"0cf72a4f66797bcc5625bd438046562c5fd444677676971e2697e1f9a3d0f23d",
}
OLD3="research/results/stage3_finetuning_policy_v2_1/stage3_finetuning_policy_manifest.json"
OLD4="research/results/stage4_adversarial_method_v2_1/stage4_adversarial_method_manifest.json"
CLOSURE="research/results/upstream_closure_v2_2/joint_upstream_closure_manifest.json"
STAGE3="research/results/stage3_finetuning_policy_v2_2/stage3_finetuning_policy_manifest.json"
OUT="research/results/stage3_stage4_gate_v2_2/"
AUDIT=OUT+"scientific_audit.json"
SELF="research/scripts/rebind_stage3_audit_stage4_v2_2.py"
POLICY={"full_network_fine_tuning":True,"reset_adamw_optimizer_state_before_adaptation":True,
        "optimizer":"AdamW","learning_rate":0.0002,"weight_decay":0.0001,"batch_size":16,
        "global_gradient_clip":1.0,"early_stopping":False,"update_budgets":[0,100,300,600,1200],
        "zero_budget_is_fine_tuning_fit":False,"adaptation_sampling":"with_replacement_within_elapsed_time_budget",
        "causal_pre_budget_history_as_input":True,"causal_pre_budget_history_as_extra_adaptation_target":False}
BUDGETS={"zero":0,"1":100,"7":300,"30":600,"full":1200}
FORBIDDEN=("final_labels","sealed_final","final_adaptation","final_features")
checked={}


def path(rel):
    if not isinstance(rel,str) or "\\" in rel or ":" in rel or ".." in Path(rel).parts:
        raise PermissionError("Repository-relative path required")
    if any(x in rel.lower() for x in FORBIDDEN):raise PermissionError("Final-target firewall")
    p=(ROOT/rel).resolve();p.relative_to(ROOT)
    return p


def read(rel):return json.loads(path(rel).read_text(encoding="utf-8"))
def sha(rel):
    h=hashlib.sha256()
    with path(rel).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def bind(rel,h):
    if rel in checked:require(checked[rel]==h,"Conflicting expected hash: "+rel)
    else:
        require(sha(rel)==h,"Absent or conflicting authoritative artifact: "+rel)
        checked[rel]=h
    return {"path":rel,"sha256":h}
def entry(rel):return {"path":rel,"sha256":sha(rel)}
def write(rel,value):
    require(rel.startswith((OUT,"research/results/upstream_closure_v2_2/","research/results/stage3_finetuning_policy_v2_2/")),"Write scope")
    p=path(rel);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
    return entry(rel)


def synchronize():
    for p,h in UPSTREAM.items():bind(p,h)
    decisions=[read(p) for p in list(UPSTREAM)[:3]]
    require(all(d["status"]=="FROZEN_VALIDATED" and d["execution_environment"]=="UNIVERSITY_A40" for d in decisions),"Frozen A40 decisions required")
    require(decisions[0]["selected_candidate"]=="LOG1P","Frozen Stage-1 winner")
    require(decisions[1]["selected_configuration"]["config_id"]=="GGRU_K04_H032_D00","Frozen Stage-2 winner")
    require(decisions[2]["selected_configuration"]["config_id"]=="VGRU_H032_D00","Frozen Stage-2B winner")
    completions=[];referenced=set();reports=[]
    for d,count in zip(decisions,(48,144,48)):
        report=d["postrun_validation"]
        if "path" in report:
            bind(report["path"],report["sha256"]);reports.append(report);report=read(report["path"])
        require(report["status"]=="PASS" and len(report["execution_records"])==count,"Validated upstream fit accounting")
        for e in report["execution_records"]:
            bind(e["path"],e["sha256"]);r=read(e["path"]);completions.append(e)
            require(r["status"] in ("COMPLETED","COMPLETED_VALIDATED_JOB"),"Unfinished authoritative job")
            refs=[r[k] for k in ("checkpoint","prediction","prediction_manifest","preflight") if k in r]
            if "metric" in r:refs.append(r["metric"]["prediction"])
            for artifact in refs:
                bind(artifact["path"],artifact["sha256"]);referenced.add(artifact["path"])
    return {"schema_version":"1.0","protocol_version":"2.2","artifact_role":"joint_upstream_closure",
            "status":"VERIFIED_UPSTREAM_CLOSURE","created_utc":datetime.now(timezone.utc).isoformat(),
            "upstream_sha256":UPSTREAM,"decisions":[entry(p) for p in list(UPSTREAM)[:3]],
            "selected_scale":"LOG1P","graph_gru":"GGRU_K04_H032_D00","vanilla_gru":"VGRU_H032_D00",
            "completed_fit_records":len(completions),"completion_counts":{"stage1":48,"stage2":144,"stage2b":48},
            "referenced_artifacts_hashed":len(referenced),"postrun_reports":reports,
            "synchronization_files_sha256":dict(sorted(checked.items())),
            "scope":"Hash/status/provenance closure; no model load, metric recomputation or selection rerun",
            "producer":entry(SELF),"specification_seal_sha256":SEAL_SHA256,
            "real_scientific_training_updates":0,"real_scientific_evaluations":0,"final_target_labels_accessed":False}


def historical_bindings():
    bind(OLD3,"c63528bfe73881eafe02167f5cfddf37e51bca769fab6289850a66836642a6c1")
    bind(OLD4,"a830e155f0f4052a4b8f3a7d9492bc27877513b11cee885dc93ab6fa1cee7d92")
    old3,old4=read(OLD3),read(OLD4)
    require(old3["policy"]==POLICY,"Contradiction in frozen Stage-3 policy")
    for m in (old3,old4):
        for group in ("governing_input_sha256","artifact_integrity"):
            for p,h in m[group].items():
                # Old datasets remain provenance only; never a V2.2 active input.
                if p.startswith("processed/"):continue
                bind(p,h)
    for base in ("stage3_finetuning_policy_v2_1/STAGE3_FINE_TUNING_POLICY_SEAL.sha256",
                 "stage4_adversarial_method_v2_1/STAGE4_ADVERSARIAL_METHOD_SEAL.sha256"):
        for line in path("research/results/"+base).read_text().splitlines():
            h,p=line.split("  ",1);bind(p,h)
    return old3,old4


def stage3_contract(closure):
    contract=Contract()
    windows=[]
    for budget,updates in BUDGETS.items():
        request=FitRequest.registered(contract,phase="development",kind="adaptation",budget=budget,target_city=532)
        start=request.start
        final_cutoff=contract.boundaries["HF"]
        final_start=contract.boundaries["H0"] if budget=="full" else final_cutoff if budget=="zero" else final_cutoff-int(budget)*24*HOUR
        windows.append({"budget":budget,"updates":updates,"is_fit":bool(updates),
                        "development_window_us":[start,request.cutoff],"final_window_us":[final_start,final_cutoff],
                        "interval":"left_closed_right_open","final_window_role":"policy_only_not_materialized_or_accessed"})
    dependencies={p:sha(p) for p in ("RESEARCH_PROTOCOL_V2_2_AMENDMENT.md","research/v2_2/contract.py",
        "research/v2_2/history.py","research/v2_2/snapshots.py","research/v2_2/artifacts.py",
        "research/development_data_v2_2/registry.py","processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json")}
    return {"schema_version":"1.0","protocol_version":"2.2","stage":3,
            "artifact_role":"adaptation_policy_rebind","status":"FROZEN_REBOUND_POLICY_ONLY",
            "authorization_scope":"PROTOCOL_REBIND_NO_TRAINING_OR_EVALUATION","producer":entry(SELF),
            "specification_sha256":SPEC_SHA256,"specification_seal_sha256":SEAL_SHA256,"cohort_static_sha256":COHORT_SHA256,
            "joint_upstream_closure":closure,"upstream_sha256":UPSTREAM,"historical_policy_provenance":entry(OLD3),
            "selected_scale":"LOG1P","selected_graph_gru":{"config_id":"GGRU_K04_H032_D00","graph_k":4,"hidden_size":32,"dropout":0.0},
            "selected_vanilla_gru":{"config_id":"VGRU_H032_D00","hidden_size":32,"dropout":0.0},
            "policy":POLICY,"optimizer_betas":[0.9,0.999],"optimizer_epsilon":1e-8,
            "checkpoint_rule":"final_fixed_step_only","budget_schedule":windows,
            "zero_budget":"unchanged_source_checkpoint_reuse_not_a_fine_tuning_fit",
            "source_checkpoint_authority":"matching validated V2.2 source checkpoint only; no V2.1 checkpoint cross-loading",
            "development_cities":[129,194,438,467,476,532,619,658],"pseudo_targets":[532,476,619,658],
            "development_seeds":[17,29,43],"final_execution_seeds":None,
            "causal_history":{"policy":"IDEALIZED_COUNT_FEED_BENCHMARK","predictor_asof":"original forecast origin t",
                "fit_row_eligibility":"registered window at fit cutoff C, hour_end <= C, O_i(h;C)*Q_i(h;C)=1",
                "fitting_cutoff_changes_predictors":False,"retrospective_Y_shifted_into_predictors":False,
                "later_backfill_allowed":False,"window_extension_allowed":False,
                "empty_required_positive_budget_pool":"STOP_NON_ESTIMABLE_NO_FALLBACK",
                "training_key_gate":"existing require_training_keys with sealed keys, values, masks and origin-feature bindings",
                "prebudget_history":"allowed as causal predictors only; never additional adaptation targets",
                "status_evidence":"timestamp <= original predictor origin for predictors; timestamp <= C for fit eligibility",
                "timestamp_precision":"integer UTC microseconds; exact elapsed UTC hourly and weekly lags",
                "cohort":"fixed 799-station retrospectively selected research cohort; supplied static metadata frozen"},
            "data_api_sha256":dependencies,"boundaries_us":dict(contract.boundaries),
            "final_embargo":{"interval":"[HF,HT)","hours":593,"fitting_allowed":False},
            "final_no_updates_interval":"[HF,HE)","final_evaluation_interval":"[HT,HE)",
            "final_prediction_commitment_before_label_opening":True,
            "label_firewall":"DevelopmentRegistry rejects all final-target data and V2.1 predictive inputs. Final windows above are documentary only.",
            "final_target_labels_accessed":False,"final_experiment_started":False,"scientific_stage3_training_authorized":False,
            "new_development_hyperparameter_search":False,"new_stage3_development_execution_required":False,
            "stage3_policy_contradictions":[],
            "downstream_blockers":["FINAL_SEED_COUNT_UNRESOLVED_3_VS_5","SEED_X_TEMPORAL_BOOTSTRAP_AGGREGATION_UNRESOLVED"]}


def stage4_audit(stage3,old4):
    return {"schema_version":"1.0","protocol_version":"2.2","artifact_role":"stage4_scientific_gate_audit_not_execution_contract",
            "status":"BLOCKED_BY_SCIENTIFIC_AMBIGUITY","producer":entry(SELF),"stage3_rebind":stage3,
            "historical_method_seal":entry(OLD4),"upstream_sha256":UPSTREAM,
            "scientific_identity":"GRL-based multi-source domain-invariant pretraining",
            "backbone":{"config_id":"GGRU_K04_H032_D00","graph_k":4,"hidden_size":32,"dropout":0.0,"target":"LOG1P"},
            "historically_bound_method":old4["method"],"lambda_schedule":old4["lambda_schedule"],
            "candidate_grid":old4["candidate_grid"],"workload":old4["workload"],
            "historically_bound_selection_rule":old4["selection_rule"],
            "confirmed_primary":"Post-7-day/300-update adaptation count-space MAE; arithmetic mean of seeds 17/29/43 within fold, then equal arithmetic mean of four folds",
            "confirmed_tie_break_order":["lower fold dispersion","smaller lambda_max","constant schedule"],
            "missing_precommitments":[
                {"id":"STAGE4_PRIMARY_TIE_DEFINITION_UNSPECIFIED","missing":"No exact tie equality/quantization rule, precision, or rounding mode is registered for Stage 4.",
                 "evidence":["DEVELOPMENT_SELECTION_PROTOCOL.md:77-89","research/governance/stage4_adversarial_method_seal.py:126-134","historical method manifest selection_rule"],
                 "why_not_inherited":"The explicit four-decimal/ROUND_HALF_UP conventions belong to other stages. No Stage-4 authority explicitly imports them."},
                {"id":"STAGE4_FOLD_DISPERSION_UNDEFINED","missing":"lower_fold_dispersion names no exact statistic or numerical comparison/quantization convention.",
                 "evidence":["DEVELOPMENT_SELECTION_PROTOCOL.md:88","research/governance/stage4_adversarial_method_seal.py:130"],
                 "why_material":"Standard deviation, variance, and range are not an explicit interchangeable Stage-4 contract. Population versus sample SD has identical ranking with four folds but remains unspecified numerically."}
            ],
            "no_silent_resolution_applied":True,"historical_specification_fully_executable":False,
            "v2_2_pooling_binding":"Current training-example target-observation mask M_i(t;C) from the sealed fit snapshot at C; no historical mask, raw demand, pseudo-target, or retrospective evaluation mask in pooling/inference",
            "ordinary_graph_gru_baseline_unchanged":True,"stage2_rerun_required":False,
            "zero_shot_selection_evaluations":0,"post_adaptation_selection_evaluations":72,
            "stage4_implementation_created":False,"stage4_execution_scientifically_authorized":False,
            "stage4_deployment_package_created":False,"stage4_job_map_created":False,
            "required_resolution":"Explicit append-only Stage-4 selection clarification defining the primary tie rule and exact dispersion statistic/comparison before implementation validation and packaging; do not modify old seals or consult Stage-4 outcomes.",
            "real_scientific_training_updates":0,"real_scientific_evaluations":0,
            "final_target_labels_accessed":False,"final_experiment_started":False,
            "downstream_blockers":["FINAL_SEED_COUNT_UNRESOLVED_3_VS_5","SEED_X_TEMPORAL_BOOTSTRAP_AGGREGATION_UNRESOLVED"],
            "final_verdict":"STAGE4_BLOCKED_BY_SCIENTIFIC_AMBIGUITY"}


def verify():
    s=read(STAGE3);a=read(AUDIT);closure=read(CLOSURE);checks=[]
    def check(n,ok):
        require(ok,n);checks.append({"check":n,"status":"PASS"})
    check("all_eight_pinned_upstream_hashes",all(sha(p)==h for p,h in UPSTREAM.items()))
    check("stage3_contract_producer_and_dependencies",sha(s["producer"]["path"])==s["producer"]["sha256"] and
          all(sha(p)==h for p,h in s["data_api_sha256"].items()))
    check("joint_closure_binding",sha(CLOSURE)==s["joint_upstream_closure"]["sha256"] and closure["completed_fit_records"]==240 and closure["referenced_artifacts_hashed"]==675)
    check("stage3_exact_policy",s["policy"]==POLICY==read(OLD3)["policy"] and s["optimizer_betas"]==[0.9,0.999] and s["optimizer_epsilon"]==1e-8)
    check("five_budget_update_counts",{r["budget"]:r["updates"] for r in s["budget_schedule"]}==BUDGETS)
    c=Contract()
    for row in s["budget_schedule"]:
        q=FitRequest.registered(c,phase="development",kind="adaptation",budget=row["budget"],target_city=532)
        check("existing_V2_2_FitRequest_window_"+row["budget"],row["development_window_us"]==[q.start,q.cutoff])
    check("cutoff_embargo_and_no_updates",s["final_embargo"]["hours"]==(c.boundaries["HT"]-c.boundaries["HF"])//HOUR==593 and s["final_no_updates_interval"]=="[HF,HE)")
    check("origin_causality_and_no_window_extension",not s["causal_history"]["fitting_cutoff_changes_predictors"] and not s["causal_history"]["window_extension_allowed"] and not s["causal_history"]["later_backfill_allowed"])
    denied=0
    for rel in ("processed/protocol_v2_2/final_labels/never_open.npz","processed/protocol_v2_2/final_adaptation/never_open.npz"):
        try:path(rel)
        except PermissionError:denied+=1
    check("final_paths_rejected_before_access",denied==2)
    check("no_new_stage3_search_or_execution",not s["new_development_hyperparameter_search"] and not s["new_stage3_development_execution_required"] and not s["scientific_stage3_training_authorized"])
    check("stage4_preserves_historical_selection_without_invented_ties",a["historically_bound_selection_rule"]==read(OLD4)["selection_rule"] and len(a["missing_precommitments"])==2)
    check("stage4_144_fit_and_72_evaluation_accounting",a["workload"]=={"source_pretraining_fits":72,"adaptation_fits":72,"parameter_changing_fits":144,"post_adaptation_evaluations":72,"zero_shot_selection_evaluations":0})
    check("scientific_ambiguity_closes_execution_and_package_gates",not a["stage4_execution_scientifically_authorized"] and not a["stage4_deployment_package_created"] and not a["stage4_job_map_created"])
    check("downstream_final_seed_conflict_retained","FINAL_SEED_COUNT_UNRESOLVED_3_VS_5" in s["downstream_blockers"] and s["final_execution_seeds"] is None)
    check("stage3_binding_in_stage4_audit",sha(STAGE3)==a["stage3_rebind"]["sha256"])
    check("all_inspected_governance_files_preserved",all(sha(p)==h for p,h in a["inspected_governance_sha256"].items()))
    return {"status":"PASS","purpose":"READ_ONLY_GOVERNANCE_REBIND_VERIFICATION_NOT_STAGE4_IMPLEMENTATION_TESTS",
            "check_count":len(checks),"checks":checks,"real_scientific_training_updates":0,"real_scientific_evaluations":0,
            "stage4_implementation_tests_run":0,"controller_startup_integration":"NOT_RUN_SCIENTIFIC_GATE_BLOCKED",
            "metadata_canonicalization_test":"NOT_RUN_NO_STAGE4_IMPLEMENTATION_CREATED",
            "final_target_labels_accessed":False,"final_experiment_started":False,"files_written_by_verifier":[]}


def create():
    require(not any(path(p).exists() for p in (CLOSURE,STAGE3,AUDIT)),"Append-only governance destinations already exist")
    # Validate every required upstream result and historical source BEFORE writes.
    joint=synchronize();old3,old4=historical_bindings()
    require(old4["workload"]["parameter_changing_fits"]==144,"Historical Stage-4 workload contradiction")
    closure=write(CLOSURE,joint)
    stage3=write(STAGE3,stage3_contract(closure))
    audit=stage4_audit(stage3,old4)
    audit["inspected_governance_sha256"]={p:h for p,h in checked.items() if not p.startswith(("processed/","research/results/stage1_scale_loss_v2_2_a40/","research/results/stage2_graphgru_selection_v2_2_a40/","research/results/stage2b_vanilla_fairness_v2_2_a40/"))}
    write(AUDIT,audit)
    validation=write(OUT+"verification_report.json",verify())
    print(json.dumps({"stage3_rebind":stage3,"joint_closure":closure,"scientific_audit":entry(AUDIT),"verification":validation,
                      "verdict":audit["final_verdict"]},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("create","verify"));args=p.parse_args()
    if args.mode=="create":create()
    else:print(json.dumps(verify(),indent=2))
