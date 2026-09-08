"""Read-only postrun verifier: persisted tensors, predictions, metrics and seals only.

No model is constructed, no forward/backward pass or optimizer step is executed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from decimal import Decimal, ROUND_HALF_UP
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/"tmp/neural_build/pydeps"),str(ROOT)]
import numpy as np
import torch
from research.development_data_v2_2.registry import DevelopmentRegistry

OUT="research/results/stage1_scale_loss_v2_2/"
DATA="processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json"
DATA_SHA="140dd8c53cf51f48be5cf66b70d162c9f2cc6a9fa0b1b7039496633e385a6ab6"
TARGETS=(532,476,619,658)
SEEDS=(17,29,43)
CANDIDATES=("RAW","LOG1P")
REGIMES=("zero_shot","7d_fine_tune")


def path(rel):
    if not isinstance(rel,str) or "\\" in rel or ":" in rel or ".." in Path(rel).parts:
        raise PermissionError("Invalid repository-relative path")
    p=(ROOT/rel).resolve(); p.relative_to(ROOT)
    if "final_labels" in p.as_posix().lower() or "sealed_final_evaluation_labels" in p.name.lower():
        raise PermissionError("Final-label firewall")
    return p


def sha(rel):
    h=hashlib.sha256()
    with path(rel).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def read(rel):
    return json.loads(path(rel).read_text(encoding="utf-8"))


def canonical(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--decision-sha256",required=True)
    args=parser.parse_args()
    checks=[]
    def check(name,ok):
        checks.append({"check":name,"status":"PASS" if ok else "FAIL"})
        if not ok:raise ValueError(name)
    result={"status":"FAIL","purpose":"READ_ONLY_STAGE1_POSTRUN_VERIFICATION","checks":checks,
            "models_constructed":0,"model_forward_calls":0,"training_steps":0,"files_written":[],
            "final_target_evaluation_labels_accessed":False}
    try:
        check("external_decision_binding",sha(OUT+"stage1_decision_manifest.json")==args.decision_sha256)
        decision=read(OUT+"stage1_decision_manifest.json")
        contract=read(OUT+"scientific_manifest.json"); jm=read(OUT+"job_map.json"); pf=read(OUT+"preflight_report.json")
        ledger=read(OUT+"hash_ledger.json")
        check("preflight_pass",pf["status"]=="PASS" and pf["scientific_training_steps"]==0 and pf["check_count"]==24
              and all(r["status"]=="PASS" for r in pf["checks"]))
        check("immutable_job_map",sha(OUT+"job_map.json")==pf["job_map"]["sha256"]==decision["job_map_sha256"])
        check("scientific_manifest_binding",sha(OUT+"scientific_manifest.json")==decision["scientific_manifest_sha256"])
        check("data_version_binding",sha(DATA)==DATA_SHA==contract["data_manifest_sha256"]==decision["development_data_manifest_sha256"])
        for p,h in {**contract["code_sha256"],**contract["design_source_sha256"],**ledger["files"]}.items():
            check("file_hash:"+p,sha(p)==h)
        check("exact_frozen_design",contract["candidates"]==list(CANDIDATES) and contract["seeds"]==list(SEEDS) and
              contract["folds"]==list(TARGETS) and contract["regimes"]==list(REGIMES) and
              contract["source_updates"]==12000 and contract["adaptation_updates"]==300 and contract["batch_size"]==16 and
              contract["graph_k"]==8 and contract["model"]["hidden_size"]==64 and contract["model"]["dropout"]==0.1)
        check("expected_48_fits",len(jm["jobs"])==48 and len(jm["tasks"])==24 and len({x["id"] for x in jm["jobs"]})==48)
        expected={(a,b,c) for a in CANDIDATES for b in TARGETS for c in SEEDS}
        check("no_substituted_tasks",{(t["candidate"],t["target"],t["seed"]) for t in jm["tasks"]}==expected)
        actualfiles={p.relative_to(ROOT).as_posix() for p in path(OUT+"jobs").glob("*.json")}
        expectedfiles={OUT+f"jobs/{t['task_id']}.json" for t in jm["tasks"]}
        check("no_missing_or_unregistered_job_record",actualfiles==expectedfiles)
        reg=DevelopmentRegistry(ROOT,DATA,DATA_SHA)
        cache=read("tmp/stage1_scale_loss_cache_v2_2/cache_manifest.json")
        check("cache_manifest_binding",sha("tmp/stage1_scale_loss_cache_v2_2/cache_manifest.json")==contract["cache_manifest_sha256"]
              and cache["protocol_version"]=="2.2" and not cache["retrospective_labels_in_cache"])
        for e in cache["cities"].values():
            for a in e["arrays"].values():check("V2_2_cache_bytes:"+a["path"],sha(a["path"])==a["sha256"])
        labels={i:reg.retrospective_labels(i) for i in TARGETS}
        rows=[];completed=[];source_count=0;adapt_count=0
        for task in jm["tasks"]:
            rel=OUT+f"jobs/{task['task_id']}.json"; record=read(rel)
            check("registered_completed_task:"+task["task_id"],record["status"]=="COMPLETED" and record["task"]==task and
                  record["job_map_sha256"]==decision["job_map_sha256"] and record["data_manifest_sha256"]==DATA_SHA)
            check("source_exclusion:"+task["task_id"],set(task["source_cities"])==set(reg.panels)-{task["target"]} and len(task["source_cities"])==7)
            previous=None
            for phase in ("source","adaptation"):
                ee=[e for e in record["executions"] if e["phase"]==phase]
                check("one_execution:"+task["task_id"]+phase,len(ee)==1)
                e=ee[0];updates=12000 if phase=="source" else 300;lr=1e-3 if phase=="source" else 2e-4
                completed.append(e["job_id"])
                source_count+=int(phase=="source");adapt_count+=int(phase=="adaptation")
                check("exact_updates:"+e["job_id"],e["updates_completed"]==updates and sum(e["city_updates"].values())==updates)
                cities=task["source_cities"] if phase=="source" else [task["target"]]
                check("update_city_ownership:"+e["job_id"],set(map(int,e["city_updates"]))==set(cities)
                      and (phase!="source" or max(e["city_updates"].values())-min(e["city_updates"].values())<=1))
                check("optimizer_configuration:"+e["job_id"],e["optimizer"]=={"name":"AdamW","learning_rate":lr,
                      "betas":[0.9,0.999],"epsilon":1e-8,"weight_decay":1e-4} and e["optimizer_state_reset"])
                cp=e["checkpoint"];check("checkpoint_bytes:"+e["job_id"],cp["path"].startswith(OUT+"checkpoints/") and sha(cp["path"])==cp["sha256"])
                payload=torch.load(path(cp["path"]),map_location="cpu",weights_only=True)
                meta=payload["metadata"]
                check("checkpoint_V2_2_identity:"+e["job_id"],meta["protocol_version"]=="2.2" and meta["data_manifest_sha256"]==DATA_SHA
                      and meta["artifact_identity"]["city_ids"]==sorted(cities) and meta["updates_completed"]==updates
                      and meta["training_snapshot"]==e["training_snapshot"] and meta["job_map_sha256"]==decision["job_map_sha256"])
                check("checkpoint_parameters_finite:"+e["job_id"],sum(v.numel() for v in payload["model_state"].values())==12939
                      and all(torch.isfinite(v).all().item() for v in payload["model_state"].values()))
                opt=payload["optimizer_state"];states=opt["state"]
                check("optimizer_actual_steps:"+e["job_id"],len(states)==len(payload["model_state"]) and
                      all(int(v["step"].item())==updates for v in states.values()) and len(opt["param_groups"])==1
                      and len(opt["param_groups"][0]["params"])==len(states))
                check("final_checkpoint_roundtrip:"+e["job_id"],e["checkpoint_roundtrip_max_abs"]==0.0)
                check("deterministic_execution:"+e["job_id"],e["runtime"]["deterministic_algorithms"] and
                      e["runtime"]["device"]=="cpu" and e["runtime"]["threads"]==2 and not e["runtime"]["cudnn_benchmark"])
                check("finite_training_logs:"+e["job_id"],all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm_before_clip"])
                      and r["valid_targets"]>0 and r["learning_rate"]==lr for r in e["logs"]) and e["logs"][-1]["step"]==updates)
                if phase=="adaptation":check("source_checkpoint_dependency:"+e["job_id"],e["source_checkpoint"]==previous)
                previous=cp
            for regime in REGIMES:
                rr=[r for r in record["metrics"] if r["regime"]==regime]
                check("one_evaluation_record:"+task["task_id"]+regime,len(rr)==1)
                r=rr[0]; p=r["prediction"];check("prediction_bytes:"+p["path"],p["path"].startswith(OUT+"predictions/") and sha(p["path"])==p["sha256"])
                with np.load(path(p["path"]),allow_pickle=False) as z:
                    pred=z["prediction_count"];ts=z["origin_us"];ids=z["station_ids"]
                lab=labels[task["target"]]; keep=(lab["origin_us"]>=reg.contract.boundaries["HD"])&(lab["origin_us"]<reg.contract.boundaries["HF"])
                check("complete_immutable_prediction_grid:"+task["task_id"]+regime,np.array_equal(ts,lab["origin_us"][keep])
                      and np.array_equal(ids,lab["station_ids"]) and pred.shape==(1440,len(ids)) and np.isfinite(pred).all() and np.all(pred>=0))
                check("prediction_key_binding:"+task["task_id"]+regime,r["prediction_key_sha256"]==canonical(
                      {"city":task["target"],"origins":list(map(int,ts)),"stations":list(map(int,ids))}))
                mask=lab["observed"][keep];y=lab["count"][keep][mask].astype(np.float64);yp=pred[mask].astype(np.float64)
                err=np.abs(y-yp);station=np.broadcast_to(ids[None,:],mask.shape)[mask]
                mae=float(err.mean());rmse=float(np.sqrt(np.mean((y-yp)**2)));wape=float(err.sum()/y.sum()) if y.sum() else None
                macro=float(np.mean([err[station==s].mean() for s in np.unique(station) if np.sum(station==s)>=24]))
                check("persisted_metric_reconciliation:"+task["task_id"]+regime,r["n"]==len(y) and r["mae"]==mae and r["rmse"]==rmse
                      and r["wape"]==wape and r["station_macro_mae"]==macro)
                check("candidate_seed_fold_identity:"+task["task_id"]+regime,r["candidate"]==task["candidate"]
                      and r["target"]==task["target"] and r["seed"]==task["seed"] and not r["final_labels_accessed"])
                rows.append(r)
            check("task_firewall:"+task["task_id"],not record["final_labels_accessed"] and not record["stage2_started"] and not record["stage2b_started"])
        check("all_fits_exactly_once",source_count==24 and adapt_count==24 and len(completed)==len(set(completed))==48
              and set(completed)=={j["id"] for j in jm["jobs"]})
        aggregates={}
        for candidate in CANDIDATES:
            cells=[]
            for target in TARGETS:
                for regime in REGIMES:
                    rr=[r for r in rows if r["candidate"]==candidate and r["target"]==target and r["regime"]==regime]
                    check(f"three_seeds:{candidate}:{target}:{regime}",sorted(r["seed"] for r in rr)==list(SEEDS))
                    cells.append(statistics.fmean(r["mae"] for r in rr))
            mean=statistics.fmean(cells);sd=statistics.pstdev(cells)
            aggregates[candidate]={"mean_count_space_MAE":mean,"rounded_four_decimals":str(Decimal(str(mean)).quantize(Decimal("0.0001"),rounding=ROUND_HALF_UP)),
                                   "population_SD_across_8_cells":sd}
        a,b=aggregates["RAW"],aggregates["LOG1P"]
        if a["rounded_four_decimals"]!=b["rounded_four_decimals"]:
            winner=min(CANDIDATES,key=lambda c:aggregates[c]["mean_count_space_MAE"])
        elif a["population_SD_across_8_cells"]!=b["population_SD_across_8_cells"]:
            winner=min(CANDIDATES,key=lambda c:aggregates[c]["population_SD_across_8_cells"])
        else:winner="RAW"
        summary=read(OUT+"candidate_summary.json")
        check("independent_selection_calculation",decision["selected_candidate"]==winner==summary["selected"]
              and decision["selection"]["candidates"]==aggregates==summary["candidates"])
        check("all_candidate_results_retained",summary["per_candidate_fold_seed_regime"]==rows)
        check("selection_rule_frozen",read(OUT+"selection_rule.json")==contract["selection"]==summary["selection_rule"])
        historical=read("research/results/causal_history_implementation_v2_2/preservation_before.json")
        for p,h in historical["files"].items():check("historical_preservation:"+p,sha(p)==h["sha256"])
        check("later_stages_not_started",not path("research/results/stage2_graphgru_selection_v2_2").exists()
              and not path("research/results/stage2b_vanilla_selection_v2_2").exists()
              and not path("research/results/stage4_adversarial_method_v2_2").exists())
        for prohibited in ("processed/protocol_v2_1/final_labels/SEALED_final_evaluation_labels.parquet",
                           "processed/protocol_v2_2/final_labels/x"):
            try:path(prohibited)
            except PermissionError:check("firewall_reject:"+prohibited,True)
            else:check("firewall_reject:"+prohibited,False)
        result.update({"status":"PASS","source_fits":source_count,"adaptation_fits":adapt_count,"evaluation_records":len(rows),
                       "selected_candidate":winner,"aggregates":aggregates,"historical_files_preserved":len(historical["files"]),
                       "decision_sha256":args.decision_sha256,"verdict":"A. STAGE1_V2_2_FROZEN_READY_FOR_STAGE2_STAGE2B_AUTHORIZATION"})
    except (OSError,ValueError,KeyError,AssertionError,RuntimeError) as exc:
        result["error"]=str(exc);result["verdict"]="B. STAGE1_V2_2_BLOCKED_BY_LISTED_FAILURES"
    result["check_count"]=len(checks)
    result["verifier_sha256"]=sha("research/scripts/verify_stage1_v2_2.py")
    print(json.dumps(result,indent=2,ensure_ascii=True))
    return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())

