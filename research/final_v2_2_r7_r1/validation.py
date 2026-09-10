"""Independent, label-free package, completion, and postrun validation."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from . import core

def _validate_job_universe(jobs):
    if len(jobs) != 410 or len({j["job_id"] for j in jobs}) != 410: raise RuntimeError("410 unique jobs required")
    counts = Counter(j["category"] for j in jobs)
    if counts != Counter(source=10, adaptation=160, target_only=160, pooled=80): raise RuntimeError("DAG category drift")
    methods=Counter(j["method"] for j in jobs)
    required_methods=Counter({"ordinary_source_graphgru":5,"grl_l50_constant_source_graphgru":5,
        "ordinary_source_adapted":80,"grl_l50_constant_adapted":80,"target_only_graph":80,
        "target_only_vanilla":80,"pooled_graph":80})
    if methods != required_methods: raise RuntimeError("seven-family distribution drift")
    expected=set()
    for seed in core.SEEDS:
        expected.add(("source","ordinary_source_graphgru",seed,None,None,12000,"GGRU_K04_H032_D00"))
        expected.add(("source","grl_l50_constant_source_graphgru",seed,None,None,12000,"GGRU_K04_H032_D00"))
        for target in core.TARGETS:
            for budget in core.NONZERO_BUDGETS:
                updates=core.UPDATES[budget]
                expected |= {("adaptation","ordinary_source_adapted",seed,target,budget,updates,"GGRU_K04_H032_D00"),
                    ("adaptation","grl_l50_constant_adapted",seed,target,budget,updates,"GGRU_K04_H032_D00"),
                    ("target_only","target_only_graph",seed,target,budget,updates,"GGRU_K04_H032_D00"),
                    ("target_only","target_only_vanilla",seed,target,budget,updates,"VGRU_H032_D00"),
                    ("pooled","pooled_graph",seed,target,budget,12000+updates,"GGRU_K04_H032_D00")}
    actual={(j["category"],j["method"],j["seed"],j["target_city_id"],j["budget"],j["updates"],j["architecture"]) for j in jobs}
    if actual != expected: raise RuntimeError("mechanically regenerated 410-job universe mismatch")
    for job in jobs:
        if job["output_mode"]!="LOG1P_TARGET_MAE" or job["attempt_policy"]!="fresh_uuid_from_update_0_no_partial_resume":
            raise RuntimeError("job scientific contract drift")
        if job["category"]=="adaptation":
            prefix="ordinary_source_seed-" if job["method"]=="ordinary_source_adapted" else "grl_l50_constant_source_seed-"
            if job["depends_on"]!=prefix+str(job["seed"]): raise RuntimeError("adaptation lineage drift")
        elif job["depends_on"] is not None: raise RuntimeError("unexpected dependency")
    return counts

def package_check(package_sha: str):
    core.verify_frozen()
    if not isinstance(package_sha, str) or len(package_sha) != 64 or core.sha(core.PACKAGE_MANIFEST) != package_sha:
        raise RuntimeError("explicit R7 package-manifest hash mismatch")
    package = core.read_json(core.PACKAGE_MANIFEST)
    if package.get("status") != "R7_R1_IMMUTABLE_PACKAGE" or package.get("phase_b_labels_included") is not False:
        raise RuntimeError("invalid R7 package identity")
    member_entry = package["package_member_manifest"]
    if core.sha(member_entry["path"]) != member_entry["sha256"]: raise RuntimeError("member manifest hash mismatch")
    members = core.read_json(member_entry["path"])
    seen = set()
    for item in members["files"]:
        if item["path"] in seen: raise RuntimeError("duplicate package member")
        seen.add(item["path"]); value = core.path(item["path"])
        if value.stat().st_size != item["bytes"] or core.sha256_path(value) != item["sha256"]:
            raise RuntimeError("package member mismatch: " + item["path"])
    for key in ("implementation_contract", "executable_code_manifest"):
        item = package[key]
        if core.sha(item["path"]) != item["sha256"]: raise RuntimeError(key + " mismatch")
    contract = core.read_json(package["implementation_contract"]["path"])
    if contract["package_member_manifest_sha256"] != member_entry["sha256"]: raise RuntimeError("contract/member closure mismatch")
    jobs = core.read_json(core.JOB_MANIFEST)["fits"]; _validate_job_universe(jobs)
    evaluations = core.read_json(core.EVALUATION_MANIFEST)["passes"]
    if len(evaluations) != 468 or len({x["evaluation_id"] for x in evaluations}) != 468: raise RuntimeError("468 evaluation passes required")
    if sum(len(x["reuse_labels"]) for x in evaluations) != 660: raise RuntimeError("660 reporting cells required")
    clarification = core.read_json("final_grl_source_domain_cardinality_v2_2.json")
    if clarification["final_grl_rule"]["final_all_source_output_dim"] != 8 or clarification["final_grl_rule"]["final_source_domain_classes"] != [
        {"domain_class": i, "city_id": city, "city": name} for i,(city,name) in enumerate(((129,"Dortmund"),(194,"Heidelberg"),(438,"Marburg"),(467,"Gießen"),(476,"Cardiff"),(532,"Bilbao"),(619,"Freiburg"),(658,"Göteborg")))]:
        raise RuntimeError("GRL clarification mapping drift")
    return {"status": "PASS", "package_members": len(members["files"]), "neural_jobs": 410,
            "evaluation_passes": 468, "reporting_cells": 660, "grl_output_dim": 8,
            "phase_b_targets_materialized": False, "final_predictions_generated": False,
            "final_experiment_started": False, "optimizer_updates": 0,
            "model_forward_calls_on_final_labels": 0}

def runtime_check(strict: bool):
    if not strict:
        return {"mode": "STRUCTURAL_ONLY", "python": sys.version, "platform": platform.platform(), "a40_checked": False}
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    if sys.version_info[:2]!=(3,10) or str(torch.__version__)!="2.0.1+cu118" or np.__version__!="1.26.4":
        raise RuntimeError("strict preflight requires frozen TS9 Python 3.10 / torch 2.0.1+cu118 / NumPy 1.26.4")
    core.configure_torch(17, "cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "NVIDIA A40":
        raise RuntimeError("strict preflight requires exactly one NVIDIA A40")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                          "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
    if len(smi.splitlines()) != 1 or smi.split(",")[0].strip() != "NVIDIA A40": raise RuntimeError("nvidia-smi identity mismatch")
    return {"mode": "STRICT_UNIVERSITY_A40", "python": sys.version, "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "nvidia_smi": smi, "settings": core.runtime_contract()}

def preflight(package_sha: str, *, strict: bool = True):
    base = package_check(package_sha); runtime = runtime_check(strict)
    existing = validate_existing(package_sha)
    return {**base, "status": "PASS_R7_PREFLIGHT", "runtime": runtime, "existing": existing,
            "phase_b_authorized": False, "scientific_execution_started": False}

def dry_run(package_sha: str):
    base = package_check(package_sha); jobs = core.read_json(core.JOB_MANIFEST)["fits"]
    from .data import structural_job_binding
    bindings = [structural_job_binding(job)["binding_sha256"] for job in jobs]
    if len(bindings) != 410: raise RuntimeError("dry-run binding enumeration drift")
    return {**base, "status": "PASS_DRY_RUN", "jobs_bound": 410, "unique_binding_hashes": len(set(bindings)),
            "files_written": [], "scientific_optimizer_updates": 0, "model_forward_calls": 0}

def completion_path(job_id): return f"{core.OUT}/completed/{job_id}.json"

def _legal_parent(job: dict[str, Any], package_sha: str):
    if job["category"] != "adaptation": return None
    jobs=core.read_json(core.JOB_MANIFEST)["fits"]
    parents=[x for x in jobs if x["job_id"]==job["depends_on"]]
    if len(parents)!=1 or parents[0]["category"]!="source" or parents[0]["seed"]!=job["seed"]:
        raise RuntimeError("illegal adaptation parent in frozen manifest")
    parent=validate_one_completion(parents[0],package_sha)
    return parent["output_checkpoint"]

def _validate_checkpoint(job,record,expected,package_sha):
    import torch
    from . import bindings
    checkpoint=record["output_checkpoint"]; cp=core.path(checkpoint["path"])
    canonical_prefix=f"{core.OUT}/attempts/{job['job_id']}/{record['attempt_uuid']}/"
    if not checkpoint["path"].startswith(canonical_prefix): raise RuntimeError("writable checkpoint collision/cross-load")
    if set(checkpoint)!={"path","bytes","sha256"}: raise RuntimeError("checkpoint entry schema mismatch")
    if cp.stat().st_size!=checkpoint["bytes"] or core.sha256_path(cp)!=checkpoint["sha256"]: raise RuntimeError("checkpoint hash mismatch")
    if record["artifact_hashes"]!={"checkpoint":checkpoint["sha256"]}: raise RuntimeError("checkpoint artifact binding mismatch")
    payload=torch.load(cp,map_location="cpu",weights_only=False)
    allowed=({"schema_version","metadata","model_state","optimizer_state"},
             {"schema_version","metadata","model_state","optimizer_state","forecast_state"})
    if set(payload) not in allowed or payload["schema_version"]!=bindings.CHECKPOINT_SCHEMA: raise RuntimeError("checkpoint payload schema mismatch")
    metadata=payload["metadata"]
    required={"schema_version","job_id","job_manifest_sha256","package_manifest_sha256","code_manifest_sha256",
        "implementation_contract_sha256","final_data_manifest_sha256","protocol_hashes","grl_clarification_hashes",
        "scientific_execution_binding","scientific_execution_binding_sha256","seed","method","model_binding","data_binding",
        "input_checkpoint","runtime","optimizer","expected_updates","completed_updates","early_stopping","partial_resume"}
    if set(metadata)!=required: raise RuntimeError("checkpoint metadata schema mismatch")
    auth=expected["authorities"]
    checks={"schema_version":bindings.CHECKPOINT_SCHEMA,"job_id":job["job_id"],
        "job_manifest_sha256":auth["job_manifest_sha256"],"package_manifest_sha256":package_sha,
        "code_manifest_sha256":auth["code_manifest_sha256"],"implementation_contract_sha256":auth["implementation_contract_sha256"],
        "final_data_manifest_sha256":auth["final_data_manifest_sha256"],"protocol_hashes":auth["protocol_hashes"],
        "grl_clarification_hashes":auth["grl_clarification_hashes"],"scientific_execution_binding":expected,
        "scientific_execution_binding_sha256":expected["scientific_execution_binding_sha256"],"seed":job["seed"],
        "method":job["method"],"model_binding":expected["model"],"data_binding":expected["data"],
        "input_checkpoint":expected["input_checkpoint"],"runtime":expected["runtime"],"optimizer":expected["optimizer"],
        "expected_updates":job["updates"],"completed_updates":job["updates"],"early_stopping":False,"partial_resume":False}
    if metadata!=checks: raise RuntimeError("checkpoint semantic binding mismatch")
    if job["category"]=="source" and job["method"].startswith("grl_"):
        from .grl_final import FinalSourceInvariantGraphGRU
        model=FinalSourceInvariantGraphGRU()
    else:
        from research.models.common import NeuralModelConfig
        from research.models.graph_gru import GraphGRU
        from research.models.vanilla_gru import VanillaGRU
        cfg=NeuralModelConfig(model_type="vanilla_gru" if job["architecture"].startswith("VGRU") else "graph_gru",
            hidden_size=32,dropout=0.0,output_mode="log1p_target")
        model=VanillaGRU(cfg) if cfg.model_type=="vanilla_gru" else GraphGRU(cfg)
    model.load_state_dict(payload["model_state"],strict=True)
    if job["category"]=="source" and job["method"].startswith("grl_"):
        if "forecast_state" not in payload: raise RuntimeError("GRL forecast state absent")
        model.forecast.load_state_dict(payload["forecast_state"],strict=True)
    elif "forecast_state" in payload: raise RuntimeError("unexpected forecast-only state")
    state=payload["optimizer_state"]
    if not isinstance(state,dict) or set(state)!={"state","param_groups"} or not state["state"] or not state["param_groups"]:
        raise RuntimeError("optimizer state absent")
    steps=[]
    for item in state["state"].values():
        if "step" not in item: raise RuntimeError("optimizer step evidence absent")
        value=item["step"]; steps.append(int(value.item() if hasattr(value,"item") else value))
    if not steps or any(step!=job["updates"] for step in steps): raise RuntimeError("optimizer final-update provenance mismatch")
    for group in state["param_groups"]:
        if tuple(group.get("betas",()))!=(0.9,0.999) or float(group.get("eps",-1))!=1e-8 or float(group.get("weight_decay",-1))!=1e-4:
            raise RuntimeError("optimizer state configuration mismatch")
        lr=float(group.get("lr",-1)); expected_lr=expected["optimizer"]["learning_rate"]
        if isinstance(expected_lr,dict):
            if lr not in set(expected_lr.values()): raise RuntimeError("pooled optimizer terminal LR invalid")
        elif lr!=expected_lr: raise RuntimeError("optimizer learning rate mismatch")

def validate_completion_record(job:dict[str,Any],record:dict[str,Any],package_sha:str):
    from . import bindings
    if not isinstance(package_sha,str) or len(package_sha)!=64: raise RuntimeError("explicit package SHA required")
    required={"schema_version","status","success","job_id","job_manifest_sha256","package_manifest_sha256",
        "code_manifest_sha256","implementation_contract_sha256","final_data_manifest_sha256","protocol_hashes",
        "grl_clarification_hashes","scientific_execution_binding","scientific_execution_binding_sha256","method","phase",
        "seed","target","budget","attempt_uuid","model_binding","data_binding","input_checkpoint","output_checkpoint",
        "optimizer","optimizer_update_count","runtime_determinism","artifact_hashes","diagnostics","started_record",
        "completed_utc","early_stopping","partial_resume","final_target_labels_accessed","final_predictions_generated"}
    if set(record)!=required: raise RuntimeError("completion schema mismatch: "+job["job_id"])
    if record["schema_version"]!=bindings.COMPLETION_SCHEMA or record["status"]!="COMPLETED_VALID_FIT" or record["success"] is not True:
        raise RuntimeError("invalid completion state")
    auth=bindings.authorities(package_sha)
    literal={"job_id":job["job_id"],"job_manifest_sha256":auth["job_manifest_sha256"],
        "package_manifest_sha256":package_sha,"code_manifest_sha256":auth["code_manifest_sha256"],
        "implementation_contract_sha256":auth["implementation_contract_sha256"],
        "final_data_manifest_sha256":auth["final_data_manifest_sha256"],"protocol_hashes":auth["protocol_hashes"],
        "grl_clarification_hashes":auth["grl_clarification_hashes"],"method":job["method"],"phase":job["category"],
        "seed":job["seed"],"target":job["target_city_id"],"budget":job["budget"],
        "optimizer_update_count":job["updates"],"early_stopping":False,"partial_resume":False,
        "final_target_labels_accessed":False,"final_predictions_generated":False}
    for key,expected_value in literal.items():
        if record[key]!=expected_value: raise RuntimeError("completion authority/identity mismatch: "+key)
    try:
        import uuid
        parsed=uuid.UUID(record["attempt_uuid"])
        if str(parsed)!=record["attempt_uuid"]: raise ValueError
    except Exception as exc: raise RuntimeError("invalid completion attempt UUID") from exc
    parent=_legal_parent(job,package_sha)
    runtime=record["runtime_determinism"]
    if not isinstance(runtime,dict) or runtime.get("device")!="cuda": raise RuntimeError("production completion device mismatch")
    expected=bindings.expected(job,package_sha,parent,"cuda")
    if record["scientific_execution_binding"]!=expected or record["scientific_execution_binding_sha256"]!=expected["scientific_execution_binding_sha256"]:
        raise RuntimeError("scientific execution binding mismatch")
    if record["model_binding"]!=expected["model"]: raise RuntimeError("model binding mismatch")
    if record["data_binding"]!=expected["data"]: raise RuntimeError("data binding mismatch")
    if record["optimizer"]!=expected["optimizer"]: raise RuntimeError("optimizer binding mismatch")
    if runtime!=expected["runtime"]: raise RuntimeError("runtime binding mismatch")
    if record["input_checkpoint"]!=parent: raise RuntimeError("input checkpoint lineage mismatch")
    started=record["started_record"]; sp=core.path(started["path"])
    if set(started)!={"path","bytes","sha256"} or sp.stat().st_size!=started["bytes"] or core.sha256_path(sp)!=started["sha256"]:
        raise RuntimeError("started-record hash mismatch")
    start_value=json.loads(sp.read_text(encoding="utf-8"))
    if start_value.get("job")!=job or start_value.get("attempt_uuid")!=record["attempt_uuid"] or start_value.get("package_manifest_sha256")!=package_sha:
        raise RuntimeError("started-record semantic mismatch")
    if start_value.get("scientific_execution_binding_sha256")!=expected["scientific_execution_binding_sha256"] or start_value.get("partial_resume") is not False:
        raise RuntimeError("started-record execution binding mismatch")
    diagnostics=record["diagnostics"]
    if not isinstance(diagnostics,list) or not diagnostics or diagnostics[-1].get("step")!=job["updates"]:
        raise RuntimeError("final update diagnostic absent")
    _validate_checkpoint(job,record,expected,package_sha)
    return record

def validate_one_completion(job:dict[str,Any],package_sha:str):
    relative=completion_path(job["job_id"]); value=core.path(relative)
    if not value.exists(): raise FileNotFoundError(relative)
    return validate_completion_record(job,json.loads(value.read_text(encoding="utf-8")),package_sha)
def validate_existing(package_sha: str):
    package_check(package_sha)
    jobs = core.read_json(core.JOB_MANIFEST)["fits"]; expected = {x["job_id"]: x for x in jobs}
    directory = core.path(core.OUT + "/completed")
    found = list(directory.glob("*.json")) if directory.exists() else []
    unexpected = [p.name for p in found if p.stem not in expected]
    if unexpected: raise RuntimeError("unexpected/corrupt completion files: " + repr(unexpected))
    valid = []
    for value in found:
        validate_one_completion(expected[value.stem], package_sha); valid.append(value.stem)
    partial_root = core.path(core.OUT + "/attempts")
    partial = 0
    if partial_root.exists():
        partial = sum(1 for p in partial_root.glob("*/*/started.json") if p.parents[1].name not in valid)
    from .locks import probe
    lock_root=core.path(core.OUT+"/locks/jobs")
    active_claims=sorted(p.stem for p in lock_root.glob("*.lock") if probe(p)) if lock_root.exists() else []
    return {"status": "PASS_VALIDATE_EXISTING", "expected": 410, "completed": len(valid),
            "remaining": 410 - len(valid), "partial_attempts_not_counted": partial,
            "duplicates": 0, "corrupt": 0, "active_job_claims": active_claims, "completed_job_ids": sorted(valid)}

def postrun_validate(package_sha: str):
    result = validate_existing(package_sha)
    if result["completed"] != 410 or result["remaining"] != 0: raise RuntimeError("postrun requires 410 valid completions")
    jobs = core.read_json(core.JOB_MANIFEST)["fits"]
    counts = _validate_job_universe(jobs)
    if Counter(j["seed"] for j in jobs) != Counter({s:82 for s in core.SEEDS}): raise RuntimeError("seed distribution drift")
    if Counter(j["target_city_id"] for j in jobs if j["target_city_id"] is not None) != Counter({t:100 for t in core.TARGETS}): raise RuntimeError("target distribution drift")
    if Counter(j["budget"] for j in jobs if j["budget"] is not None) != Counter({b:100 for b in core.NONZERO_BUDGETS}): raise RuntimeError("budget distribution drift")
    methods=Counter(j["method"] for j in jobs)
    return {**result, "status": "PASS_POSTRUN_410", "categories": dict(counts), "methods": dict(methods),
            "prediction_generation_performed": False, "final_labels_accessed": False}

