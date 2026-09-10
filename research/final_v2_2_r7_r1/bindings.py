"""Independent canonical scientific execution bindings for frozen R7 jobs."""
from __future__ import annotations
import random
from typing import Any
from . import core,data

SCHEMA="final_v2_2_r7_r1.scientific_execution_binding.1"
CHECKPOINT_SCHEMA="final_v2_2_r7_r1.checkpoint.2"
COMPLETION_SCHEMA="final_v2_2_r7_r1.completion.2"
CONTROLLER_LOCK_SCHEMA="final_v2_2_r7_r1.controller_lock.1"
JOB_CLAIM_SCHEMA="final_v2_2_r7_r1.job_claim.1"

PROTOCOL_HASHES={
    "FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md":core.FROZEN["FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md"],
    "final_evaluation_protocol_v2_2.json":core.FROZEN["final_evaluation_protocol_v2_2.json"]}
GRL_CLARIFICATION_HASHES={
    "FINAL_GRL_SOURCE_DOMAIN_CARDINALITY_CLARIFICATION_V2_2.md":core.FROZEN["FINAL_GRL_SOURCE_DOMAIN_CARDINALITY_CLARIFICATION_V2_2.md"],
    "final_grl_source_domain_cardinality_v2_2.json":core.FROZEN["final_grl_source_domain_cardinality_v2_2.json"]}

def grl_policy()->dict[str,Any]:
    return {"lambda":0.5,"schedule":"constant",
        "discriminator":["Linear(32,64)","ReLU","Dropout(0.10)","Linear(64,8)"],"output_dim":8,
        "domain_map":{str(city):domain for city,domain in core.DOMAIN_MAP.items()}}

def model_binding(job:dict[str,Any])->dict[str,Any]:
    architecture=job["architecture"]
    if architecture not in ("VGRU_H032_D00","GGRU_K04_H032_D00"): raise RuntimeError("unregistered architecture")
    grl_branch=job["method"].startswith("grl_")
    return {"architecture":architecture,"model_type":"vanilla_gru" if architecture.startswith("VGRU") else "graph_gru",
        "hidden_size":32,"dropout":0.0,"graph_k":None if architecture.startswith("VGRU") else 4,
        "output_mode":job["output_mode"],"grl_source_policy":grl_policy() if grl_branch else None,
        "grl_discriminator_present_in_checkpoint":bool(grl_branch and job["category"]=="source")}

def _source_schedule(job):
    values=list(core.SOURCES)*(12000//len(core.SOURCES))
    random.Random(core.seed_for(job,"source-city-schedule")).shuffle(values)
    return values

def optimizer_binding(job:dict[str,Any])->dict[str,Any]:
    category=job["category"]; schedule_hash=None
    if category=="pooled":
        learning_rate={"source":1e-3,"target":2e-4}; schedule="frozen_balanced_source_plus_target_token_shuffle"
        tokens=[["source",c] for c in _source_schedule(job)]+[["target",job["target_city_id"]] for _ in range(core.UPDATES[job["budget"]])]
        random.Random(core.seed_for(job,"pooled-token-shuffle")).shuffle(tokens); schedule_hash=core.canonical_hash(tokens)
    elif category in ("target_only","adaptation"): learning_rate=2e-4; schedule="constant"
    elif category=="source": learning_rate=1e-3; schedule="constant"; schedule_hash=core.canonical_hash(_source_schedule(job))
    else: raise RuntimeError("unregistered job category")
    return {"name":"AdamW","learning_rate":learning_rate,"learning_rate_schedule":schedule,
        "schedule_sha256":schedule_hash,"betas":[0.9,0.999],"eps":1e-8,"weight_decay":1e-4,
        "batch_size_city_hours":16,"global_gradient_clip":1.0,"required_updates":job["updates"],
        "early_stopping":False,"partial_resume":False}

def runtime_binding(job:dict[str,Any],device:str="cuda")->dict[str,Any]:
    return core.runtime_contract()|{"device":device,"seed":core.seed_for(job,"model-initialization")}

def authorities(package_sha:str)->dict[str,Any]:
    return {"job_manifest_sha256":core.FROZEN[core.JOB_MANIFEST],"package_manifest_sha256":package_sha,
        "code_manifest_sha256":core.sha(core.CODE_MANIFEST),"implementation_contract_sha256":core.sha(core.IMPLEMENTATION_CONTRACT),
        "final_data_manifest_sha256":core.FROZEN[core.DATA_MANIFEST],"protocol_hashes":PROTOCOL_HASHES,
        "grl_clarification_hashes":GRL_CLARIFICATION_HASHES,"frozen_authorities":dict(sorted(core.FROZEN.items()))}

def expected(job:dict[str,Any],package_sha:str,input_checkpoint:dict[str,Any]|None,device:str="cuda")->dict[str,Any]:
    value={"schema_version":SCHEMA,
        "job":{"job_id":job["job_id"],"method":job["method"],"phase":job["category"],"seed":job["seed"],
               "target":job["target_city_id"],"budget":job["budget"]},"authorities":authorities(package_sha),
        "model":model_binding(job),"data":data.structural_job_binding(job),"optimizer":optimizer_binding(job),
        "input_checkpoint":input_checkpoint,"runtime":runtime_binding(job,device)}
    value["scientific_execution_binding_sha256"]=core.canonical_hash(value)
    return value