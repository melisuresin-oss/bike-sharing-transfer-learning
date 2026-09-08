"""Hash-bound Stage-4 definitions and exact metadata normalization."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
import uuid
import numpy as np
from research.stage4_v2_2.selection import FOLDS,SEEDS,LAMBDAS,SCHEDULES,candidate_id
from research.stage4_v2_2.verify_selection import CLARIFICATION,CLARIFICATION_SHA
from research.v2_2.contract import Contract,DEVELOPMENT,HOUR,SPEC_SHA256,SEAL_SHA256,COHORT_SHA256
from research.models.common import NeuralModelConfig
from research.training.trainer import OptimizerConfig

ROOT=Path(__file__).resolve().parents[2]
GOV="research/results/stage4_source_invariance_v2_2/"
OUT="research/results/stage4_source_invariance_v2_2_a40/"
PACKAGE="deployment/stage4_v2_2_a40/"
CONTRACT=GOV+"scientific_contract.json"
JOBMAP=GOV+"job_map.json"
MANIFEST=PACKAGE+"BUNDLE_MANIFEST.json"
STAGE3="research/results/stage3_finetuning_policy_v2_2/stage3_finetuning_policy_manifest.json"
UPSTREAM={
 "research/results/stage1_scale_loss_v2_2_a40/stage1_decision_manifest.json":"8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed",
 "research/results/stage2_graphgru_selection_v2_2_a40/stage2_decision_manifest.json":"9cca604ec3496a28f4440d89a9d6d1184311fc507c35f27b428a3ca768a6eeca",
 "research/results/stage2b_vanilla_fairness_v2_2_a40/stage2b_decision_manifest.json":"9d85d6df6a509013f1b5411002246aee81c254fbde078f639293976674b4f36d",
 STAGE3:"f5cb2653dd097eb1288f419884778ae01e9034e9e1d88006b33beac4cff069b5",
 CLARIFICATION:CLARIFICATION_SHA}
DATA="processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json"
DATA_SHA="140dd8c53cf51f48be5cf66b70d162c9f2cc6a9fa0b1b7039496633e385a6ab6"
FIT="processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json"
CACHE="tmp/stage1_scale_loss_cache_v2_2/cache_manifest.json"
CACHE_SHA="836e7d483a76a2253b5ec63180e23683d22ba1160d170e290d83e68d5d79756d"
GRAPH="research/results/causal_history_audit_v2_2/graph_static_reconciliation.json"
PINNED_DATA={DATA:DATA_SHA,CACHE:CACHE_SHA,FIT:"467aefd170c789c6533a7f2f1ebc0f3c343944f8e2781c9e2e5e0927d200077d",
 GRAPH:"e425ce873da946a4973fb03a57ffd6ecf92027da82a4220f56b6231343ff9758"}
ENVIRONMENT="UNIVERSITY_A40"
SOURCE_UPDATES=12000
ADAPTATION_UPDATES=300
WORKERS=4
BLOCKERS=["FINAL_SEED_COUNT_UNRESOLVED_3_VS_5","FINAL_SEED_BOOTSTRAP_AGGREGATION_UNRESOLVED"]


def require(ok,message):
    if not ok:raise RuntimeError(message)
def utc():return datetime.now(timezone.utc).isoformat()
def path(rel,output=False):
    if not isinstance(rel,str) or "\\" in rel or ":" in rel or ".." in Path(rel).parts:raise PermissionError("Repository-relative path required")
    p=(ROOT/rel).resolve();p.relative_to(ROOT)
    if any(x in rel.lower() for x in ("final_labels","sealed_final","final_adaptation","final_features")):raise PermissionError("Final-target firewall")
    if output and not rel.startswith(OUT):raise PermissionError("Stage-4 output isolation")
    return p
def sha(rel):
    h=hashlib.sha256()
    with path(rel).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def read(rel):return json.loads(path(rel).read_text(encoding="utf-8"))
def entry(rel):return {"path":rel,"sha256":sha(rel),"bytes":path(rel).stat().st_size}
def bound(e):
    require(sha(e["path"])==e["sha256"],"Hash mismatch: "+e["path"])
    return path(e["path"])
def canonicalize(value):
    if isinstance(value,np.generic):return canonicalize(value.item())
    if isinstance(value,os.PathLike):return os.fspath(value)
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError("Nonfinite metadata")
        return value
    if isinstance(value,(tuple,list)):return [canonicalize(x) for x in value]
    if isinstance(value,dict):
        if not all(isinstance(k,str) for k in value):raise TypeError("String metadata keys required")
        return {k:canonicalize(v) for k,v in value.items()}
    raise TypeError("Unsupported metadata type: "+type(value).__name__)
def canonical_bytes(value):return json.dumps(canonicalize(value),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def metadata_equal(a,b):return canonical_bytes(a)==canonical_bytes(b)
def canonical_hash(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()
def write_bytes(rel,data):
    p=path(rel,True);p.parent.mkdir(parents=True,exist_ok=True)
    pending=p.with_name(p.name+".pending-"+uuid.uuid4().hex)
    with pending.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    require(not p.exists(),"Append-only destination exists: "+rel)
    pending.rename(p)
    return entry(rel)
def write(rel,value):return write_bytes(rel,(json.dumps(canonicalize(value),indent=2,allow_nan=False)+"\n").encode())
def model_config():return NeuralModelConfig(model_type="graph_gru",hidden_size=32,dropout=0.0,output_mode="log1p_target")
def optimizer_config(phase):
    require(phase in ("source","adaptation"),"Unknown phase")
    return OptimizerConfig(name="AdamW",learning_rate=1e-3 if phase=="source" else 2e-4,betas=(.9,.999),epsilon=1e-8,weight_decay=1e-4)
def seed_for(label,target,seed):
    # Candidate-independent source ordering matches the frozen ordinary baseline.
    prefix="stage1-v2.1" if label=="fine-tune" else "stage2-v2.1"
    return int.from_bytes(hashlib.sha256(f"{prefix}|{label}|{target}|{seed}".encode()).digest()[:4],"big")
def registered_jobs():
    fit=read(FIT);jobs=[]
    for lam in LAMBDAS:
        for schedule in SCHEDULES:
            cid=candidate_id(lam,schedule)
            for target in FOLDS:
                sources=[i for i in DEVELOPMENT if i!=target]
                src=next(x for x in fit["source_folds"] if x["target_city"]==target)
                adapt=next(x for x in fit["adaptations"] if x["city_id"]==target and x["budget"]=="7")
                for seed in SEEDS:
                    task=f"{cid}_target-{target}_seed-{seed}"
                    common={"task_id":task,"candidate_id":cid,"lambda_max":lam,"schedule":schedule,"pseudo_target":target,
                            "seed":seed,"source_cities":sources,"domain_classes":{str(i):j for j,i in enumerate(sources)},
                            "source_schedule_seed":seed_for("source-city-schedule",target,seed),
                            "source_anchor_seed":seed_for("source-anchor-sampling",target,seed),"adaptation_seed":seed_for("fine-tune",target,seed)}
                    jobs.append({**common,"job_id":task+"_source","phase":"source","updates":12000,"depends_on":None,"snapshot":src})
                    jobs.append({**common,"job_id":task+"_7d","phase":"adaptation","updates":300,"depends_on":task+"_source","snapshot":adapt})
    return jobs
def validate_jobs(jobs):
    require(metadata_equal(jobs,registered_jobs()),"Immutable Stage-4 job definitions differ")
    require(len(jobs)==144 and len({j['job_id'] for j in jobs})==144,"144 unique immutable fits")
    for phase in ("source","adaptation"):require(sum(j["phase"]==phase for j in jobs)==72,"72 fits per phase")
