"""Stage-4 A40 execution, commitment validation and read-only postrun checks."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid

if os.environ.get("CUBLAS_WORKSPACE_CONFIG",":4096:8")!=":4096:8":raise RuntimeError("Conflicting CuBLAS deterministic configuration")
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["OMP_NUM_THREADS"]="2";os.environ["MKL_NUM_THREADS"]="2"
import numpy as np
import torch
from research.stage4_v2_2 import core as c,method,selection
from research.stage4_v2_2.sampler import EqualCitySampler
from research.stage4_v2_2.verify_selection import run_checks
from research.stage1_v2_2.core import City as FrozenCity
from research.models.graph_gru import GraphGRU
from research.development_data_v2_2.registry import DevelopmentRegistry
from research.v2_2.snapshots import FitRequest
from research.v2_2.artifacts import ArtifactIdentity
from research.training.reproducibility import set_seed
from research.evaluation.metrics import compute_metrics

SETTINGS={"CUBLAS_WORKSPACE_CONFIG":":4096:8","deterministic_algorithms":True,"cuda_matmul_tf32":False,
          "cudnn_benchmark":False,"cudnn_deterministic":True,"cudnn_tf32":False}


def configure_runtime():
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def runtime():
    configure_runtime()
    py=Path(os.environ["USERPROFILE"])/"bike_env/Scripts/python.exe"
    c.require(Path(sys.executable).resolve()==py.resolve(),"Required USERPROFILE\\bike_env\\Scripts\\python.exe")
    c.require(sys.version_info[:2]==(3,10) and str(torch.__version__)=="2.0.1+cu118" and np.__version__=="1.26.4","Frozen TS9 runtime versions")
    for mod in (torch,np):c.require(py.parent.parent.resolve() in Path(mod.__file__).resolve().parents,"Dependencies must load from bike_env")
    c.require(torch.cuda.is_available() and torch.cuda.device_count()==1,"One CUDA device required")
    c.require(torch.cuda.get_device_name(0)=="NVIDIA A40","Actual NVIDIA A40 required")
    smi=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader,nounits"],check=True,capture_output=True,text=True).stdout.strip()
    c.require(len(smi.splitlines())==1 and smi.split(",")[0].strip()=="NVIDIA A40","nvidia-smi A40 identity")
    settings={"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"],
              "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
              "cuda_matmul_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_benchmark":torch.backends.cudnn.benchmark,
              "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_tf32":torch.backends.cudnn.allow_tf32}
    c.require(settings==SETTINGS,"Deterministic configuration")
    return {"execution_environment":c.ENVIRONMENT,"python_executable":str(py.resolve()),"python":sys.version,
            "torch":str(torch.__version__),"numpy":np.__version__,"cuda":torch.version.cuda,
            "cudnn":torch.backends.cudnn.version(),"device":"NVIDIA A40","nvidia_smi":smi,"settings":settings}


@contextmanager
def execution_lock(name):
    import msvcrt
    p=c.path(c.OUT+"locks/"+name+".lock",True);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a+b") as f:
        if p.stat().st_size==0:f.write(b"0");f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError("Already active: "+name)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def verify_package(manifest_sha):
    c.require(c.sha(c.MANIFEST)==manifest_sha,"External package manifest hash")
    m=c.read(c.MANIFEST)
    c.require(m["execution_environment"]==c.ENVIRONMENT and m["default_workers"]==4,"A40 package identity")
    for p,h in m["files"].items():c.require(c.sha(p)==h,"Package member mismatch: "+p)
    for p,h in {**c.UPSTREAM,**c.PINNED_DATA}.items():c.require(c.sha(p)==h,"Upstream or V2.2 data hash mismatch: "+p)
    for p,h in c.read(c.CLARIFICATION)["upstream_sha256"].items():c.require(c.sha(p)==h,"Clarification upstream closure")
    c.require(c.sha(c.CONTRACT)==m["scientific_contract_sha256"] and c.sha(c.JOBMAP)==m["job_map_sha256"],"Contract/job-map hash")
    c.require(c.read(c.CONTRACT)["prospective_selection_clarification"]["sha256"]==c.CLARIFICATION_SHA,"Clarification binding")
    for p,h in c.read(c.CONTRACT)["implementation_sha256"].items():c.require(c.sha(p)==h,"Sealed implementation bytes")
    c.require(run_checks()["status"]=="PASS","Independent prospective ranking verifier")
    jobs=c.read(c.JOBMAP)["jobs"];c.validate_jobs(jobs)
    cache=c.read(c.CACHE);reg=DevelopmentRegistry(c.ROOT,c.DATA,c.DATA_SHA)
    impl="research/results/causal_history_implementation_v2_2/implementation_manifest.json"
    c.require(c.sha(impl)=="a1fe0ee486c55b1348e92d91edd42289dd49ce982129cdaa544691eafd605a9d","Accepted causal implementation")
    for p,h in c.read(impl)["implementation_files_sha256"].items():
        if p.startswith("research/v2_2/") and p.endswith(".py"):c.require(c.sha(p)==h,"Unchanged causal-history API")
    c.require(c.sha("research/development_data_v2_2/registry.py")=="b5a142bbe1052cbc98b6fad8e927138b6cdbddf25ea6558b26e794d983b21051","Accepted V2.2 registry")
    la=reg.json_artifact("processed/protocol_v2_2/development_labels/retrospective_label_manifest.json")
    for target in c.FOLDS:
        e=next(e for e in la["labels"] if e["city_id"]==target)
        c.require(m["files"].get(e["path"])==e["sha256"],"Bound pseudo-target development label payload")
    c.require(set(map(int,cache["cities"]))==set(c.DEVELOPMENT) and not cache["retrospective_labels_in_cache"],"Development-only fitting cache")
    graph=reg.json_artifact(c.GRAPH)
    for city in cache["cities"].values():
        c.require(city["predictor"]==reg.panels[city["city_id"]],"V2.2 predictor provenance")
        for e in city["arrays"].values():c.require(m["files"].get(e["path"])==e["sha256"],"Exact frozen cache payload")
        g=next(x for x in graph["graphs"] if x["city_id"]==city["city_id"] and x["k"]==4)
        c.require(m["files"].get(g["path"])==g["sha256"] and g["reused_unchanged"],"Reconciled k4 graph")
        c.require(g["v2_2_roster_sha256"]==reg.contract.city_roster_hash(city["city_id"]),"Frozen k4 node roster")
    # Existing seal membership, fit cutoffs and predictor-origin binding stay intact.
    for target in c.FOLDS:
        for phase in ("source","adaptation"):
            j=next(j for j in jobs if j["pseudo_target"]==target and j["phase"]==phase)
            e=j["snapshot"];c.bound(e);info=c.read(e["path"]);seal=info["training_key_seal"]
            q=FitRequest.registered(reg.contract,phase="development",kind=phase,budget="full" if phase=="source" else "7",target_city=target)
            c.require(c.metadata_equal(info["request"],asdict(q)),"Fixed elapsed fit window")
            c.require(seal["status"]=="SEALED_TRAINING_KEYS" and seal["purpose"]=="SCIENTIFIC_ARTIFACT" and seal["row_count"]>0,"Scientific fitting-key gate")
            c.require(c.canonical_hash(seal)==info["training_key_seal_sha256"],"Fitting seal hash")
            ArtifactIdentity.create(reg.contract,"fit_snapshot",q.cutoff,q.city_ids).validate(seal["identity"])
    for target in (195,199,237,617):
        try:reg.predictors(target)
        except PermissionError:pass
        else:raise RuntimeError("Final-target firewall failed")
    try:reg.path("processed/protocol_v2_1/development/development_panel.parquet")
    except PermissionError:pass
    else:raise RuntimeError("V2.1 predictor cross-loading permitted")
    c.require(m["implementation_gate_passed"] and not m["final_experiment_authorized"],"Implementation gate or final firewall")
    return jobs,cache


def toy(device="cpu"):
    return {"x_hist":torch.zeros(2,24,3,2,device=device),"m_hist":torch.zeros(2,24,3,dtype=torch.bool,device=device),
            "x_week":torch.zeros(2,3,2,device=device),"x_static":torch.zeros(3,2,device=device),
            "x_calendar":torch.zeros(2,6,device=device),"adjacency":torch.eye(3,device=device)}


def preflight(manifest_sha,resume=False):
    jobs,cache=verify_package(manifest_sha);rt=runtime()
    has_attempts=any(p.is_file() for p in c.path(c.OUT+"attempts").rglob("*"))
    has_completed=any(c.path(c.OUT+"completed").glob("*.json"))
    has_decision=c.path(c.OUT+"stage4_decision_manifest.json").exists()
    c.require(resume or not (has_attempts or has_completed or has_decision),"Existing Stage-4 outputs require explicit --resume")
    existing=validate_existing(manifest_sha,verified=(jobs,cache)) if resume else {"completed":0,"partial_attempts":0}
    set_seed(17);model=method.SourceInvariantModel().to("cuda");inputs=toy("cuda")
    mask=torch.ones(2,3,dtype=torch.bool,device="cuda");target=torch.ones(2,3,device="cuda")
    with torch.no_grad():
        set_seed(17);one=model.objectives(inputs,target,mask,0,.1)
        set_seed(17);two=model.objectives(inputs,target,mask,0,.1)
    c.require(all(torch.equal(x,y) for x,y in zip(one,two)),"Synthetic deterministic A40 GRL/discriminator forward")
    c.require(sum(p.numel() for p in model.forecast.parameters())==3403 and sum(p.numel() for p in model.parameters())==5970,"Frozen architecture sizes")
    rec={"status":"PASS","protocol_version":"2.2","stage":4,"manifest_sha256":manifest_sha,
         "scientific_contract_sha256":c.sha(c.CONTRACT),"job_map_sha256":c.sha(c.JOBMAP),"runtime":rt,
         "workers":4,"source_fits":72,"adaptation_fits":72,"total_fits":144,"resume":resume,"existing":existing,
         "synthetic_forward_only":True,"scientific_optimizer_updates":0,"scientific_evaluations":0,
         "final_target_labels_accessed":False,"final_experiment_started":False,"timestamp_utc":c.utc()}
    return c.write(c.OUT+"preflight/"+uuid.uuid4().hex+".json",rec)


def check_preflight(e,manifest_sha,rt=None):
    c.require(e["path"].startswith(c.OUT+"preflight/"),"Stage-4 preflight namespace")
    c.bound(e);r=c.read(e["path"])
    c.require(r["status"]=="PASS" and r["manifest_sha256"]==manifest_sha and r["workers"]==4 and
              r["scientific_optimizer_updates"]==r["scientific_evaluations"]==0,"Strict preflight required")
    c.require(r["runtime"]["settings"]==SETTINGS and r["runtime"]["device"]=="NVIDIA A40" and
              r["runtime"]["execution_environment"]==c.ENVIRONMENT,"Persisted deterministic A40 contract")
    c.require(r["scientific_contract_sha256"]==c.sha(c.CONTRACT) and r["job_map_sha256"]==c.sha(c.JOBMAP),"Preflight scientific bindings")
    if rt is not None:c.require(c.metadata_equal(r["runtime"],rt),"Runtime drift")
    return r


class City(FrozenCity):
    def __init__(self,cache_entry,device):
        if cache_entry["city_id"] not in c.DEVELOPMENT:raise PermissionError("Development cities only")
        graph=next(g for g in c.read(c.GRAPH)["graphs"] if g["city_id"]==cache_entry["city_id"] and g["k"]==4)
        c.bound(graph)
        e={**cache_entry,"arrays":{**cache_entry["arrays"],"adjacency":graph}}
        super().__init__(e)
        self.device=device;self.static=self.static.to(device);self.graph=self.graph.to(device)
    def inputs(self,indices):
        return {k:v.to(self.device) for k,v in super().inputs(indices).items()}
    def batch(self,indices,phase):
        b=super().batch(indices,phase)
        return {"model_inputs":b["model_inputs"],"target":b["target"].to(self.device),"mask":b["mask"].to(self.device)}


def metadata(job,manifest_sha,rt,source):
    return {"schema_version":"1.0","protocol_version":"2.2","stage":4,"execution_environment":c.ENVIRONMENT,
            "job":job,"initial_optimizer_updates":0,"updates_completed":job["updates"],
            "optimizer":asdict(c.optimizer_config(job["phase"])),"architecture":c.model_config().to_dict(),
            "manifest_sha256":manifest_sha,"scientific_contract_sha256":c.sha(c.CONTRACT),"job_map_sha256":c.sha(c.JOBMAP),
            "selection_clarification_sha256":c.CLARIFICATION_SHA,"stage3_sha256":c.UPSTREAM[c.STAGE3],
            "data_manifest_sha256":c.DATA_SHA,"runtime":rt,"source_checkpoint":source,
            "source_discriminator_present":job["phase"]=="source","partial_resumed":False,"legacy_checkpoint_loaded":False,
            "final_target_labels_accessed":False}


def save_checkpoint(rel,model,opt,meta,rng,counts):
    payload={"metadata":meta,"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),
             "city_updates":counts.copy(),"anchor_rng":rng.bit_generator.state,"python_rng":random.getstate(),
             "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()}
    buffer=io.BytesIO();torch.save(payload,buffer);return c.write_bytes(rel,buffer.getvalue())


def prediction_metrics(e,job,cache):
    c.require(job["phase"]=="adaptation","No source/zero-shot Stage-4 selection evaluation")
    with np.load(c.bound(e),allow_pickle=False) as z:
        c.require(set(z.files)=={"origin_us","station_ids","prediction_count"},"Prediction-only schema")
        orig,nodes,pred=z["origin_us"],z["station_ids"],z["prediction_count"]
    city=cache["cities"][str(job["pseudo_target"])];q=c.Contract()
    expected_orig=np.load(c.bound(city["arrays"]["origin_us"]),allow_pickle=False)
    expected_nodes=np.load(c.bound(city["arrays"]["station_ids"]),allow_pickle=False)
    ix=np.flatnonzero((expected_orig>=q.boundaries["HD"])&(expected_orig<q.boundaries["HF"]))
    c.require(len(orig)==1440 and np.array_equal(orig,expected_orig[ix]) and np.array_equal(nodes,expected_nodes),"Frozen development prediction grid")
    c.require(pred.shape==(1440,len(nodes)) and np.isfinite(pred).all() and (pred>=0).all(),"Finite nonnegative predictions")
    labels=DevelopmentRegistry(c.ROOT,c.DATA,c.DATA_SHA).retrospective_labels(job["pseudo_target"])
    c.require(np.array_equal(labels["origin_us"],expected_orig) and np.array_equal(labels["station_ids"],nodes),"Label key alignment")
    mask=labels["observed"][ix];targets=labels["count"][ix]
    metrics=compute_metrics(targets[mask],pred[mask],np.broadcast_to(nodes[None,:],mask.shape)[mask]).as_dict()
    c.require(metrics["mae"] is not None and metrics["n"]==int(mask.sum()),"Nonempty development evaluation")
    return {"candidate_id":job["candidate_id"],"pseudo_target":job["pseudo_target"],"seed":job["seed"],
            "regime":"post_7d_adaptation",**metrics,"prediction":e,"final_target_labels_accessed":False}


@torch.no_grad()
def predict(model,city,job,attempt,cache):
    model.eval();q=c.Contract();ix=np.flatnonzero((city.origin_us>=q.boundaries["HD"])&(city.origin_us<q.boundaries["HF"]))
    pred=np.concatenate([model(**city.inputs(ix[i:i+64])).count_prediction.cpu().numpy() for i in range(0,len(ix),64)])
    b=io.BytesIO();np.savez_compressed(b,origin_us=np.array(city.origin_us[ix]),station_ids=np.array(city.station_ids),prediction_count=pred)
    e=c.write_bytes(attempt+"prediction.npz",b.getvalue())
    # Full forecast grid is committed before opening the retrospective label reader.
    commitment=c.write(attempt+"prediction_commitment.json",{"prediction":e,"job_id":job["job_id"],
           "origin_key_hash":c.canonical_hash(list(map(int,city.origin_us[ix]))),"station_key_hash":c.canonical_hash(list(map(int,city.station_ids))),
           "labels_opened_before_prediction_commit":False,"timestamp_utc":c.utc()})
    return prediction_metrics(e,job,cache),commitment


def completed_path(job_id):return c.OUT+"completed/"+job_id+".json"
def commit_path(job_id):return c.OUT+"commits/"+job_id+".json"
def commit_completion(job,record):
    e=c.write(completed_path(job["job_id"]),record)
    c.write(commit_path(job["job_id"]),{"status":"IMMUTABLE_COMPLETION_COMMITTED","job_id":job["job_id"],"completion":e})
    return e


def validate_completed(job,manifest_sha,cache,recompute_metrics=False):
    commit=c.read(commit_path(job["job_id"]))
    c.require(commit["status"]=="IMMUTABLE_COMPLETION_COMMITTED" and commit["job_id"]==job["job_id"] and
              commit["completion"]["path"]==completed_path(job["job_id"]),"Completion hash commitment")
    c.bound(commit["completion"]);r=c.read(completed_path(job["job_id"]))
    c.require(r["status"]=="COMPLETED_VALIDATED_FIT" and r["completed_fit"] and
              c.metadata_equal(r["job"],job) and r["manifest_sha256"]==manifest_sha,"Incomplete/corrupt completion identity")
    check_preflight(r["preflight"],manifest_sha,r["runtime"])
    source=None
    if job["phase"]=="adaptation":
        sc=c.read(commit_path(job["depends_on"]));c.bound(sc["completion"])
        sr=c.read(sc["completion"]["path"])
        c.require(sr["status"]=="COMPLETED_VALIDATED_FIT" and sr["job"]["job_id"]==job["depends_on"] and
                  sr["job"]["phase"]=="source" and sr["manifest_sha256"]==manifest_sha,"Authoritative source dependency")
        source=sr["checkpoint"];c.bound(source)
    prefix=c.OUT+"attempts/"+job["job_id"]+"/"
    c.require(r["checkpoint"]["path"].startswith(prefix),"Checkpoint namespace; no cross-loading")
    cp=torch.load(c.bound(r["checkpoint"]),map_location="cpu",weights_only=True)
    c.require(c.metadata_equal(cp["metadata"],metadata(job,manifest_sha,r["runtime"],source)),"Genuine checkpoint metadata mismatch")
    schema=(method.SourceInvariantModel() if job["phase"]=="source" else GraphGRU(c.model_config())).state_dict()
    c.require(cp["model_state"].keys()==schema.keys() and all(cp["model_state"][k].shape==v.shape and
              cp["model_state"][k].dtype==v.dtype for k,v in schema.items()),"Exact parameter names, shapes and dtypes")
    expected_params=5970 if job["phase"]=="source" else 3403
    c.require(sum(t.numel() for t in cp["model_state"].values())==expected_params,"Complete registered parameter set")
    c.require(all(torch.isfinite(t).all() for t in cp["model_state"].values()),"Finite model state")
    opt=cp["optimizer_state"];states=opt["state"]
    c.require(len(states)==len(cp["model_state"]) and all(int(v["step"].item())==job["updates"] for v in states.values()),"All parameters reached exact update budget")
    for v in states.values():
        c.require(all(torch.isfinite(t).all() for t in v.values() if isinstance(t,torch.Tensor)),"Finite optimizer state")
    group=opt["param_groups"]
    cfg=c.optimizer_config(job["phase"])
    c.require(len(group)==1 and group[0]["lr"]==cfg.learning_rate and c.metadata_equal(group[0]["betas"],cfg.betas) and
              group[0]["eps"]==cfg.epsilon and group[0]["weight_decay"]==cfg.weight_decay,"Frozen AdamW settings")
    c.require(len(group[0]["params"])==len(set(group[0]["params"]))==len(schema) and
              set(group[0]["params"])==set(states),"Optimizer parameter ownership after reload")
    for pid,t in zip(group[0]["params"],schema.values()):
        c.require(states[pid]["exp_avg"].shape==t.shape and states[pid]["exp_avg_sq"].shape==t.shape,"AdamW moment shapes")
    seq=EqualCitySampler(tuple(job["source_cities"]),job["source_schedule_seed"]).sequence(12000) if job["phase"]=="source" else [job["pseudo_target"]]*300
    counts={i:seq.count(i) for i in set(seq)}
    c.require(cp["city_updates"]==counts,"Exact source/adaptation city exposures")
    c.require(r["optimizer_initial_state_empty"] and r["checkpoint_roundtrip_max_abs"]==0.,"Fresh optimizer and checkpoint roundtrip")
    c.require(r["logs"][0]["step"]==1 and r["logs"][-1]["step"]==job["updates"],"Log budget endpoints")
    if job["phase"]=="source":
        c.require(not r["evaluation_completed"] and r["metric"] is None and r["prediction_commitment"] is None,"Source fit has no selection evaluation")
        for log in r["logs"]:
            c.require(log["lambda_t"]==method.lambda_at(job["lambda_max"],job["schedule"],log["step"]-1),"Logged lambda schedule")
            c.require(all(k in log for k in ("forecast_loss","domain_ce","domain_accuracy","city_updates")),"Required diagnostics")
    else:
        c.require(r["evaluation_completed"] and all(not k.startswith("discriminator") for k in cp["model_state"]),"Adaptation discriminator absent")
        c.require(all(r["metric"][k]==job[k] for k in ("candidate_id","pseudo_target","seed")) and
                  r["metric"]["regime"]=="post_7d_adaptation","Metric cell identity")
        c.bound(r["prediction_commitment"]);pc=c.read(r["prediction_commitment"]["path"])
        c.require(pc["job_id"]==job["job_id"] and pc["prediction"]==r["metric"]["prediction"] and not pc["labels_opened_before_prediction_commit"],"Prediction-before-label commitment")
        c.require(r["metric"]["prediction"]["path"].startswith(prefix),"Prediction namespace")
        c.bound(r["metric"]["prediction"])
        if recompute_metrics:c.require(c.metadata_equal(prediction_metrics(r["metric"]["prediction"],job,cache),r["metric"]),"Recomputed metrics mismatch")
    return r


def validate_existing(manifest_sha,verified=None):
    jobs,cache=verified or verify_package(manifest_sha)
    known={j["job_id"] for j in jobs};present={p.stem for p in c.path(c.OUT+"completed").glob("*.json")}
    commits={p.stem for p in c.path(c.OUT+"commits").glob("*.json")}
    c.require(present<=known and commits==present,"Unknown or half-committed completion: fail closed")
    for job in jobs:
        if job["job_id"] in present:validate_completed(job,manifest_sha,cache)
    finished={c.path(c.read(completed_path(jid))["checkpoint"]["path"]).parent for jid in present}
    attempts=list(c.path(c.OUT+"attempts").glob("*/*/started.json"))
    partial=[p for p in attempts if p.parent not in finished]
    return {"status":"PASS","completed":len(present),"remaining":144-len(present),"partial_attempts":len(partial),
            "partial_policy":"never load a partial checkpoint; fresh UUID attempt from optimizer update 0",
            "corrupt_completion_policy":"STOP; preserve corrupt evidence, no automatic replacement","files_written":[]}


def run_job(job_id,manifest_sha,pf_entry):
    jobs,cache=verify_package(manifest_sha);rt=runtime();check_preflight(pf_entry,manifest_sha,rt)
    job=next(j for j in jobs if j["job_id"]==job_id)
    if c.path(completed_path(job_id)).exists() or c.path(commit_path(job_id)).exists():
        validate_completed(job,manifest_sha,cache);print("SKIP_VALIDATED_COMPLETION "+job_id,flush=True);return
    source=None
    if job["phase"]=="adaptation":
        sj=next(j for j in jobs if j["job_id"]==job["depends_on"])
        source=validate_completed(sj,manifest_sha,cache)["checkpoint"]
    attempt=c.OUT+"attempts/"+job_id+"/"+uuid.uuid4().hex+"/"
    c.write(attempt+"started.json",{"job":job,"manifest_sha256":manifest_sha,"runtime":rt,"initial_optimizer_updates":0,
             "partial_checkpoint_resumed":False,"source_checkpoint":source,"timestamp_utc":c.utc()})
    phase=job["phase"];ids=job["source_cities"] if phase=="source" else [job["pseudo_target"]]
    cities={i:City(cache["cities"][str(i)],torch.device("cuda")) for i in ids}
    c.require((job["pseudo_target"] not in cities) if phase=="source" else list(cities)==[job["pseudo_target"]],"Pseudo-target excluded throughout source pretraining")
    set_seed(job["seed"])
    if phase=="source":
        model=method.SourceInvariantModel().to("cuda");opt=method.source_optimizer(model)
        seq=EqualCitySampler(tuple(ids),job["source_schedule_seed"]).sequence(12000)
        rng=np.random.default_rng(job["source_anchor_seed"])
    else:
        model=GraphGRU(c.model_config()).to("cuda")
        original=torch.load(c.bound(source),map_location="cpu",weights_only=True)
        forecast_state={k[len("forecast."):]:v for k,v in original["model_state"].items() if k.startswith("forecast.")}
        model.load_state_dict(forecast_state,strict=True);del original
        set_seed(job["adaptation_seed"]);trainer=method.adaptation_trainer(model,job["seed"]);opt=trainer.optimizer
        seq=[job["pseudo_target"]]*300;rng=np.random.default_rng(job["adaptation_seed"])
    c.require(not opt.state,"Optimizer starts empty at update zero")
    logs=[];counts={i:0 for i in ids};start=time.monotonic();meta=metadata(job,manifest_sha,rt,source)
    for step,city_id in enumerate(seq,1):
        city=cities[city_id];anchors=city.source_anchors if phase=="source" else city.adaptation_anchors
        ix=rng.choice(anchors,size=16,replace=True).astype(np.int64);batch=city.batch(ix,phase)
        if phase=="source":
            log=method.source_step(model,opt,batch,job["domain_classes"][str(city_id)],method.lambda_at(job["lambda_max"],job["schedule"],step-1))
        else:log=asdict(trainer.train_step(batch))
        counts[city_id]+=1
        if step==1 or step%100==0 or step==job["updates"]:
            logs.append({**log,"step":step,"city_updates":{str(i):n for i,n in counts.items()}})
        if step%2000==0:
            save_checkpoint(attempt+f"partial_{step:05d}.pt",model,opt,{**meta,"updates_completed":step,"completed_fit":False},rng,counts)
            print(f"{job_id} {step}/{job['updates']} elapsed={time.monotonic()-start:.1f}s",flush=True)
        if phase=="adaptation" and len(trainer.logs)>=1000:trainer.logs.clear()
    cp=save_checkpoint(attempt+"final.pt",model,opt,meta,rng,counts)
    restored=method.SourceInvariantModel().to("cuda") if phase=="source" else GraphGRU(c.model_config()).to("cuda")
    payload=torch.load(c.bound(cp),map_location="cpu",weights_only=True)
    c.require(c.metadata_equal(payload["metadata"],meta),"Semantic checkpoint roundtrip metadata")
    restored.load_state_dict(payload["model_state"],strict=True);restored.eval();model.eval()
    # Source probes use a source city's registered fitting inputs, never the held-out pseudo-target.
    probe_city=cities[ids[0]];anchor=probe_city.source_anchors[0] if phase=="source" else probe_city.adaptation_anchors[0]
    inputs=probe_city.inputs(np.array([anchor],dtype=np.int64))
    with torch.no_grad():
        left=(model.forecast if phase=="source" else model)(**inputs).count_prediction
        right=(restored.forecast if phase=="source" else restored)(**inputs).count_prediction
    c.require(torch.equal(left,right),"Exact forecast checkpoint reload")
    if phase=="source":
        # Dropout disabled: validate discriminator reload as well, with source mask.
        with torch.no_grad():
            h=model.forecast(**inputs).hidden;mask=probe_city.batch(np.array([anchor]),"source")["mask"]
            z=method.pool_current_mask(h,mask)
            c.require(torch.equal(model.discriminator(z),restored.discriminator(z)),"Exact discriminator checkpoint reload")
    metric,commitment=(None,None) if phase=="source" else predict(restored,cities[job["pseudo_target"]],job,attempt,cache)
    record={"status":"COMPLETED_VALIDATED_FIT","completed_fit":True,"evaluation_completed":phase=="adaptation",
            "job":job,"manifest_sha256":manifest_sha,"runtime":rt,"preflight":pf_entry,"checkpoint":cp,
            "metric":metric,"prediction_commitment":commitment,"optimizer_initial_state_empty":True,
            "checkpoint_roundtrip_max_abs":0.,"logs":logs,"seconds":time.monotonic()-start,"completed_utc":c.utc()}
    commit_completion(job,record);validate_completed(job,manifest_sha,cache)
    print("COMPLETED_VALIDATED_FIT "+job_id,flush=True)


def validate_all(manifest_sha):
    jobs,cache=verify_package(manifest_sha);existing=validate_existing(manifest_sha,verified=(jobs,cache))
    c.require(existing["completed"]==144,"144/144 completed fits required")
    exit_records={}
    for p in c.path(c.OUT+"controller_runs").glob("*/workers/*.exit.json"):
        rel=p.relative_to(c.ROOT).as_posix();r=c.read(rel)
        if r.get("status")=="EXITED_ZERO_WITH_COMPLETION" and r.get("manifest_sha256")==manifest_sha:
            c.require(r["exit_code"]==0,"Worker exit code")
            c.require(r["completion"]["path"]==completed_path(r["job_id"]),"Worker exit binds its own immutable job")
            c.bound(r["completion"]);exit_records[r["job_id"]]=c.entry(rel)
    rows=[];records=[];rt=None
    for job in jobs:
        c.require(job["job_id"] in exit_records,"Missing successful worker-exit evidence: "+job["job_id"])
        r=validate_completed(job,manifest_sha,cache,recompute_metrics=job["phase"]=="adaptation")
        if rt is None:rt=r["runtime"]
        c.require(c.metadata_equal(rt,r["runtime"]),"One authoritative runtime across all candidates")
        records.append(c.entry(completed_path(job["job_id"])))
        if job["phase"]=="adaptation":rows.append(r["metric"])
    report={"status":"PASS","purpose":"READ_ONLY_POSTRUN_VALIDATION","manifest_sha256":manifest_sha,
            "scientific_contract_sha256":c.sha(c.CONTRACT),"job_map_sha256":c.sha(c.JOBMAP),
            "source_fits":72,"adaptation_fits":72,"completed_fits":144,"post_adaptation_evaluations":72,
            "zero_shot_selection_evaluations":0,"execution_records":records,"successful_exit_records":exit_records,
            "model_forward_calls":0,"optimizer_updates":0,"files_written":[],"final_target_labels_accessed":False}
    return report,rows


def freeze(manifest_sha):
    report,rows=validate_all(manifest_sha);result=selection.select(rows)
    decision={"status":"FROZEN_VALIDATED","protocol_version":"2.2","stage":4,"execution_environment":c.ENVIRONMENT,
              "manifest_sha256":manifest_sha,"scientific_contract":c.entry(c.CONTRACT),"job_map":c.entry(c.JOBMAP),
              "selection_clarification":c.entry(c.CLARIFICATION),"stage3":c.entry(c.STAGE3),"upstream_sha256":c.UPSTREAM,
              "selection":result,"selected_candidate":result["selected_candidate"],"postrun_validation":report,
              "downstream_blockers":c.BLOCKERS,"final_target_labels_accessed":False,"final_experiment_started":False}
    rel=c.OUT+"stage4_decision_manifest.json"
    if c.path(rel).exists():c.require(c.metadata_equal(c.read(rel),decision),"Existing frozen decision differs")
    else:c.write(rel,decision)
    return {"status":"STAGE4_FROZEN_STOP","decision":c.entry(rel),"selected_candidate":result["selected_candidate"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("package-check","preflight","validate-existing","job","validate","freeze"))
    p.add_argument("--manifest-sha256",required=True);p.add_argument("--resume",action="store_true")
    p.add_argument("--job-id");p.add_argument("--preflight");p.add_argument("--preflight-sha256")
    a=p.parse_args()
    if a.mode=="package-check":verify_package(a.manifest_sha256);r={"status":"PASS_PACKAGE_ONLY_NO_A40_CLAIM"}
    elif a.mode=="preflight":r=preflight(a.manifest_sha256,a.resume)
    elif a.mode=="validate-existing":r=validate_existing(a.manifest_sha256)
    elif a.mode=="job":
        c.require(a.job_id and a.preflight and a.preflight_sha256,"Job and bound preflight required")
        c.require(a.job_id in {j["job_id"] for j in c.read(c.JOBMAP)["jobs"]},"Registered immutable job ID required")
        with execution_lock(a.job_id):run_job(a.job_id,a.manifest_sha256,{"path":a.preflight,"sha256":a.preflight_sha256})
        return
    elif a.mode=="validate":
        r,rows=validate_all(a.manifest_sha256)
        if c.path(c.OUT+"stage4_decision_manifest.json").exists():
            d=c.read(c.OUT+"stage4_decision_manifest.json")
            c.require(c.metadata_equal(d["selection"],selection.select(rows)) and d["selected_candidate"]==d["selection"]["selected_candidate"],"Frozen winner/ranking verification")
    else:r=freeze(a.manifest_sha256)
    print(json.dumps(r,indent=2),flush=True)


if __name__=="__main__":main()
