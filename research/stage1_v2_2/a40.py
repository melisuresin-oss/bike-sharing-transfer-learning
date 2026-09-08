"""Append-only UNIVERSITY_A40 execution adapter for the unchanged Stage-1 job map.

Only the runtime and execution granularity change. Scientific definitions and
the accepted causal fitting cache are imported by their existing hashes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager

if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
    raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np
import torch
from research.stage1_v2_2 import core as c

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "deployment/stage1_v2_2_a40/"
OUT = "research/results/stage1_scale_loss_v2_2_a40/"
MANIFEST = PACKAGE + "BUNDLE_MANIFEST.json"
AUTH = PACKAGE + "EXECUTION_AUTHORIZATION.json"
JOBMAP = c.OUT + "job_map.json"
JOBMAP_SHA = "4709c222b799a9a2d4ef209ce80ae5ee859b5a818db5c1cb97df2954ffce3937"
DESIGN_SHA = "25f1959ccaacf49f3aa50641427739d9c291843270b7d8961c5843cb5465024b"
ENVIRONMENT = "UNIVERSITY_A40"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def path(rel, output=False):
    p = c.path(rel)
    if output:
        require(rel.startswith(OUT), "A40 output isolation")
    return p


def write(rel, value):
    p = path(rel, True)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A finished JSON becomes visible only after its complete bytes exist.
    temp = p.with_name(p.name + ".pending-" + uuid.uuid4().hex)
    with temp.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write("\n")
        f.flush(); os.fsync(f.fileno())
    require(not p.exists(), "Append-only destination exists: " + rel)
    # Windows rename refuses an existing destination (no replacing old records).
    temp.rename(p)
    return {"path": rel, "sha256": c.sha(rel)}


def entry(rel):
    return {"path": rel, "sha256": c.sha(rel)}


def bound(e):
    require(c.sha(e["path"]) == e["sha256"], "Hash mismatch: " + e["path"])
    return path(e["path"])


@contextmanager
def execution_lock(name):
    """Windows releases this OS lock even if a worker or coordinator crashes."""
    import msvcrt
    p=path(OUT+"locks/"+name+".lock",True);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a+b") as f:
        if p.stat().st_size==0:f.write(b"0");f.flush()
        f.seek(0)
        try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RuntimeError("An execution is already active for "+name)
        try:yield
        finally:
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)


def verify_package(expected_sha):
    require(c.sha(MANIFEST) == expected_sha, "External bundle manifest binding")
    m = c.read(MANIFEST)
    require(m["execution_environment"] == ENVIRONMENT, "Wrong bundle authority")
    for rel, h in m["files"].items():
        require(not any(x in rel.lower() for x in ("final_label", "final_adaptation", "sealed_final", "__pycache__")), "Forbidden bundle payload")
        require(not rel.endswith(".pt"), "Bundle must contain no local checkpoint")
        require(c.sha(rel) == h, "Bundle bytes changed: " + rel)
    require(c.sha(JOBMAP) == JOBMAP_SHA, "Immutable job-map hash")
    jm = c.read(JOBMAP)
    require(c.sha(jm["scientific_manifest"]["path"]) == DESIGN_SHA, "Unchanged scientific definitions")
    bound(jm["selection_rule"])
    design = c.read(jm["scientific_manifest"]["path"])
    require(design["selection"] == c.SELECTION_RULE == c.read(jm["selection_rule"]["path"]), "Frozen selection rule")
    for rel, h in {**design["code_sha256"], **design["design_source_sha256"]}.items():
        require(c.sha(rel) == h, "Frozen design dependency: " + rel)
    require(c.sha(c.DATA) == c.DATA_SHA and c.sha(c.IMPL) == c.IMPL_SHA, "V2.2 authority")
    implementation=c.read(c.IMPL)
    for rel,h in implementation["implementation_files_sha256"].items():
        if rel in m["files"]:require(m["files"][rel]==h,"Accepted V2.2 implementation bytes")
    devseal_path="research/results/causal_history_audit_v2_2/development_data_verification_seal.json"
    require(c.sha(devseal_path)=="dd77b530c21107825d8773d8d700ad03d952883d98f3b3ca69bfedf2225c0bfe","Accepted development gate")
    devseal=c.read(devseal_path);bound(devseal["verification_report"])
    report=c.read(devseal["verification_report"]["path"])
    require(report["status"]=="PASS" and report["check_count"]==960 and report["manifest_sha256"]==c.DATA_SHA and
            c.sha("research/development_data_v2_2/registry.py")==devseal["registry_sha256"],"Verified V2.2 data reader")
    require(c.sha(c.CACHE + "cache_manifest.json") == design["cache_manifest_sha256"], "Frozen fitting cache")
    require(len(jm["tasks"]) == 24 and len(jm["jobs"]) == 48, "Expected 24 source + 24 adaptation fits")
    require(len({j["id"] for j in jm["jobs"]}) == 48, "Duplicate immutable job")
    require({(t["candidate"], t["target"], t["seed"]) for t in jm["tasks"]} ==
            {(a,b,s) for a in c.CANDIDATES for b in c.TARGETS for s in c.SEEDS}, "Registered factorial grid")
    for t in jm["tasks"]:
        require(t["source_cities"] == [i for i in c.DEVELOPMENT if i != t["target"]], "Pseudo-target exclusion")
        require(t["seeds"] == {"initialization":t["seed"], **{k:c.derived_seed(k,t["target"],t["seed"])
                for k in ("source-city-schedule", "source-anchor-sampling", "fine-tune")}}, "Registered seeds")
        bound(t["source_snapshot"]); bound(t["adaptation_snapshot"])
        for phase,snapshot in (("source",t["source_snapshot"]),("adaptation",t["adaptation_snapshot"])):
            info=c.read(snapshot["path"]);seal=info["training_key_seal"]
            require(c.canonical_hash(seal)==info["training_key_seal_sha256"] and
                    seal["status"]=="SEALED_TRAINING_KEYS" and seal["purpose"]=="SCIENTIFIC_ARTIFACT" and
                    seal["row_count"]>0,"Sealed scientific fitting keys")
            from research.v2_2.snapshots import FitRequest
            expected_request=FitRequest.registered(c.Contract(),phase="development",kind=phase,
                                 budget="full" if phase=="source" else "7",target_city=t["target"])
            require(info["request"]=={**asdict(expected_request),"city_ids":list(expected_request.city_ids)},"Frozen fitting window/roster")
            c.ArtifactIdentity.create(c.Contract(),"fit_snapshot",expected_request.cutoff,expected_request.city_ids).validate(seal["identity"])
        expected = [{"id":t["source_id"], "phase":"source", "updates":12000, "task_id":t["task_id"], "depends_on":None},
                    {"id":t["adaptation_id"], "phase":"adaptation", "updates":300, "task_id":t["task_id"], "depends_on":t["source_id"]}]
        require([j for j in jm["jobs"] if j["task_id"] == t["task_id"]] == expected, "Registered fit dependency/budget")
    require(sum(j["phase"] == "source" for j in jm["jobs"]) == 24, "24 source fits")
    require(sum(j["phase"] == "adaptation" for j in jm["jobs"]) == 24, "24 adaptation fits")
    cache = c.read(c.CACHE + "cache_manifest.json")
    require(set(map(int,cache["cities"])) == set(c.DEVELOPMENT) and not cache["retrospective_labels_in_cache"], "Development-only cache")
    for city in cache["cities"].values():
        for a in city["arrays"].values():
            require(m["files"].get(a["path"])==a["sha256"],"Cache member must be verified bundle payload")
        require(city["target_values_from_fit_snapshot_only"], "Fit labels must be causally sealed")
        require(city["predictor"]["path"].startswith("processed/protocol_v2_2/development/"),"V2.2 predictor origin")
    reg = c.DevelopmentRegistry(ROOT, c.DATA, c.DATA_SHA)
    label_authority=reg.json_artifact("processed/protocol_v2_2/development_labels/retrospective_label_manifest.json")
    for e in label_authority["labels"]:
        if e["city_id"] in c.TARGETS:
            require(m["files"].get(e["path"])==e["sha256"] and e["role"]=="DEVELOPMENT_RETROSPECTIVE_LABELS","Separate development labels")
    graph_authority=reg.json_artifact("research/results/causal_history_audit_v2_2/graph_static_reconciliation.json")
    for city_id,city in cache["cities"].items():
        graph=next(g for g in graph_authority["graphs"] if g["city_id"]==int(city_id) and g["k"]==8)
        require(city["arrays"]["adjacency"]["sha256"]==graph["sha256"] and graph["reused_unchanged"],"Reconciled k8 static adjacency")
        require(city["predictor"]==reg.panels[int(city_id)],"Cache references accepted V2.2 predictor")
    for forbidden in ("processed/protocol_v2_1/development/development_panel.parquet",
                      "processed/protocol_v2_2/final_labels/payload.npz"):
        try: reg.path(forbidden)
        except PermissionError: pass
        else: raise RuntimeError("Data firewall failed")
    for city in (195,199,237,617):
        try: reg.predictors(city)
        except PermissionError: pass
        else: raise RuntimeError("Final-target reader allowed")
    auth = c.read(AUTH)
    require(auth["stage1_execution_authorized_after_preflight_pass"] and
            auth["job_map_sha256"] == JOBMAP_SHA and auth["execution_environment"] == ENVIRONMENT and
            not auth["stage2_authorized"] and not auth["stage2b_authorized"], "Execution scope")
    return jm, cache


def runtime():
    c.setup()
    expected_python = Path(os.environ["USERPROFILE"]) / "bike_env/Scripts/python.exe"
    require(Path(sys.executable).resolve() == expected_python.resolve(), "Use USERPROFILE\\bike_env\\Scripts\\python.exe")
    require(sys.version_info[:2] == (3,10), "Validated TS9 Python 3.10 required")
    require(str(torch.__version__) == "2.0.1+cu118" and np.__version__ == "1.26.4", "Validated TS9 torch/numpy versions required")
    for mod in (torch, np):
        require(expected_python.parent.parent.resolve() in Path(mod.__file__).resolve().parents, "Dependencies must load from bike_env")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device required")
    require(torch.cuda.get_device_name(0) == "NVIDIA A40", "CUDA device must be NVIDIA A40")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True).stdout.strip()
    require(len(smi.splitlines()) == 1 and smi.split(",")[0].strip() == "NVIDIA A40", "nvidia-smi A40 identity")
    settings = {"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"],
                "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
                "cuda_matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
                "cudnn_benchmark":torch.backends.cudnn.benchmark,
                "cudnn_deterministic":torch.backends.cudnn.deterministic,
                "cudnn_tf32":torch.backends.cudnn.allow_tf32}
    require(settings == {"CUBLAS_WORKSPACE_CONFIG":":4096:8", "deterministic_algorithms":True,
                         "cuda_matmul_tf32":False, "cudnn_benchmark":False,
                         "cudnn_deterministic":True, "cudnn_tf32":False}, "Deterministic CUDA contract")
    return {"execution_environment":ENVIRONMENT, "python_executable":str(expected_python.resolve()),
            "python":sys.version, "torch":str(torch.__version__), "numpy":np.__version__,
            "cuda":torch.version.cuda, "cudnn":torch.backends.cudnn.version(), "device":"NVIDIA A40",
            "nvidia_smi":smi, "settings":settings}


def preflight(manifest_sha):
    jm, cache = verify_package(manifest_sha)
    rt = runtime()
    # Synthetic forwards only; no optimizer, backward pass, or scientific labels.
    c.set_seed(17)
    model = c.GraphGRU(c.configuration("RAW")).to("cuda")
    require(sum(p.numel() for p in model.parameters()) == 12939, "Frozen model parameter count")
    toy = {"x_hist":torch.zeros(2,24,3,2,device="cuda"), "m_hist":torch.zeros(2,24,3,dtype=torch.bool,device="cuda"),
           "x_week":torch.zeros(2,3,2,device="cuda"), "x_static":torch.zeros(3,2,device="cuda"),
           "x_calendar":torch.zeros(2,6,device="cuda"), "adjacency":torch.eye(3,device="cuda")}
    model.train()
    with torch.no_grad():
        c.set_seed(17); one = model(**toy).prediction
        c.set_seed(17); two = model(**toy).prediction
    require(torch.equal(one,two), "Synthetic deterministic CUDA/dropout forward")
    c.set_seed(29)
    other = c.GraphGRU(c.configuration("RAW")).to("cuda")
    other.load_state_dict(model.state_dict()); model.eval(); other.eval()
    with torch.no_grad():
        require(torch.equal(model(**toy).prediction, other(**toy).prediction), "CUDA state roundtrip")
    result = {"status":"PASS", "bundle_manifest_sha256":manifest_sha, "job_map_sha256":JOBMAP_SHA,
              "runtime":rt, "scientific_training_steps":0, "scientific_evaluation_performed":False,
              "source_fits":24, "adaptation_fits":24, "synthetic_cuda_forward_passed":True,
              "final_target_labels_accessed":False, "v2_1_predictor_cross_loading_rejected":True,
              "created_utc":c.utc()}
    rel = OUT + "preflight/" + uuid.uuid4().hex + ".json"
    saved = write(rel,result)
    print(json.dumps({"status":"PASS", "preflight":saved}),flush=True)
    return saved


def check_preflight(pf, manifest_sha, actual_runtime=None):
    require(pf.startswith(OUT + "preflight/"), "A40 preflight namespace")
    p = c.read(pf)
    require(p["status"] == "PASS" and p["scientific_training_steps"] == 0 and
            p["bundle_manifest_sha256"] == manifest_sha and p["job_map_sha256"] == JOBMAP_SHA, "A40 preflight PASS required")
    rt=p["runtime"]
    require(rt["execution_environment"]==ENVIRONMENT and rt["device"]=="NVIDIA A40" and
            rt["torch"]=="2.0.1+cu118" and rt["numpy"]=="1.26.4" and
            rt["settings"]=={"CUBLAS_WORKSPACE_CONFIG":":4096:8","deterministic_algorithms":True,
                "cuda_matmul_tf32":False,"cudnn_benchmark":False,"cudnn_deterministic":True,"cudnn_tf32":False},
            "Persisted A40 runtime contract")
    if actual_runtime is not None:
        require(p["runtime"] == actual_runtime, "Runtime drift after preflight")
    return p


class CudaCity(c.City):
    def __init__(self, e):
        super().__init__(e)
        self.static = self.static.to("cuda"); self.graph = self.graph.to("cuda")
        self.gpu = {k:torch.tensor(np.array(getattr(self,k)),device="cuda") for k in
                    ("x_hist","m_hist","x_week","x_calendar","fit_count","fit_mask","adaptation_7d_mask")}

    def inputs(self, indices):
        ix = torch.tensor(indices,device="cuda",dtype=torch.long)
        return {**{k:self.gpu[k].index_select(0,ix) for k in ("x_hist","m_hist","x_week","x_calendar")},
                "x_static":self.static,"adjacency":self.graph}

    def batch(self, indices, phase):
        ix = torch.tensor(indices,device="cuda",dtype=torch.long)
        return {"model_inputs":self.inputs(indices), "target":self.gpu["fit_count"].index_select(0,ix),
                "mask":self.gpu["fit_mask" if phase == "source" else "adaptation_7d_mask"].index_select(0,ix)}


def expected_metadata(job,task,manifest_sha,rt,source):
    return {"protocol_version":"2.2", "execution_environment":ENVIRONMENT, "immutable_job":job,
            "task":task, "updates_completed":job["updates"], "initial_optimizer_updates":0,
            "architecture":c.configuration(task["candidate"]).to_dict(),
            "job_map_sha256":JOBMAP_SHA, "scientific_manifest_sha256":DESIGN_SHA,
            "bundle_manifest_sha256":manifest_sha, "data_manifest_sha256":c.DATA_SHA,
            "implementation_manifest_sha256":c.IMPL_SHA, "runtime":rt,
            "source_checkpoint":source, "local_cpu_checkpoint_loaded":False,
            "partial_checkpoint_resumed":False, "final_labels_accessed":False}


def checkpoint(rel,model,trainer,meta,rng,counts):
    p = path(rel,True); p.parent.mkdir(parents=True,exist_ok=True)
    require(not p.exists(),"Append-only checkpoint")
    payload = {"metadata":meta, "model_state":model.state_dict(), "optimizer_state":trainer.optimizer.state_dict(),
               "torch_rng":torch.get_rng_state(), "cuda_rng":torch.cuda.get_rng_state_all(),
               "python_rng":__import__("random").getstate(), "anchor_rng":rng.bit_generator.state,
               "city_updates":counts.copy()}
    torch.save(payload,p)
    return entry(rel)


def metrics_for_prediction(pred_entry,task,regime,cache):
    with np.load(bound(pred_entry),allow_pickle=False) as z:
        require(set(z.files) == {"origin_us","station_ids","prediction_count"}, "Prediction schema")
        origins,stations,pred = z["origin_us"],z["station_ids"],z["prediction_count"]
    e = cache["cities"][str(task["target"])]
    orig = np.load(bound(e["arrays"]["origin_us"]),allow_pickle=False)
    nodes = np.load(bound(e["arrays"]["station_ids"]),allow_pickle=False)
    hd = c.Contract().boundaries["HD"]; hf = c.Contract().boundaries["HF"]
    indices = np.flatnonzero((orig>=hd)&(orig<hf))
    require(len(indices)==1440 and np.array_equal(orig[indices],origins) and np.array_equal(nodes,stations), "Immutable evaluation grid")
    require(pred.shape==(1440,len(nodes)) and np.isfinite(pred).all() and (pred>=0).all(), "Finite nonnegative count predictions")
    labels = c.DevelopmentRegistry(ROOT,c.DATA,c.DATA_SHA).retrospective_labels(task["target"])
    require(np.array_equal(labels["origin_us"],orig) and np.array_equal(labels["station_ids"],nodes), "Label key alignment")
    mask = labels["observed"][indices]; target = labels["count"][indices]
    st = np.broadcast_to(nodes[None,:],mask.shape)
    metrics = c.compute_metrics(target[mask],pred[mask],st[mask]).as_dict()
    return {"candidate":task["candidate"], "target":task["target"], "seed":task["seed"], "regime":regime,
            **metrics, "prediction":pred_entry, "evaluation_period":"[HD,HF)", "final_labels_accessed":False}


@torch.no_grad()
def evaluate(model,city,task,regime,attempt,cache):
    model.eval(); hd=c.Contract().boundaries["HD"]; hf=c.Contract().boundaries["HF"]
    indices=np.flatnonzero((city.origin_us>=hd)&(city.origin_us<hf))
    pred=np.concatenate([model(**city.inputs(indices[i:i+64])).count_prediction.cpu().numpy()
                         for i in range(0,len(indices),64)])
    rel=attempt+"prediction.npz"
    require(not path(rel,True).exists(),"Append-only prediction")
    np.savez_compressed(path(rel,True),origin_us=np.array(city.origin_us[indices]),
                        station_ids=np.array(city.station_ids),prediction_count=pred)
    # Persist the full prediction grid before the retrospective label reader runs.
    return metrics_for_prediction(entry(rel),task,regime,cache)


def validate_completed(job,task,manifest_sha,cache,with_metrics=True):
    rel = OUT + "completed/" + job["id"] + ".json"
    record = c.read(rel)
    require(record["status"] == "COMPLETED" and record["completed_fit"] and record["evaluation_completed"], "Partial job is not completed")
    require(record["job"] == job and record["task"] == task and record["bundle_manifest_sha256"] == manifest_sha, "Completion identity")
    pf = check_preflight(record["preflight"]["path"],manifest_sha)
    bound(record["preflight"])
    require(record["runtime"] == pf["runtime"], "Completion runtime provenance")
    source = None
    if job["phase"] == "adaptation":
        source_record = c.read(OUT + "completed/" + job["depends_on"] + ".json")
        require(source_record["status"] == "COMPLETED" and source_record["completed_fit"] and
                source_record["bundle_manifest_sha256"] == manifest_sha, "Authoritative source prerequisite")
        source = source_record["checkpoint"]
        bound(source)
    require(record["checkpoint"]["path"].startswith(OUT+"attempts/"+job["id"]+"/"), "No CPU or other-job checkpoint")
    cp = torch.load(bound(record["checkpoint"]),map_location="cpu",weights_only=True)
    require(cp["metadata"] == expected_metadata(job,task,manifest_sha,record["runtime"],source), "Checkpoint authority/updates")
    require(cp["optimizer_state"]["state"] and all(int(s["step"].item())==job["updates"]
            for s in cp["optimizer_state"]["state"].values()), "Every optimizer parameter reached final budget")
    require(sum(t.numel() for t in cp["model_state"].values())==12939 and
            len(cp["optimizer_state"]["state"])==len(cp["model_state"]),"All registered parameters optimized")
    require(all(torch.isfinite(t).all() for t in cp["model_state"].values()), "Finite final checkpoint")
    expected_counts = {i:0 for i in task["source_cities"]} if job["phase"]=="source" else {task["target"]:300}
    if job["phase"]=="source":
        for i in c.city_sequence(task["source_cities"],task["seeds"]["source-city-schedule"]): expected_counts[i]+=1
    require(cp["city_updates"] == expected_counts, "Exact frozen city schedule")
    group = cp["optimizer_state"]["param_groups"]
    require(len(group)==1 and group[0]["lr"]==(1e-3 if job["phase"]=="source" else 2e-4) and
            group[0]["betas"]==(0.9,0.999) and group[0]["eps"]==1e-8 and group[0]["weight_decay"]==1e-4, "Frozen AdamW configuration")
    require(record["checkpoint_roundtrip_max_abs"]==0.0 and record["optimizer_initial_state_empty"], "Fresh optimizer and checkpoint roundtrip")
    require(record["logs"][0]["step"]==1 and record["logs"][-1]["step"]==job["updates"],"Update log endpoints")
    bound(record["metric"]["prediction"])
    require(record["metric"]["prediction"]["path"].startswith(OUT+"attempts/"+job["id"]+"/"), "Prediction job namespace")
    if with_metrics:
        recomputed=metrics_for_prediction(record["metric"]["prediction"],task,
                    "zero_shot" if job["phase"]=="source" else "7d_fine_tune",cache)
        require(recomputed == record["metric"], "Independently recomputed persisted-prediction metrics")
    return record


def run_job(job_id,manifest_sha,pf):
    jm,cache=verify_package(manifest_sha); rt=runtime(); check_preflight(pf,manifest_sha,rt)
    job=next(j for j in jm["jobs"] if j["id"]==job_id)
    task=next(t for t in jm["tasks"] if t["task_id"]==job["task_id"])
    complete=OUT+"completed/"+job_id+".json"
    if path(complete).exists():
        validate_completed(job,task,manifest_sha,cache)
        print("SKIP validated completed immutable job "+job_id,flush=True); return
    source=None
    if job["phase"]=="adaptation":
        source_job=next(j for j in jm["jobs"] if j["id"]==job["depends_on"])
        source=validate_completed(source_job,task,manifest_sha,cache)["checkpoint"]
    attempt=OUT+"attempts/"+job_id+"/"+uuid.uuid4().hex+"/"
    write(attempt+"started.json",{"job":job,"task":task,"initial_optimizer_updates":0,
          "bundle_manifest_sha256":manifest_sha,"runtime":rt,"source_checkpoint":source,
          "local_cpu_checkpoint_loaded":False,"partial_checkpoint_resumed":False,"created_utc":c.utc()})
    city_ids=sorted(set(task["source_cities"]+[task["target"]])) if job["phase"]=="source" else [task["target"]]
    cities={i:CudaCity(cache["cities"][str(i)]) for i in city_ids}
    c.set_seed(task["seed"])
    model=c.GraphGRU(c.configuration(task["candidate"])).to("cuda")
    phase=job["phase"]
    if source:
        cp=torch.load(bound(source),map_location="cpu",weights_only=True)
        model.load_state_dict(cp["model_state"],strict=True)
        del cp
        c.set_seed(task["seeds"]["fine-tune"])
    trainer=c.NeuralTrainer(model,c.optimizer(1e-3 if phase=="source" else 2e-4),
            c.TrainerConfig(seed=task["seed"],output_mode=c.MODES[task["candidate"]],gradient_clip_global_norm=1.0,checkpoint_mode="final"))
    require(not trainer.optimizer.state and trainer.step==0,"Fresh optimizer from update zero")
    sequence=c.city_sequence(task["source_cities"],task["seeds"]["source-city-schedule"]) if phase=="source" else [task["target"]]*300
    rng=np.random.default_rng(task["seeds"]["source-anchor-sampling" if phase=="source" else "fine-tune"])
    counts={i:0 for i in set(sequence)}; logs=[]; start=time.monotonic()
    meta=expected_metadata(job,task,manifest_sha,rt,source)
    print(job_id+" START from optimizer update 0",flush=True)
    for step,city_id in enumerate(sequence,1):
        city=cities[city_id]; anchors=city.source_anchors if phase=="source" else city.adaptation_anchors
        indices=rng.choice(anchors,size=16,replace=True).astype(np.int64)
        log=trainer.train_step(city.batch(indices,phase)); counts[city_id]+=1
        if step==1 or step%100==0: logs.append(asdict(log))
        if step%2000==0:
            partial={**meta,"updates_completed":step,"completed_fit":False,"evaluation_completed":False}
            checkpoint(attempt+f"partial_{step:05d}.pt",model,trainer,partial,rng,counts)
            print(f"{job_id} {step}/{job['updates']} elapsed={time.monotonic()-start:.1f}s",flush=True)
        if len(trainer.logs)>=1000:trainer.logs.clear()
    require(trainer.step==job["updates"],"Full fixed update budget")
    cp_entry=checkpoint(attempt+"final.pt",model,trainer,meta,rng,counts)
    saved=torch.load(bound(cp_entry),map_location="cpu",weights_only=True)
    restored=c.GraphGRU(c.configuration(task["candidate"])).to("cuda")
    restored.load_state_dict(saved["model_state"],strict=True); restored.eval(); model.eval()
    probe=cities[task["target"]].inputs(np.array([int((c.Contract().boundaries["HD"]-c.Contract().boundaries["H0"])//c.HOUR)]))
    with torch.no_grad():
        require(torch.equal(model(**probe).count_prediction,restored(**probe).count_prediction),"Exact CUDA checkpoint reload")
    metric=evaluate(restored,cities[task["target"]],task,"zero_shot" if phase=="source" else "7d_fine_tune",attempt,cache)
    record={"status":"COMPLETED","completed_fit":True,"evaluation_completed":True,"job":job,"task":task,
            "bundle_manifest_sha256":manifest_sha,"runtime":rt,"preflight":entry(pf),"checkpoint":cp_entry,
            "metric":metric,"checkpoint_roundtrip_max_abs":0.0,"optimizer_initial_state_empty":True,
            "logs":logs,"seconds":time.monotonic()-start,"completed_utc":c.utc()}
    write(complete,record)
    validate_completed(job,task,manifest_sha,cache)
    print(job_id+" COMPLETE AND VALIDATED",flush=True)


def validate_all(manifest_sha):
    jm,cache=verify_package(manifest_sha)
    expected={j["id"]+".json" for j in jm["jobs"]}
    actual={p.name for p in path(OUT+"completed").glob("*.json")}
    require(actual==expected,"Missing or unregistered completed jobs; no selection permitted")
    tasks={t["task_id"]:t for t in jm["tasks"]}; rows=[]; records=[]
    runtime_used=None
    for j in jm["jobs"]:
        r=validate_completed(j,tasks[j["task_id"]],manifest_sha,cache)
        if runtime_used is None:runtime_used=r["runtime"]
        require(r["runtime"]==runtime_used,"Runtime changed within scientific execution")
        rows.append(r["metric"]);records.append(entry(OUT+"completed/"+j["id"]+".json"))
    return {"status":"PASS","purpose":"READ_ONLY_POSTRUN_VALIDATION","source_fits":24,"adaptation_fits":24,
            "evaluation_records":48,"training_steps":0,"model_forward_calls":0,"files_written":[],
            "bundle_manifest_sha256":manifest_sha,"job_map_sha256":JOBMAP_SHA,"execution_records":records,
            "final_target_labels_accessed":False},rows


def freeze(manifest_sha):
    report,rows=validate_all(manifest_sha)
    selected=c.select(rows)
    decision={"status":"FROZEN_VALIDATED","protocol_version":"2.2","stage":1,
              "execution_environment":ENVIRONMENT,"bundle_manifest_sha256":manifest_sha,
              "job_map_sha256":JOBMAP_SHA,"scientific_manifest_sha256":DESIGN_SHA,
              "development_data_manifest_sha256":c.DATA_SHA,"implementation_manifest_sha256":c.IMPL_SHA,
              "selected_candidate":selected["selected"],"selected_mode":c.MODES[selected["selected"]],
              "selection":selected,"postrun_validation":report,"local_cpu_results_used":False,
              "stage2_started":False,"stage2b_started":False,"final_target_labels_accessed":False}
    rel=OUT+"stage1_decision_manifest.json"
    if path(rel).exists():require(c.read(rel)==decision,"Existing decision differs")
    else:write(rel,decision)
    print(json.dumps({"status":"STAGE1_FROZEN_STOP","decision":entry(rel),"selected_candidate":selected["selected"]}),flush=True)


def launch(manifest_sha):
    pf=preflight(manifest_sha)["path"]
    jm=c.read(JOBMAP)
    # Each immutable source/adaptation fit has its own Python process. The
    # coordinator has no model state and can be restarted without losing jobs.
    for j in jm["jobs"]:
        subprocess.run([sys.executable,"-X","utf8","-B","-m","research.stage1_v2_2.a40",
                        "job","--manifest-sha256",manifest_sha,"--preflight",pf,"--job-id",j["id"]],cwd=ROOT,check=True)
    # A separate read-only validator must succeed before selection/freeze.
    subprocess.run([sys.executable,"-X","utf8","-B","-m","research.stage1_v2_2.a40",
                    "validate","--manifest-sha256",manifest_sha],cwd=ROOT,check=True)
    freeze(manifest_sha)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("package-check","preflight","launch","job","validate","freeze"))
    p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--preflight");p.add_argument("--job-id")
    a=p.parse_args()
    if a.mode=="package-check":
        verify_package(a.manifest_sha256);print(json.dumps({"status":"PASS","scope":"PACKAGE_ONLY_NO_CUDA_CLAIM","scientific_training_steps":0}))
    elif a.mode=="preflight":preflight(a.manifest_sha256)
    elif a.mode=="launch":
        with execution_lock("coordinator"):launch(a.manifest_sha256)
    elif a.mode=="job":
        require(a.job_id and a.preflight,"Immutable job ID and preflight path required")
        require(a.job_id in {j["id"] for j in c.read(JOBMAP)["jobs"]},"Unregistered job ID")
        with execution_lock(a.job_id):run_job(a.job_id,a.manifest_sha256,a.preflight)
    elif a.mode=="validate":
        report,rows=validate_all(a.manifest_sha256)
        decision=OUT+"stage1_decision_manifest.json"
        if path(decision).exists():
            d=c.read(decision)
            require(d["selection"]==c.select(rows) and d["postrun_validation"]==report and
                    d["selected_candidate"]==d["selection"]["selected"] and
                    d["selected_mode"]==c.MODES[d["selected_candidate"]] and
                    d["job_map_sha256"]==JOBMAP_SHA and d["bundle_manifest_sha256"]==a.manifest_sha256 and
                    d["status"]=="FROZEN_VALIDATED" and not d["local_cpu_results_used"] and
                    not d["stage2_started"] and not d["stage2b_started"],"Frozen decision verification")
            report={**report,"decision":entry(decision)}
        print(json.dumps(report,indent=2))
    elif a.mode=="freeze":freeze(a.manifest_sha256)


if __name__=="__main__":main()
