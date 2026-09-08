"""Frozen Stage-1 design, V2.2-only inputs, deterministic execution and selection."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import asdict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
import numpy as np
import torch
from research.development_data_v2_2.registry import DevelopmentRegistry
from research.v2_2.contract import Contract, DEVELOPMENT, HOUR, SPEC_SHA256, SEAL_SHA256, COHORT_SHA256, canonical_bytes
from research.v2_2.artifacts import ArtifactIdentity
from research.models.common import NeuralModelConfig
from research.models.graph_gru import GraphGRU
from research.training.trainer import NeuralTrainer, OptimizerConfig, TrainerConfig
from research.training.reproducibility import set_seed
from research.training.losses import raw_count_mae, log1p_target_mae
from research.evaluation.metrics import compute_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT="research/results/stage1_scale_loss_v2_2/"
CACHE="tmp/stage1_scale_loss_cache_v2_2/"
DATA="processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json"
DATA_SHA="140dd8c53cf51f48be5cf66b70d162c9f2cc6a9fa0b1b7039496633e385a6ab6"
IMPL="research/results/causal_history_implementation_v2_2/implementation_manifest.json"
IMPL_SHA="a1fe0ee486c55b1348e92d91edd42289dd49ce982129cdaa544691eafd605a9d"
TARGETS=(532,476,619,658)
SEEDS=(17,29,43)
CANDIDATES=("RAW","LOG1P")
MODES={"RAW":"raw_count","LOG1P":"log1p_target"}
REGIMES=("zero_shot","7d_fine_tune")
DESIGN_FILES=("DEVELOPMENT_SELECTION_PROTOCOL.md","SCALE_AND_LOSS_PROTOCOL.md","MODEL_SPECIFICATION_V2_1.md",
 "TRAINING_BUDGET_PROTOCOL.md","NEURAL_ARCHITECTURE_SPEC.md","NEURAL_REPRODUCIBILITY_PROTOCOL.md","EVALUATION_PROTOCOL_V2.md",
 "research/development/stage1_scale_loss.py","research/scripts/run_stage1_scale_loss.py","research/data/budget_sampler.py",
 "research/models/common.py","research/models/graph_gru.py","research/models/heads.py","research/training/losses.py",
 "research/training/trainer.py","research/training/reproducibility.py","research/evaluation/metrics.py")
NEW_CODE=("research/stage1_v2_2/__init__.py","research/stage1_v2_2/core.py","research/scripts/run_stage1_v2_2.py")
SELECTION_RULE={"primary":"count_space_MAE_over_all_observed_station_hour_evaluation_keys_within_cell",
 "seed_aggregation":"arithmetic_mean_of_17_29_43_within_each_fold_regime_cell",
 "cell_aggregation":"arithmetic_mean_of_8_equally_weighted_cells_4_folds_x_2_regimes",
 "primary_tie":"Decimal(str(mean)).quantize(Decimal('0.0001'), ROUND_HALF_UP)",
 "tie_1":"lower_unrounded_population_SD_of_8_seed_averaged_cell_MAEs","tie_2":"RAW",
 "secondary_metrics":["RMSE","WAPE_ratio","station_macro_MAE_minimum_24_valid_hours"],"secondary_metrics_affect_selection":False}


def utc():
    return datetime.now(timezone.utc).isoformat()


def path(rel,write=False):
    if not isinstance(rel,str) or "\\" in rel or ":" in rel or ".." in Path(rel).parts:
        raise PermissionError("Repository-relative path required")
    p=(ROOT/rel).resolve(); relative=p.relative_to(ROOT).as_posix().lower()
    if "final_labels" in relative or "sealed_final_evaluation_labels" in relative:
        raise PermissionError("Final evaluation label firewall")
    if write and not rel.startswith((OUT,CACHE)):
        raise PermissionError("Stage-1 V2.2 output scope")
    return p


def sha(rel):
    h=hashlib.sha256()
    with path(rel).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def read(rel):
    return json.loads(path(rel).read_text(encoding="utf-8"))


def write(rel,value):
    p=path(rel,True)
    if p.exists(): raise FileExistsError("Append-only output exists: "+rel)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+"\n",encoding="utf-8")
    return {"path":rel,"sha256":sha(rel)}


def canonical_hash(v):
    return hashlib.sha256(canonical_bytes(v)).hexdigest()


def configuration(candidate):
    if candidate not in CANDIDATES: raise ValueError("Unregistered candidate")
    return NeuralModelConfig(model_type="graph_gru",hidden_size=64,dropout=0.10,output_mode=MODES[candidate])


def optimizer(lr):
    return OptimizerConfig(name="AdamW",learning_rate=lr,betas=(0.9,0.999),epsilon=1e-8,weight_decay=1e-4)


def derived_seed(label,target,seed):
    # The historical namespace is a frozen RNG derivation, not an artifact version.
    return int.from_bytes(hashlib.sha256(f"stage1-v2.1|{label}|{target}|{seed}".encode()).digest()[:4],"big")


def city_sequence(cities,seed):
    seq=[cities[i%len(cities)] for i in range(12000)]
    random.Random(seed).shuffle(seq)
    return seq


def setup():
    torch.set_num_threads(2)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def runtime():
    return {"python":__import__("sys").version,"torch":str(torch.__version__),"numpy":np.__version__,
            "device":"cpu","cuda_available_to_runtime":torch.cuda.is_available(),"threads":torch.get_num_threads(),
            "interop_threads":torch.get_num_interop_threads(),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_deterministic":torch.backends.cudnn.deterministic,
            "tf32":False,"CUBLAS_WORKSPACE_CONFIG":os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def authority():
    assert sha(DATA)==DATA_SHA and sha(IMPL)==IMPL_SHA
    reg=DevelopmentRegistry(ROOT,DATA,DATA_SHA)
    verification=read("research/results/causal_history_audit_v2_2/development_verification_report.json")
    assert verification["status"]=="PASS" and verification["check_count"]==960
    assert verification["manifest_sha256"]==DATA_SHA
    assert sha("research/development_data_v2_2/registry.py")==verification["registry_sha256"]
    assert sha("research/scripts/verify_development_data_v2_2.py")==verification["verifier_sha256"]
    impl=read(IMPL)
    for group in ("implementation_files_sha256","test_files_sha256","evidence_files_sha256"):
        for p,h in impl[group].items(): assert sha(p)==h,p
    seal=read("research/results/causal_history_spec_v2_2/causal_history_spec_seal.json")
    for group in ("artifact_sha256","governing_source_sha256"):
        for p,h in seal[group].items(): assert sha(p)==h,p
    return reg


def save_array(rel,a):
    p=path(rel,True)
    if p.exists(): raise FileExistsError(rel)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.save(p,a,allow_pickle=False)
    return {"path":rel,"sha256":sha(rel),"shape":list(a.shape),"dtype":str(a.dtype)}


def prepare_cache(reg):
    manifest_path=CACHE+"cache_manifest.json"
    if path(manifest_path).exists():
        m=read(manifest_path)
        assert m["data_manifest_sha256"]==DATA_SHA and m["protocol_version"]=="2.2"
        for city in m["cities"].values():
            for entry in city["arrays"].values(): assert sha(entry["path"])==entry["sha256"]
        return m
    c=reg.contract; h0,hd=(c.boundaries[k] for k in ("H0","HD"))
    fit=reg.json_artifact("processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json")
    # Complete sealed source/adaptation validation precedes every cache construction.
    for target in TARGETS:
        source=reg.sealed_fit(target=target)
        assert source.request.city_ids==tuple(i for i in DEVELOPMENT if i!=target)
        del source
        adaptation=reg.sealed_fit(target=target,budget="7")
        assert adaptation.request.start==hd-7*24*HOUR
        del adaptation
        print(f"Validated sealed fitting pools for {target}",flush=True)
    cities={}
    for city in DEVELOPMENT:
        a=reg.predictors(city); H=len(a["origin_us"]); N=len(a["station_ids"])
        assert np.all(a["x_hist"][:,0]==0)
        src=next(x for x in fit["source_city_partitions"] if x["city_id"]==city)
        s=reg.arrays(src)
        y=np.zeros((H,N),dtype=np.float32); mask=np.zeros((H,N),dtype=bool); ft=np.zeros((H,N),dtype=bool)
        pos={int(v):j for j,v in enumerate(a["station_ids"])}
        t=((s["origin_us"]-h0)//HOUR).astype(np.int64); j=np.array([pos[int(v)] for v in s["station_id"]])
        y[t,j]=s["count"]; mask[t,j]=True
        assert np.array_equal(s["origin_feature_sha256"],a["origin_feature_sha256"][t])
        adapt_entry=None
        if city in TARGETS:
            meta=next(x for x in fit["adaptations"] if x["city_id"]==city and x["budget"]=="7")
            adapt_entry=meta
            ss=reg.arrays(meta["data"])
            tt=((ss["origin_us"]-h0)//HOUR).astype(np.int64); jj=np.array([pos[int(v)] for v in ss["station_id"]])
            ft[tt,jj]=True
            assert np.all(mask[tt,jj]) and np.array_equal(y[tt,jj],ss["count"])
        adjacency,identity=reg.graph(city,8)
        payload={k:a[k] for k in ("origin_us","station_ids","x_hist","m_hist","x_week","x_static","x_calendar","origin_feature_sha256")}
        payload.update({"fit_count":y,"fit_mask":mask,"adaptation_7d_mask":ft,"adjacency":adjacency,
                        "source_anchors":np.flatnonzero(mask.any(axis=1)).astype(np.int64),
                        "adaptation_anchors":np.flatnonzero(ft.any(axis=1)).astype(np.int64)})
        arrays={k:save_array(CACHE+f"city_{city}/{k}.npy",v) for k,v in payload.items()}
        cities[str(city)]={"city_id":city,"station_count":N,"arrays":arrays,"source_snapshot":src,"adaptation_snapshot":adapt_entry,
                          "graph_identity":identity.as_json(),"predictor":reg.panels[city],"target_values_from_fit_snapshot_only":True}
        print(f"V2.2 fitting cache sealed for {city}",flush=True)
    m={"protocol_version":"2.2","role":"STAGE1_CAUSAL_FITTING_CACHE","data_manifest_sha256":DATA_SHA,
       "implementation_manifest_sha256":IMPL_SHA,"cohort_static_sha256":COHORT_SHA256,"cities":cities,
       "retrospective_labels_in_cache":False,"final_labels_accessed":False}
    write(manifest_path,m)
    return m


def freeze_design(cache):
    doc=path("SCALE_AND_LOSS_PROTOCOL.md").read_text()
    assert all(s in doc for s in ("softplus","expm1","equally","four decimal places","choose raw counts"))
    contract={"protocol_version":"2.2","stage":1,"candidates":list(CANDIDATES),"folds":list(TARGETS),"seeds":list(SEEDS),
      "regimes":list(REGIMES),"model":configuration("RAW").to_dict(),"candidate_output_modes":MODES,"graph_k":8,
      "source_updates":12000,"adaptation_updates":300,"batch_size":16,"source_optimizer":asdict(optimizer(1e-3)),
      "adaptation_optimizer":asdict(optimizer(2e-4)),"optimizer_reset_before_adaptation":True,"gradient_clip_global_norm":1.0,
      "checkpoint_rule":"final_step_only_for_evaluation_and_selection; intermediate checkpoints solely for exact recovery",
      "early_stopping":False,"source_sampling":"balanced_shuffled_12000_city_tokens; exposure difference <=1; uniform eligible anchors with replacement",
      "initialization_seed":"registered seed","data_seed_derivation":"SHA256(stage1-v2.1|label|target|registered_seed), first 4 bytes big endian, inherited unchanged",
      "sampling_labels":["source-city-schedule","source-anchor-sampling","fine-tune"],"selection":SELECTION_RULE,
      "precision":"float32 CPU; no AMP, no compilation or architecture rewrite","runtime":runtime(),
      "source_cutoff":"HD","adaptation_window":"[HD-7d,HD)","evaluation_window":"[HD,HF)",
      "history":"sealed V2.2 unchanged, including all-zero lag 1","expected_source_fits":24,"expected_adaptation_fits":24,
      "expected_evaluation_records":48,"design_source_sha256":{p:sha(p) for p in DESIGN_FILES},
      "code_sha256":{p:sha(p) for p in NEW_CODE},"data_manifest_sha256":DATA_SHA,"implementation_manifest_sha256":IMPL_SHA,
      "specification_sha256":SPEC_SHA256,"specification_seal_sha256":SEAL_SHA256,"cohort_static_sha256":COHORT_SHA256,
      "cache_manifest_sha256":sha(CACHE+"cache_manifest.json"),"historical_result_values_used":False,
      "ambiguities_resolved_from_authorities":{"cell_count":"4 folds x 2 regimes = 8; SCALE_AND_LOSS_PROTOCOL plus historical executable selection",
       "MAE":"observed station-hour MAE per cell; station-macro MAE remains secondary",
       "tie_rounding":"historical executable Decimal ROUND_HALF_UP; unrounded population SD tie-break",
       "device":"same validated CPU runtime; CUDA determinism configured but no CUDA kernel used"},
      "contradictions":[],"stage2_authorized":False,"stage2b_authorized":False,"final_labels_accessed":False}
    design=write(OUT+"scientific_manifest.json",contract)
    selection=write(OUT+"selection_rule.json",SELECTION_RULE)
    fit=read("processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json")
    jobs=[]; tasks=[]
    for candidate in CANDIDATES:
        for target in TARGETS:
            for seed in SEEDS:
                base=f"{candidate.lower()}_target-{target}_seed-{seed}"
                source=next(r for r in fit["source_folds"] if r["target_city"]==target)
                adapt=next(r for r in fit["adaptations"] if r["city_id"]==target and r["budget"]=="7")
                task={"task_id":base,"candidate":candidate,"target":target,"seed":seed,"source_cities":[i for i in DEVELOPMENT if i!=target],
                      "source_snapshot":source,"adaptation_snapshot":adapt,"source_id":base+"_source","adaptation_id":base+"_7d",
                      "seeds":{"initialization":seed,"source-city-schedule":derived_seed("source-city-schedule",target,seed),
                       "source-anchor-sampling":derived_seed("source-anchor-sampling",target,seed),"fine-tune":derived_seed("fine-tune",target,seed)}}
                tasks.append(task)
                jobs.extend([{"id":task["source_id"],"phase":"source","updates":12000,"task_id":base,"depends_on":None},
                             {"id":task["adaptation_id"],"phase":"adaptation","updates":300,"task_id":base,"depends_on":task["source_id"]}])
    jobmap=write(OUT+"job_map.json",{"protocol_version":"2.2","scientific_manifest":design,"selection_rule":selection,"jobs":jobs,"tasks":tasks,
                  "source_fits":24,"adaptation_fits":24,"evaluation_records":48,"immutable":True})
    return contract,jobmap


def checkpoint(rel,model,opt,metadata,rng=None,extra=None):
    payload={"metadata":metadata,"model_state":model.state_dict(),"optimizer_state":opt.state_dict(),
             "torch_rng":torch.get_rng_state(),"python_rng":random.getstate(),"anchor_rng":rng.bit_generator.state if rng else None,"extra":extra or {}}
    p=path(rel,True)
    if p.exists(): raise FileExistsError(rel)
    p.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,p)
    return {"path":rel,"sha256":sha(rel)}


def load_checkpoint(entry,model,opt,expected):
    if not entry["path"].startswith(OUT): raise PermissionError("No legacy checkpoint accepted")
    if sha(entry["path"])!=entry["sha256"]: raise ValueError("Checkpoint hash mismatch")
    p=torch.load(path(entry["path"]),map_location="cpu",weights_only=True)
    if p["metadata"]!=expected or expected["protocol_version"]!="2.2":
        raise ValueError("Checkpoint metadata mismatch")
    model.load_state_dict(p["model_state"],strict=True); opt.load_state_dict(p["optimizer_state"])
    return p


def preflight(reg,cache,contract,jobmap):
    results=[]
    def check(n,name,ok):
        results.append({"number":n,"name":name,"status":"PASS" if ok else "FAIL"})
        if not ok: raise RuntimeError("PREFLIGHT_STOP: "+name)
    checks=[
      ("specification_bindings",contract["specification_sha256"]==SPEC_SHA256 and contract["specification_seal_sha256"]==SEAL_SHA256),
      ("implementation_binding",sha(IMPL)==IMPL_SHA),("development_data_binding",sha(DATA)==DATA_SHA),
      ("fixed_development_cohort",sum(x["station_count"] for x in cache["cities"].values())==523),
      ("fold_source_rosters",all(len(t["source_cities"])==7 for t in read(jobmap["path"])["tasks"])),
      ("pseudo_target_exclusion",all(t["target"] not in t["source_cities"] for t in read(jobmap["path"])["tasks"])),
      ("source_snapshot_hashes",all(sha(t["source_snapshot"]["path"])==t["source_snapshot"]["sha256"] for t in read(jobmap["path"])["tasks"])),
      ("adaptation_snapshot_hashes",all(sha(t["adaptation_snapshot"]["path"])==t["adaptation_snapshot"]["sha256"] for t in read(jobmap["path"])["tasks"])),
      ("no_final_target_input",set(map(int,cache["cities"]))==set(DEVELOPMENT))]
    for n,(name,ok) in enumerate(checks,1):check(n,name,ok)
    try: reg.predictors(195)
    except PermissionError: denied=True
    else: denied=False
    try: reg.path("processed/protocol_v2_1/development/development_panel.parquet")
    except PermissionError: legacy=True
    else: legacy=False
    check(10,"legacy_predictor_cross_loading_rejected",legacy)
    check(11,"exact_candidates",CANDIDATES==("RAW","LOG1P"))
    check(12,"exact_seeds",SEEDS==(17,29,43))
    jm=read(jobmap["path"])
    check(13,"48_parameter_changing_fits",len(jm["jobs"])==48 and len(jm["tasks"])==24 and len({j["id"] for j in jm["jobs"]})==48)
    set_seed(17); one=GraphGRU(configuration("RAW")); state={k:v.clone() for k,v in one.state_dict().items()}
    set_seed(17); two=GraphGRU(configuration("RAW"))
    check(14,"deterministic_initialization",all(torch.equal(state[k],v) for k,v in two.state_dict().items()))
    check(15,"deterministic_runtime",torch.are_deterministic_algorithms_enabled() and torch.get_num_threads()==2 and
          not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic and os.environ["CUBLAS_WORKSPACE_CONFIG"]==":4096:8")
    opt=torch.optim.AdamW(one.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4)
    owned=[id(p) for group in opt.param_groups for p in group["params"]]
    check(16,"optimizer_owns_exactly_all_parameters",len(owned)==len(set(owned)) and set(owned)=={id(p) for p in one.parameters()} and not opt.state)
    check(17,"exact_update_budgets",all(j["updates"]==(12000 if j["phase"]=="source" else 300) for j in jm["jobs"]))
    metadata={"protocol_version":"2.2","purpose":"NON_SCIENTIFIC_PREFLIGHT","data_manifest_sha256":DATA_SHA,"updates":0}
    entry=checkpoint(OUT+"preflight/untrained_roundtrip.pt",one,opt,metadata)
    opt2=torch.optim.AdamW(two.parameters(),lr=1e-3)
    load_checkpoint(entry,two,opt2,metadata)
    toy={"x_hist":torch.zeros(2,24,3,2),"m_hist":torch.zeros(2,24,3,dtype=torch.bool),"x_week":torch.zeros(2,3,2),
         "x_static":torch.zeros(3,2),"x_calendar":torch.zeros(2,6),"adjacency":torch.eye(3)}
    one.eval();two.eval()
    with torch.no_grad():
        before=one(**toy).count_prediction;after=two(**toy).count_prediction
    check(18,"checkpoint_roundtrip_zero_difference",torch.equal(before,after) and all(torch.equal(one.state_dict()[k],v) for k,v in two.state_dict().items()) and not opt2.state)
    check(19,"evaluation_labels_separate",cache["retrospective_labels_in_cache"] is False and all("target" not in x["arrays"] for x in cache["cities"].values()))
    check(20,"final_label_firewall",denied and legacy)
    y=torch.tensor([[0.,3.]]);pred=torch.tensor([[1.,2.]]);mask=torch.tensor([[True,True]])
    check(21,"RAW_loss_contract",raw_count_mae(pred,y,mask).item()==1.0)
    check(22,"LOG1P_loss_contract",torch.equal(log1p_target_mae(pred,y,mask),torch.mean(torch.abs(pred-torch.log1p(y)))))
    set_seed(17);one.train()
    with torch.no_grad(): d1=one(**toy).prediction
    set_seed(17)
    with torch.no_grad(): d2=one(**toy).prediction
    check(23,"deterministic_dropout_forward",torch.equal(d1,d2))
    check(24,"parameter_count_12939",sum(p.numel() for p in one.parameters())==12939)
    result={"status":"PASS","checks":results,"check_count":len(results),"scientific_training_steps":0,
            "scientific_evaluation_performed":False,"synthetic_forward_only":True,"checkpoint_roundtrip_max_abs":0.0,
            "runtime":runtime(),"job_map":jobmap,"final_target_evaluation_labels_accessed":False}
    return write(OUT+"preflight_report.json",result)


class City:
    def __init__(self,entry):
        self.entry=entry
        for k,e in entry["arrays"].items(): setattr(self,k,np.load(path(e["path"]),mmap_mode="r",allow_pickle=False))
        self.static=torch.tensor(np.array(self.x_static),dtype=torch.float32)
        self.graph=torch.tensor(np.array(self.adjacency),dtype=torch.float32)

    def inputs(self,indices):
        x=torch.from_numpy(np.array(self.x_hist[indices],dtype=np.float32))
        return {"x_hist":x,"m_hist":torch.from_numpy(np.array(self.m_hist[indices],dtype=bool)),
                "x_week":torch.from_numpy(np.array(self.x_week[indices],dtype=np.float32)),"x_static":self.static,
                "x_calendar":torch.from_numpy(np.array(self.x_calendar[indices],dtype=np.float32)),"adjacency":self.graph}

    def batch(self,indices,phase):
        return {"model_inputs":self.inputs(indices),"target":torch.from_numpy(np.array(self.fit_count[indices],dtype=np.float32)),
                "mask":torch.from_numpy(np.array((self.fit_mask if phase=="source" else self.adaptation_7d_mask)[indices],dtype=bool))}


def metadata(task,phase,updates,jm,contract):
    cities=task["source_cities"] if phase=="source" else [task["target"]]
    cache=read(CACHE+"cache_manifest.json")
    graphhash=canonical_hash({str(i):cache["cities"][str(i)]["graph_identity"] for i in cities})
    identity=ArtifactIdentity.create(Contract(),"checkpoint",Contract().boundaries["HD"],cities,
                 model_family="graph_gru",graph_binding_sha256=graphhash).as_json()
    snapshot=task["source_snapshot"] if phase=="source" else task["adaptation_snapshot"]
    return {"protocol_version":"2.2","artifact_identity":identity,"task_id":task["task_id"],"phase":phase,
            "updates_completed":updates,"architecture":configuration(task["candidate"]).to_dict(),"seed":task["seed"],
            "seeds":task["seeds"],"optimizer":asdict(optimizer(1e-3 if phase=="source" else 2e-4)),
            "job_map_sha256":jm,"scientific_manifest_sha256":sha(OUT+"scientific_manifest.json"),
            "data_manifest_sha256":DATA_SHA,"implementation_manifest_sha256":IMPL_SHA,"training_snapshot":snapshot,
            "runtime":runtime(),"code_sha256":contract["code_sha256"],"final_labels_accessed":False}


@torch.no_grad()
def evaluate(model,city,task,regime):
    model.eval(); h0=Contract().boundaries["H0"];hd=Contract().boundaries["HD"]
    F=int((hd-h0)//HOUR); indices=np.arange(F,len(city.origin_us))
    predictions=[]
    for start in range(0,len(indices),64):
        pp=model(**city.inputs(indices[start:start+64])).count_prediction.numpy()
        if not np.isfinite(pp).all() or np.any(pp<0):raise FloatingPointError("Invalid count-space predictions")
        predictions.append(pp)
    pred=np.concatenate(predictions)
    rel=OUT+f"predictions/{task['task_id']}_{regime}.npz"
    p=path(rel,True);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(rel)
    # Commit every immutable origin/station prediction BEFORE reading retrospective labels.
    np.savez_compressed(p,origin_us=np.array(city.origin_us[indices]),station_ids=np.array(city.station_ids),prediction_count=pred)
    artifact={"path":rel,"sha256":sha(rel)}
    reg=DevelopmentRegistry(ROOT,DATA,DATA_SHA)
    labels=reg.retrospective_labels(task["target"])
    mask=labels["observed"][indices];target=labels["count"][indices]
    stations=np.broadcast_to(city.station_ids[None,:],mask.shape)
    metrics=compute_metrics(target[mask],pred[mask],stations[mask]).as_dict()
    assert metrics["mae"] is not None and metrics["n"]==int(mask.sum())
    values=pred[mask].astype(np.float64);actual=target[mask].astype(np.float64)
    ratio=float(values.std()/actual.std()) if actual.std()>0 else None
    return {"candidate":task["candidate"],"target":task["target"],"seed":task["seed"],"regime":regime,**metrics,
            "prediction":artifact,"all_prediction_keys":int(pred.size),"prediction_key_sha256":canonical_hash(
              {"city":task["target"],"origins":list(map(int,city.origin_us[indices])),"stations":list(map(int,city.station_ids))}),
            "prediction_std":float(values.std()),"prediction_mean":float(values.mean()),"prediction_std_target_std_ratio":ratio,
            "near_constant_diagnostic":bool(values.std()<=1e-6 or (ratio is not None and ratio<0.05)),
            "evaluation_label_role":"separate_retrospective_development","evaluation_period":"[HD,HF)","final_labels_accessed":False}


def worker(task):
    setup(); start=time.monotonic()
    contract=read(OUT+"scientific_manifest.json"); jm=sha(OUT+"job_map.json")
    if read(OUT+"preflight_report.json")["status"]!="PASS":raise RuntimeError("No scientific launch without preflight PASS")
    if task not in read(OUT+"job_map.json")["tasks"]:raise ValueError("Unregistered task")
    for p,h in contract["code_sha256"].items():assert sha(p)==h,p
    cache=read(CACHE+"cache_manifest.json");assert sha(CACHE+"cache_manifest.json")==contract["cache_manifest_sha256"]
    for e in cache["cities"].values():
        for a in e["arrays"].values(): assert sha(a["path"])==a["sha256"]
    for snapshot in (task["source_snapshot"],task["adaptation_snapshot"]):
        assert sha(snapshot["path"])==snapshot["sha256"]
    destination=OUT+f"jobs/{task['task_id']}.json"
    if path(destination).exists():raise RuntimeError("Registered task already has execution record")
    cities={i:City(cache["cities"][str(i)]) for i in DEVELOPMENT}
    set_seed(task["seed"])
    model=GraphGRU(configuration(task["candidate"]))
    metrics=[];executions=[];source_checkpoint=None
    for phase,updates,lr,rngseed in (("source",12000,1e-3,task["seeds"]["source-anchor-sampling"]),
                                    ("adaptation",300,2e-4,task["seeds"]["fine-tune"])):
        if phase=="adaptation":set_seed(task["seeds"]["fine-tune"])
        trainer=NeuralTrainer(model,optimizer(lr),TrainerConfig(seed=task["seed"],output_mode=MODES[task["candidate"]],gradient_clip_global_norm=1.0,checkpoint_mode="final"))
        assert not trainer.optimizer.state
        sequence=city_sequence(task["source_cities"],task["seeds"]["source-city-schedule"]) if phase=="source" else [task["target"]]*300
        rng=np.random.default_rng(rngseed);counts={i:0 for i in set(sequence)};validcounts={i:0 for i in set(sequence)}
        step_logs=[];phase_start=time.monotonic()
        print(f"{task['task_id']} {phase} START",flush=True)
        for step,city_id in enumerate(sequence,1):
            city=cities[city_id]
            anchors=city.source_anchors if phase=="source" else city.adaptation_anchors
            indices=rng.choice(anchors,size=16,replace=True).astype(np.int64)
            log=trainer.train_step(city.batch(indices,phase))
            counts[city_id]+=1;validcounts[city_id]+=log.valid_targets
            if step==1 or step%(100 if phase=="source" else 25)==0:step_logs.append(asdict(log))
            if phase=="source" and step%2000==0:
                cp=checkpoint(OUT+f"recovery/{task['task_id']}_{phase}_{step:05d}.pt",model,trainer.optimizer,
                    metadata(task,phase,step,jm,contract),rng,{"city_updates":counts,"valid_targets":validcounts,"logs":step_logs})
                print(f"{task['task_id']} {phase} {step}/{updates} elapsed={time.monotonic()-phase_start:.1f}s",flush=True)
            if len(trainer.logs)>=1000:trainer.logs.clear()
        assert trainer.step==updates and sum(counts.values())==updates
        if phase=="source":assert max(counts.values())-min(counts.values())<=1
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        meta=metadata(task,phase,updates,jm,contract)
        entry=checkpoint(OUT+f"checkpoints/{task['task_id']}_{phase}.pt",model,trainer.optimizer,meta)
        probe=cities[task["target"]].inputs(np.array([int((Contract().boundaries["HD"]-Contract().boundaries["H0"])//HOUR)]))
        model.eval()
        with torch.no_grad():before=model(**probe).count_prediction.clone()
        restored=GraphGRU(configuration(task["candidate"]))
        restored_opt=torch.optim.AdamW(restored.parameters(),lr=lr,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-4)
        load_checkpoint(entry,restored,restored_opt,meta);restored.eval()
        with torch.no_grad():after=restored(**probe).count_prediction
        assert torch.equal(before,after)
        model=restored
        execution={"job_id":task["source_id"] if phase=="source" else task["adaptation_id"],"phase":phase,
          "updates_completed":updates,"optimizer":asdict(optimizer(lr)),"optimizer_state_reset":True,"city_updates":counts,
          "valid_target_exposures":validcounts,"checkpoint":entry,"source_checkpoint":source_checkpoint if phase=="adaptation" else None,
          "training_snapshot":task["source_snapshot"] if phase=="source" else task["adaptation_snapshot"],
          "checkpoint_roundtrip_max_abs":0.0,"seconds":time.monotonic()-phase_start,"logs":step_logs,"runtime":runtime()}
        executions.append(execution)
        if phase=="source":source_checkpoint=entry
        metrics.append(evaluate(model,cities[task["target"]],task,"zero_shot" if phase=="source" else "7d_fine_tune"))
        print(f"{task['task_id']} {phase} COMPLETE",flush=True)
    record={"status":"COMPLETED","protocol_version":"2.2","task":task,"job_map_sha256":jm,
            "scientific_manifest_sha256":sha(OUT+"scientific_manifest.json"),"data_manifest_sha256":DATA_SHA,
            "executions":executions,"metrics":metrics,"elapsed_seconds":time.monotonic()-start,"completed_utc":utc(),
            "final_labels_accessed":False,"stage2_started":False,"stage2b_started":False}
    return write(destination,record)


def select(rows):
    assert len(rows)==48
    cells=[];summary={}
    for candidate in CANDIDATES:
        vals=[]
        for target in TARGETS:
            for regime in REGIMES:
                rr=[r for r in rows if r["candidate"]==candidate and r["target"]==target and r["regime"]==regime]
                assert sorted(r["seed"] for r in rr)==list(SEEDS)
                mean=statistics.fmean(r["mae"] for r in rr)
                cells.append({"candidate":candidate,"target":target,"regime":regime,"mae_seed_mean":mean,
                              "mae_seed_population_sd":statistics.pstdev(r["mae"] for r in rr),"n":rr[0]["n"]})
                vals.append(mean)
        mean=statistics.fmean(vals); sd=statistics.pstdev(vals)
        summary[candidate]={"mean_count_space_MAE":mean,"rounded_four_decimals":str(Decimal(str(mean)).quantize(Decimal("0.0001"),rounding=ROUND_HALF_UP)),
                            "population_SD_across_8_cells":sd}
    raw,log=summary["RAW"],summary["LOG1P"]
    if raw["rounded_four_decimals"]!=log["rounded_four_decimals"]:
        winner=min(CANDIDATES,key=lambda x:summary[x]["mean_count_space_MAE"]);reason="lower_equal_cell_count_space_MAE"
    elif raw["population_SD_across_8_cells"]!=log["population_SD_across_8_cells"]:
        winner=min(CANDIDATES,key=lambda x:summary[x]["population_SD_across_8_cells"]);reason="four_decimal_tie_lower_population_SD"
    else:winner="RAW";reason="primary_and_dispersion_tie_RAW"
    return {"cells":cells,"candidates":summary,"selected":winner,"tie_break_path":reason,"selection_rule":SELECTION_RULE}


def close_stage1():
    jm=read(OUT+"job_map.json");entries=[];rows=[]
    for task in jm["tasks"]:
        rel=OUT+f"jobs/{task['task_id']}.json";r=read(rel)
        assert r["status"]=="COMPLETED" and r["task"]==task
        entries.append({"path":rel,"sha256":sha(rel)});rows.extend(r["metrics"])
    result=select(rows)
    summary=write(OUT+"candidate_summary.json",{"per_candidate_fold_seed_regime":rows,**result})
    decision=write(OUT+"stage1_decision_manifest.json",{"protocol_version":"2.2","stage":1,"status":"FROZEN_PENDING_POSTRUN_VERIFICATION",
        "selected_candidate":result["selected"],"selected_mode":MODES[result["selected"]],"selection":result,
        "scientific_manifest_sha256":sha(OUT+"scientific_manifest.json"),"job_map_sha256":sha(OUT+"job_map.json"),
        "specification_seal_sha256":SEAL_SHA256,"implementation_manifest_sha256":IMPL_SHA,"development_data_manifest_sha256":DATA_SHA,
        "fit_snapshot_manifest_sha256":sha("processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json"),
        "candidate_summary":summary,"execution_records":entries,"source_fits":24,"adaptation_fits":24,
        "final_labels_accessed":False,"stage2_started":False,"stage2b_started":False,"created_utc":utc()})
    # Historical results are opened ONLY after the independent V2.2 selection is frozen.
    old=read("research/results/stage1_scale_loss_v2_1/stage1_decision_manifest.json")
    comparison=write(OUT+"v2_1_descriptive_comparison.json",{"purpose":"DESCRIPTIVE_ONLY_AFTER_V2_2_SELECTION",
             "v2_1_decision":old,"v2_1_sha256":sha("research/results/stage1_scale_loss_v2_1/stage1_decision_manifest.json"),
             "v2_2_decision":decision,"used_in_selection":False})
    ledger={}
    for p in sorted(path(OUT).rglob("*")):
        if p.is_file():ledger[p.relative_to(ROOT).as_posix()]=sha(p.relative_to(ROOT).as_posix())
    write(OUT+"hash_ledger.json",{"protocol_version":"2.2","files":ledger,"excludes_self_and_later_verification":True})
    print(json.dumps({"status":"STAGE1_COMPLETED_PENDING_POSTRUN_VERIFICATION","decision":decision,"comparison":comparison}),flush=True)
