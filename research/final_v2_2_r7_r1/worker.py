"""One fresh process executes exactly one immutable R7 neural-fit job."""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import core

def _models():
    import torch
    from research.models.common import NeuralModelConfig
    from research.models.graph_gru import GraphGRU
    from research.models.vanilla_gru import VanillaGRU
    from research.training.trainer import NeuralTrainer, OptimizerConfig, TrainerConfig
    return torch, NeuralModelConfig, GraphGRU, VanillaGRU, NeuralTrainer, OptimizerConfig, TrainerConfig

def _optimizer_config(phase: str):
    _, _, _, _, _, OptimizerConfig, _ = _models()
    if phase == "source": return OptimizerConfig(learning_rate=1e-3, weight_decay=1e-4)
    if phase == "adaptation": return OptimizerConfig(learning_rate=2e-4, weight_decay=1e-4)
    raise ValueError("unknown optimizer phase")

def _model(job):
    _, NeuralModelConfig, GraphGRU, VanillaGRU, _, _, _ = _models()
    config = NeuralModelConfig(model_type="vanilla_gru" if job["architecture"].startswith("VGRU") else "graph_gru",
        hidden_size=32, dropout=0.0, output_mode="log1p_target")
    return (VanillaGRU(config) if config.model_type == "vanilla_gru" else GraphGRU(config)), config

def _trainer(model, phase, seed):
    _, _, _, _, NeuralTrainer, _, TrainerConfig = _models()
    trainer = NeuralTrainer(model, _optimizer_config(phase), TrainerConfig(seed=seed,
        output_mode="log1p_target", gradient_clip_global_norm=1.0, checkpoint_mode="final"))
    if trainer.optimizer.state or trainer.step != 0: raise RuntimeError("optimizer was not fresh")
    return trainer

def _sample(city, rng, device, vanilla=False):
    import numpy as np
    if len(city.eligible_origins) == 0: raise RuntimeError("no eligible fitting origin")
    indices = rng.choice(city.eligible_origins, size=16, replace=True).astype(np.int64)
    return city.vanilla_batch(indices, device) if vanilla else city.graph_batch(indices, device)

def _source_schedule(job):
    values = list(core.SOURCES) * (12000 // len(core.SOURCES))
    if len(values) != 12000: raise RuntimeError("source schedule is not exactly balanced")
    random.Random(core.seed_for(job, "source-city-schedule")).shuffle(values)
    return values

def _load_dependency(job, device, package_sha):
    import torch
    from .validation import validate_one_completion
    source_job = next(x for x in core.read_json(core.JOB_MANIFEST)["fits"] if x["job_id"] == job["depends_on"])
    record = validate_one_completion(source_job, package_sha)
    checkpoint = torch.load(core.path(record["output_checkpoint"]["path"]), map_location=device)
    state = checkpoint.get("forecast_state", checkpoint["model_state"])
    return state, record["output_checkpoint"]

def _checkpoint(job, attempt, model, optimizer, metadata):
    import torch
    payload = {"schema_version": "final_v2_2_r7_r1.checkpoint.2", "metadata": metadata,
               "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
    if hasattr(model, "forecast"): payload["forecast_state"] = model.forecast.state_dict()
    buffer = io.BytesIO(); torch.save(payload, buffer)
    return core.atomic_bytes(f"{core.OUT}/attempts/{job['job_id']}/{attempt}/checkpoint.pt", buffer.getvalue())

def _train_production(job, device, package_sha):
    import numpy as np
    import torch
    from . import data
    from .grl_final import FinalSourceInvariantGraphGRU, source_step
    init_seed = core.seed_for(job, "model-initialization")
    core.configure_torch(init_seed, str(device))
    rng = np.random.default_rng(core.seed_for(job, "anchor-sampling"))
    diagnostics = []
    source_checkpoint = None
    if job["category"] == "source":
        cities = {c: data.source_city(c) for c in core.SOURCES}
        if job["method"] == "grl_l50_constant_source_graphgru":
            model = FinalSourceInvariantGraphGRU().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=1e-4)
            for step, city_id in enumerate(_source_schedule(job), 1):
                log = source_step(model, optimizer, _sample(cities[city_id], rng, device), core.DOMAIN_MAP[city_id])
                if step == 1 or step % 1000 == 0 or step == 12000: diagnostics.append({"step": step, **log})
        else:
            model, _ = _model(job); model = model.to(device); trainer = _trainer(model, "source", init_seed)
            for step, city_id in enumerate(_source_schedule(job), 1):
                log = trainer.train_step(_sample(cities[city_id], rng, device))
                if step == 1 or step % 1000 == 0 or step == 12000: diagnostics.append(asdict(log))
                if len(trainer.logs) > 1000: trainer.logs.clear()
            optimizer = trainer.optimizer
    elif job["category"] in ("target_only", "adaptation"):
        city = data.target_city(job["target_city_id"], job["budget"])
        model, _ = _model(job); model = model.to(device)
        if job["category"] == "adaptation":
            state, source_checkpoint = _load_dependency(job, device, package_sha); model.load_state_dict(state, strict=True)
        trainer = _trainer(model, "adaptation", init_seed)
        for step in range(1, job["updates"] + 1):
            log = trainer.train_step(_sample(city, rng, device, vanilla=job["architecture"].startswith("VGRU")))
            if step == 1 or step % 100 == 0 or step == job["updates"]: diagnostics.append(asdict(log))
        optimizer = trainer.optimizer
    elif job["category"] == "pooled":
        sources = {c: data.source_city(c) for c in core.SOURCES}
        target = data.target_city(job["target_city_id"], job["budget"])
        model, _ = _model(job); model = model.to(device); trainer = _trainer(model, "source", init_seed)
        tokens = [("source", c) for c in _source_schedule(job)] + [("target", job["target_city_id"])] * core.UPDATES[job["budget"]]
        random.Random(core.seed_for(job, "pooled-token-shuffle")).shuffle(tokens)
        for step, (kind, city_id) in enumerate(tokens, 1):
            trainer.optimizer.param_groups[0]["lr"] = 1e-3 if kind == "source" else 2e-4
            log = trainer.train_step(_sample(sources[city_id] if kind == "source" else target, rng, device))
            if step == 1 or step % 1000 == 0 or step == len(tokens): diagnostics.append(asdict(log))
            if len(trainer.logs) > 1000: trainer.logs.clear()
        optimizer = trainer.optimizer
    else:
        raise RuntimeError("unsupported frozen job family")
    if len(diagnostics) == 0: raise RuntimeError("fit produced no diagnostics")
    return model, optimizer, diagnostics, source_checkpoint

def _legal_input_checkpoint(job, package_sha):
    if job["category"] != "adaptation": return None
    from .validation import validate_one_completion
    source_job=next(x for x in core.read_json(core.JOB_MANIFEST)["fits"] if x["job_id"]==job["depends_on"])
    parent=validate_one_completion(source_job,package_sha)
    return parent["output_checkpoint"]

def execute(job_id: str, package_sha: str, attempt_uuid: str, device_name: str = "cuda"):
    from . import bindings, data, validation
    from .locks import LifetimeLock
    if not uuid.UUID(attempt_uuid) or len(attempt_uuid) != 36: raise ValueError("canonical UUID attempt required")
    validation.package_check(package_sha)
    jobs=core.read_json(core.JOB_MANIFEST)["fits"]; matches=[x for x in jobs if x["job_id"]==job_id]
    if len(matches)!=1: raise RuntimeError("job is not in immutable 410-job universe")
    job=matches[0]
    owner={"schema_version":"final_v2_2_r7_r1.job_claim.1","job_id":job_id,"attempt_uuid":attempt_uuid,
           "pid":os.getpid(),"package_manifest_sha256":package_sha,"acquired_utc":core.utc()}
    claim=LifetimeLock(core.path(f"{core.OUT}/locks/jobs/{job_id}.lock",output=True),
                       core.path(f"{core.OUT}/locks/jobs/{job_id}.owner.json",output=True),owner)
    with claim:
        completion_rel=f"{core.OUT}/completed/{job_id}.json"
        if core.path(completion_rel).exists():
            validation.validate_one_completion(job,package_sha)
            return {"status":"ALREADY_COMPLETED_VALID","job_id":job_id}
        input_checkpoint=_legal_input_checkpoint(job,package_sha)
        binding=data.structural_job_binding(job)
        scientific=bindings.expected(job,package_sha,input_checkpoint,device_name)
        runtime=scientific["runtime"]
        attempt_root=f"{core.OUT}/attempts/{job_id}/{attempt_uuid}"
        core.atomic_json(attempt_root+"/started.json",{"status":"STARTED_FROM_UPDATE_0","job":job,
            "attempt_uuid":attempt_uuid,"package_manifest_sha256":package_sha,
            "scientific_execution_binding_sha256":scientific["scientific_execution_binding_sha256"],
            "data_binding":binding,"runtime":runtime,"started_utc":core.utc(),"partial_resume":False})
        device=core.configure_torch(runtime["seed"],device_name)
        model,optimizer,diagnostics,source_checkpoint=_train_production(job,device,package_sha)
        if source_checkpoint != input_checkpoint: raise RuntimeError("resolved input checkpoint changed during fit")
        metadata={"schema_version":bindings.CHECKPOINT_SCHEMA,"job_id":job_id,"job_manifest_sha256":core.FROZEN[core.JOB_MANIFEST],
            "package_manifest_sha256":package_sha,"code_manifest_sha256":core.sha(core.CODE_MANIFEST),
            "implementation_contract_sha256":core.sha(core.IMPLEMENTATION_CONTRACT),
            "final_data_manifest_sha256":scientific["authorities"]["final_data_manifest_sha256"],
            "protocol_hashes":scientific["authorities"]["protocol_hashes"],
            "grl_clarification_hashes":scientific["authorities"]["grl_clarification_hashes"],
            "scientific_execution_binding":scientific,
            "scientific_execution_binding_sha256":scientific["scientific_execution_binding_sha256"],
            "seed":job["seed"],"method":job["method"],"model_binding":scientific["model"],
            "data_binding":binding,"input_checkpoint":input_checkpoint,"runtime":runtime,
            "optimizer":scientific["optimizer"],"expected_updates":job["updates"],"completed_updates":job["updates"],
            "early_stopping":False,"partial_resume":False}
        checkpoint=_checkpoint(job,attempt_uuid,model,optimizer,metadata)
        record={"schema_version":bindings.COMPLETION_SCHEMA,"status":"COMPLETED_VALID_FIT","success":True,
            "job_id":job_id,"job_manifest_sha256":core.FROZEN[core.JOB_MANIFEST],
            "package_manifest_sha256":package_sha,"code_manifest_sha256":core.sha(core.CODE_MANIFEST),
            "implementation_contract_sha256":core.sha(core.IMPLEMENTATION_CONTRACT),
            "final_data_manifest_sha256":scientific["authorities"]["final_data_manifest_sha256"],
            "protocol_hashes":scientific["authorities"]["protocol_hashes"],
            "grl_clarification_hashes":scientific["authorities"]["grl_clarification_hashes"],
            "scientific_execution_binding":scientific,
            "scientific_execution_binding_sha256":scientific["scientific_execution_binding_sha256"],
            "method":job["method"],"phase":job["category"],"seed":job["seed"],
            "target":job["target_city_id"],"budget":job["budget"],"attempt_uuid":attempt_uuid,
            "model_binding":scientific["model"],"data_binding":binding,"input_checkpoint":input_checkpoint,
            "output_checkpoint":checkpoint,"optimizer":scientific["optimizer"],
            "optimizer_update_count":job["updates"],"runtime_determinism":runtime,
            "artifact_hashes":{"checkpoint":checkpoint["sha256"]},"diagnostics":diagnostics,
            "started_record":core.entry(attempt_root+"/started.json"),"completed_utc":core.utc(),
            "early_stopping":False,"partial_resume":False,
            "final_target_labels_accessed":False,"final_predictions_generated":False}
        validation.validate_completion_record(job,record,package_sha)
        if not claim.held: raise RuntimeError("job claim lost before completion publication")
        try:
            completion=core.atomic_json(completion_rel,record)
        except FileExistsError:
            validation.validate_one_completion(job,package_sha)
            raise RuntimeError("another writer published this completion; local evidence was not substituted")
        return {"status":"COMPLETED_VALID_FIT","job_id":job_id,"completion":completion}
def synthetic_execute(config_path: str, job_id: str, attempt_uuid: str):
    """Exercise real model/trainer/checkpoint mechanics on explicit synthetic data."""
    import numpy as np
    import torch
    from research.models.common import NeuralModelConfig
    from research.models.graph_gru import GraphGRU
    from research.training.trainer import NeuralTrainer,OptimizerConfig,TrainerConfig
    from .locks import LifetimeLock
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg.get("purpose")!="R7_NON_SCIENTIFIC_PRODUCTION_PATH_MICRO_TEST": raise PermissionError("synthetic purpose required")
    job=next(x for x in cfg["jobs"] if x["job_id"]==job_id)
    if not job_id.startswith("synthetic_") or job["updates"]>2: raise PermissionError("scientific job forbidden in fixture")
    root=Path(cfg["output_root"]).resolve(); root.mkdir(parents=True,exist_ok=True)
    owner={"schema_version":"r7.synthetic.job_claim.1","job_id":job_id,"attempt_uuid":attempt_uuid,"pid":os.getpid()}
    with LifetimeLock(root/"locks"/f"{job_id}.lock",root/"locks"/f"{job_id}.owner.json",owner):
        if cfg.get("hold_claim_seconds",0):
            import time; time.sleep(float(cfg["hold_claim_seconds"]))
        dest=root/f"{job_id}.completion.json"
        if dest.exists(): raise FileExistsError("synthetic completion already exists")
        started=root/f"{job_id}.{attempt_uuid}.started.json"
        started.write_text(json.dumps({"job_id":job_id,"attempt_uuid":attempt_uuid,"before_optimizer_updates":True}),encoding="utf-8")
        core.configure_torch(job["seed"],"cpu")
        model=GraphGRU(NeuralModelConfig(model_type="graph_gru",hidden_size=32,dropout=0.0,output_mode="log1p_target"))
        trainer=NeuralTrainer(model,OptimizerConfig(learning_rate=2e-4,weight_decay=1e-4),
            TrainerConfig(seed=job["seed"],output_mode="log1p_target",gradient_clip_global_norm=1.0,checkpoint_mode="final"))
        batch={"model_inputs":{"x_hist":torch.zeros(2,24,3,2),"m_hist":torch.zeros(2,24,3,dtype=torch.bool),
            "x_week":torch.zeros(2,3,2),"x_static":torch.zeros(3,2),"x_calendar":torch.zeros(2,6),"adjacency":torch.eye(3)},
            "target":torch.ones(2,3),"mask":torch.ones(2,3,dtype=torch.bool)}
        for _ in range(job["updates"]): trainer.train_step(batch)
        cp=root/f"{job_id}.{attempt_uuid}.pt"; torch.save({"model_state":model.state_dict(),"optimizer_state":trainer.optimizer.state_dict()},cp)
        completion={"schema_version":"r7.synthetic.completion.2","status":"SYNTHETIC_COMPLETED","purpose":cfg["purpose"],
            "job_id":job_id,"attempt_uuid":attempt_uuid,"updates":job["updates"],"checkpoint_sha256":core.sha256_path(cp)}
        payload=json.dumps(completion,sort_keys=True).encode(); temp=root/f".{job_id}.{attempt_uuid}.pending"
        with temp.open("xb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        try: os.link(temp,dest)
        finally: temp.unlink(missing_ok=True)
        return completion
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--job-id", required=True)
    parser.add_argument("--attempt-uuid", required=True); parser.add_argument("--package-sha256")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--synthetic-config")
    args = parser.parse_args()
    result = synthetic_execute(args.synthetic_config, args.job_id, args.attempt_uuid) if args.synthetic_config else execute(args.job_id, args.package_sha256, args.attempt_uuid, args.device)
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__": main()
