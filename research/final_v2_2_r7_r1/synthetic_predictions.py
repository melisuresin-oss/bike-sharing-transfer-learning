"""Synthetic-only harness around the production model/checkpoint forward path."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import core

def run(config_path: str):
    import torch
    from research.models.common import NeuralModelConfig
    from research.models.graph_gru import GraphGRU
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"));root=Path(cfg["output_root"])
    if cfg.get("purpose")!="R7_NON_SCIENTIFIC_PRODUCTION_PATH_MICRO_TEST":raise PermissionError("synthetic purpose required")
    job=cfg["jobs"][0];completion=json.loads((root/f"{job['job_id']}.completion.json").read_text())
    checkpoint_path=next(root.glob(f"{job['job_id']}.*.pt"));checkpoint=torch.load(checkpoint_path,map_location="cpu")
    model=GraphGRU(NeuralModelConfig(model_type="graph_gru",hidden_size=32,dropout=0.0,output_mode="log1p_target"));model.load_state_dict(checkpoint["model_state"],strict=True);model.eval()
    with torch.no_grad():
        output=model(x_hist=torch.zeros(2,24,3,2),m_hist=torch.zeros(2,24,3,dtype=torch.bool),x_week=torch.zeros(2,3,2),x_static=torch.zeros(3,2),x_calendar=torch.zeros(2,6),adjacency=torch.eye(3)).count_prediction.numpy().reshape(-1)
    available=np.isfinite(output)&(output>=0);opaque=np.arange(len(output),dtype=np.uint64)
    payload=root/"synthetic_prediction.npz";np.savez_compressed(payload,prediction=output.astype(np.float32),availability=available,opaque_label_join_key=opaque)
    manifest={"status":"SYNTHETIC_PREDICTION_COMPLETE","payload_sha256":core.sha256_path(payload),"rows":len(output),"missing":0,"extra":0,"duplicates":0}
    manifest_path=root/"synthetic_prediction_manifest.json";manifest_path.write_text(json.dumps(manifest),encoding="utf-8")
    with np.load(payload,allow_pickle=False) as z:
        if len(z["prediction"])!=6 or not z["availability"].all() or len(np.unique(z["opaque_label_join_key"]))!=6:raise RuntimeError("synthetic completeness failed")
    commitment={"status":"SYNTHETIC_STRONG_COMMITMENT","completion_sha256":core.sha256_path(root/f"{job['job_id']}.completion.json"),"checkpoint_sha256":core.sha256_path(checkpoint_path),"prediction_manifest_sha256":core.sha256_path(manifest_path),"payload_sha256":core.sha256_path(payload)}
    commitment_path=root/"synthetic_commitment.json";commitment_path.write_text(json.dumps(commitment),encoding="utf-8")
    return {"status":"PASS_SYNTHETIC_PRODUCTION_MICRO_TEST","controller_completion":completion["status"],"prediction_rows":6,"commitment_sha256":core.sha256_path(commitment_path)}
