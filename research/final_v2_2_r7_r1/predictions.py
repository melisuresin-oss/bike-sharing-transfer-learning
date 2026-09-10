"""Label-free 468-pass prediction routing and independent completeness validation."""
from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
from . import core

LEARNED = {"target_only_vanilla", "target_only_graph", "ordinary_source_adapted", "pooled_graph",
           "grl_l50_constant_adapted", "ordinary_source_graphgru", "grl_l50_constant_source_graphgru"}
BASELINES = {"persistence", "seasonal_naive", "HA_SOURCE", "HA_TARGET"}

def _job_for(evaluation, jobs):
    method = evaluation["method"]
    if method == "ordinary_source_adapted": prefix = "ordinary_adaptation_"
    elif method == "grl_l50_constant_adapted": prefix = "grl_adaptation_"
    else: prefix = method + "_"
    matches = [j for j in jobs if j["method"] == method and j["seed"] == evaluation["seed"] and
               (j["target_city_id"] == evaluation["target_city_id"] or j["category"] == "source") and
               (j["budget"] == evaluation["budget"] or j["category"] == "source")]
    if len(matches) != 1: raise RuntimeError("evaluation-to-checkpoint routing is not unique: " + evaluation["evaluation_id"])
    return matches[0]

def _learned(evaluation, job, panel, completion, device):
    import torch
    from research.models.common import NeuralModelConfig
    from research.models.graph_gru import GraphGRU
    from research.models.vanilla_gru import VanillaGRU
    config = NeuralModelConfig(model_type="vanilla_gru" if job["architecture"].startswith("VGRU") else "graph_gru",
                               hidden_size=32, dropout=0.0, output_mode="log1p_target")
    model = (VanillaGRU(config) if config.model_type == "vanilla_gru" else GraphGRU(config)).to(device)
    checkpoint = torch.load(core.path(completion["output_checkpoint"]["path"]), map_location=device)
    model.load_state_dict(checkpoint.get("forecast_state", checkpoint["model_state"]), strict=True); model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(panel.origin_us), 64):
            indices = np.arange(start, min(start + 64, len(panel.origin_us)), dtype=np.int64)
            batch = panel.vanilla_batch(indices, device) if config.model_type == "vanilla_gru" else panel.graph_batch(indices, device)
            value = model(**batch["model_inputs"]).count_prediction.detach().cpu().numpy()
            if config.model_type == "vanilla_gru": value = value.reshape(len(indices), len(panel.station_ids))
            parts.append(value.astype(np.float32))
    prediction = np.concatenate(parts); available = np.isfinite(prediction) & (prediction >= 0)
    if not available.all(): raise RuntimeError("learned prediction is nonfinite or negative")
    return prediction, available

def _keys(city_id):
    rec = next(x for x in core.read_json(core.DATA_MANIFEST)["targets"] if x["city_id"] == city_id)
    with np.load(core.path(rec["prediction_keys"]["path"]), allow_pickle=False) as z:
        return rec["prediction_keys"], np.array(z["city_id"], np.int64), np.array(z["station_id"], np.int64), np.array(z["forecast_origin_us"], np.int64)

def _write_payload(evaluation, prediction, available, checkpoint):
    key_entry, city, station, origin = _keys(evaluation["target_city_id"])
    flat = prediction.reshape(-1).astype(np.float32); avail = available.reshape(-1).astype(bool)
    flat = np.where(avail, flat, np.float32(0.0)).astype(np.float32)
    if len(flat) != len(city): raise RuntimeError("prediction/key row count mismatch")
    opaque = (city.astype(np.uint64) << np.uint64(32)) | np.arange(len(city), dtype=np.uint64)
    buffer = io.BytesIO(); np.savez_compressed(buffer, city_id=city, station_id=station,
        forecast_origin_us=origin, prediction=flat.astype(np.float32), availability=avail,
        opaque_label_join_key=opaque)
    relative = f"{core.OUT}/predictions/payloads/{evaluation['evaluation_id']}.npz"
    entry = core.atomic_bytes(relative, buffer.getvalue())
    return {"evaluation_item_id": evaluation["evaluation_id"], "city": evaluation["target_city_id"],
        "method": evaluation["method"], "budget": evaluation["budget"], "seed": evaluation["seed"],
        "payload": entry, "prediction_key_binding": key_entry, "checkpoint": checkpoint,
        "implementation_binding": {"code_manifest_sha256": core.sha(core.CODE_MANIFEST),
            "baseline_source": core.entry("research/final_v2_2_r7_r1/baselines.py") if evaluation["method"] in BASELINES else None},
        "row_schema": {"key_scope": "evaluation_item metadata plus payload row", "prediction": "float32",
            "availability": "bool", "opaque_label_join_key": "uint64(city_id,key_index)"},
        "held_out_target_fields_present": False}

def predict_all(package_sha: str, device_name="cuda"):
    from . import baselines, data, validation
    validation.postrun_validate(package_sha)
    device = core.configure_torch(17, device_name)
    evaluations = core.read_json(core.EVALUATION_MANIFEST)["passes"]; jobs = core.read_json(core.JOB_MANIFEST)["fits"]
    items = []
    for evaluation in evaluations:
        panel = data.target_city(evaluation["target_city_id"], "0", evaluation=True)
        if evaluation["method"] in LEARNED:
            job = _job_for(evaluation, jobs); completion = validation.validate_one_completion(job, package_sha)
            prediction, available = _learned(evaluation, job, panel, completion, device); checkpoint = completion["output_checkpoint"]
        else:
            prediction, available = baselines.predict(evaluation["method"], panel, evaluation["budget"]); checkpoint = None
        items.append(_write_payload(evaluation, prediction, available, checkpoint))
    manifest = {"schema_version": "final_v2_2_r7_r1.predictions.1", "status": "ALL_468_PHASE_A_PREDICTIONS_CREATED",
        "package_manifest_sha256": package_sha, "evaluation_manifest_sha256": core.FROZEN[core.EVALUATION_MANIFEST],
        "items": items, "count": len(items), "canonical_order": "frozen evaluation-manifest order, then frozen city key order",
        "final_labels_joined": False, "created_utc": core.utc()}
    return core.atomic_json(core.PREDICTION_MANIFEST, manifest)

def validate_predictions(package_sha: str):
    from . import validation
    validation.postrun_validate(package_sha)
    manifest=core.read_json(core.PREDICTION_MANIFEST); expected=core.read_json(core.EVALUATION_MANIFEST)["passes"]
    manifest_fields={"schema_version","status","package_manifest_sha256","evaluation_manifest_sha256","items","count","canonical_order","final_labels_joined","created_utc"}
    if set(manifest)!=manifest_fields or manifest["schema_version"]!="final_v2_2_r7_r1.predictions.1" or manifest["status"]!="ALL_468_PHASE_A_PREDICTIONS_CREATED":
        raise RuntimeError("prediction manifest schema/state invalid")
    if manifest["package_manifest_sha256"]!=package_sha or manifest["evaluation_manifest_sha256"]!=core.FROZEN[core.EVALUATION_MANIFEST] or manifest["count"]!=468:
        raise RuntimeError("prediction manifest authority/count mismatch")
    if manifest["canonical_order"]!="frozen evaluation-manifest order, then frozen city key order" or manifest["final_labels_joined"] is not False:
        raise RuntimeError("prediction manifest order/firewall mismatch")
    items=manifest["items"]; expected_ids=[x["evaluation_id"] for x in expected]
    if len(items)!=468 or [x["evaluation_item_id"] for x in items]!=expected_ids or len(set(expected_ids))!=468:
        raise RuntimeError("missing, extra, duplicate, or noncanonical prediction item")
    payload_dir=core.path(core.OUT+"/predictions/payloads")
    actual_files={p.relative_to(core.ROOT).as_posix() for p in payload_dir.glob("*.npz")} if payload_dir.exists() else set()
    declared={x["payload"]["path"] for x in items}
    if actual_files!=declared: raise RuntimeError("undeclared, missing, or extra prediction payload file")
    jobs=core.read_json(core.JOB_MANIFEST)["fits"]
    item_fields={"evaluation_item_id","city","method","budget","seed","payload","prediction_key_binding","checkpoint","implementation_binding","row_schema","held_out_target_fields_present"}
    expected_row_schema={"key_scope":"evaluation_item metadata plus payload row","prediction":"float32","availability":"bool","opaque_label_join_key":"uint64(city_id,key_index)"}
    for item,evaluation in zip(items,expected):
        if set(item)!=item_fields or item["held_out_target_fields_present"] is not False or item["city"]!=evaluation["target_city_id"] or item["method"]!=evaluation["method"] or item["budget"]!=evaluation["budget"] or item["seed"]!=evaluation["seed"]:
            raise RuntimeError("prediction metadata mismatch")
        if item["row_schema"]!=expected_row_schema: raise RuntimeError("prediction row-schema mismatch")
        payload=item["payload"]; canonical=f"{core.OUT}/predictions/payloads/{evaluation['evaluation_id']}.npz"
        if set(payload)!={"path","bytes","sha256"} or payload["path"]!=canonical: raise RuntimeError("prediction payload path/entry mismatch")
        value=core.path(payload["path"])
        if core.sha256_path(value)!=payload["sha256"] or value.stat().st_size!=payload["bytes"]: raise RuntimeError("prediction payload changed")
        key_entry,city,station,origin=_keys(evaluation["target_city_id"])
        if item["prediction_key_binding"]!=key_entry: raise RuntimeError("prediction-key binding mismatch")
        with np.load(value,allow_pickle=False) as z:
            if set(z.files)!={"city_id","station_id","forecast_origin_us","prediction","availability","opaque_label_join_key"}: raise RuntimeError("prediction payload schema")
            if z["city_id"].dtype!=np.int64 or z["station_id"].dtype!=np.int64 or z["forecast_origin_us"].dtype!=np.int64 or z["prediction"].dtype!=np.float32 or z["availability"].dtype!=np.bool_ or z["opaque_label_join_key"].dtype!=np.uint64:
                raise RuntimeError("prediction payload dtype mismatch")
            if not np.array_equal(z["city_id"],city) or not np.array_equal(z["station_id"],station) or not np.array_equal(z["forecast_origin_us"],origin): raise RuntimeError("prediction key universe mismatch")
            expected_opaque=(city.astype(np.uint64)<<np.uint64(32))|np.arange(len(city),dtype=np.uint64)
            if not np.array_equal(z["opaque_label_join_key"],expected_opaque) or len(np.unique(z["opaque_label_join_key"]))!=len(city): raise RuntimeError("opaque join key duplicate/drift")
            prediction=np.asarray(z["prediction"]); available=np.asarray(z["availability"])
            if prediction.shape!=city.shape or available.shape!=city.shape: raise RuntimeError("prediction row count")
            if np.any(~np.isfinite(prediction[available])) or np.any(prediction[available]<0) or np.any(prediction[~available]!=0):
                raise RuntimeError("prediction value/availability canonicalization mismatch")
        expected_impl={"code_manifest_sha256":core.sha(core.CODE_MANIFEST),
            "baseline_source":core.entry("research/final_v2_2_r7_r1/baselines.py") if evaluation["method"] in BASELINES else None}
        if item["implementation_binding"]!=expected_impl: raise RuntimeError("prediction implementation binding mismatch")
        if evaluation["method"] in LEARNED:
            job=_job_for(evaluation,jobs); completion=validation.validate_one_completion(job,package_sha)
            if item["checkpoint"]!=completion["output_checkpoint"]: raise RuntimeError("learned prediction checkpoint mismatch")
        elif item["checkpoint"] is not None: raise RuntimeError("baseline unexpectedly has checkpoint")
    return {"status":"PASS_468_PREDICTIONS_COMPLETE","expected":468,"present":468,"missing":0,"extra":0,
        "duplicates":0,"held_out_target_fields":0,"canonical_order":True,"final_labels_accessed":False}