"""Phase-A-only tensor loaders bound to the sealed final-data manifest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from . import core

def _verify(entry: dict[str, Any]) -> Path:
    value = core.path(entry["path"])
    if value.stat().st_size != entry["bytes"] or core.sha256_path(value) != entry["sha256"]:
        raise RuntimeError("Phase-A payload mismatch: " + entry["path"])
    return value

@dataclass
class CityTensorData:
    city_id: int
    origin_us: np.ndarray
    station_ids: np.ndarray
    x_hist: np.ndarray
    m_hist: np.ndarray
    x_week: np.ndarray
    x_static: np.ndarray
    x_calendar: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    adjacency: np.ndarray | None
    predictor_binding: dict[str, Any]
    snapshot_binding: dict[str, Any]
    graph_binding: dict[str, Any] | None

    @property
    def eligible_origins(self):
        return np.flatnonzero(self.mask.any(axis=1))

    def graph_batch(self, indices, device):
        import torch
        ix = np.asarray(indices, dtype=np.int64)
        return {"model_inputs": {"x_hist": torch.as_tensor(self.x_hist[ix], device=device),
                "m_hist": torch.as_tensor(self.m_hist[ix], device=device),
                "x_week": torch.as_tensor(self.x_week[ix], device=device),
                "x_static": torch.as_tensor(self.x_static, device=device),
                "x_calendar": torch.as_tensor(self.x_calendar[ix], device=device),
                "adjacency": torch.as_tensor(self.adjacency, device=device)},
                "target": torch.as_tensor(self.target[ix], device=device),
                "mask": torch.as_tensor(self.mask[ix], device=device)}

    def vanilla_batch(self, indices, device):
        import torch
        ix = np.asarray(indices, dtype=np.int64); hours = len(ix); nodes = len(self.station_ids)
        hist = self.x_hist[ix].transpose(0, 2, 1, 3).reshape(hours * nodes, 24, 2)
        week = self.x_week[ix].reshape(hours * nodes, 2)
        static = np.broadcast_to(self.x_static[None], (hours, nodes, 2)).reshape(hours * nodes, 2).copy()
        calendar = np.broadcast_to(self.x_calendar[ix, None], (hours, nodes, 6)).reshape(hours * nodes, 6).copy()
        return {"model_inputs": {"x_hist": torch.as_tensor(hist, device=device),
                "m_hist": torch.as_tensor(hist[..., 1] > .5, device=device),
                "x_week": torch.as_tensor(week, device=device),
                "x_static": torch.as_tensor(static, device=device),
                "x_calendar": torch.as_tensor(calendar, device=device)},
                "target": torch.as_tensor(self.target[ix].reshape(-1), device=device),
                "mask": torch.as_tensor(self.mask[ix].reshape(-1), device=device)}

def _load_predictors(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {"origin_us", "station_ids", "x_hist", "m_hist", "x_week", "x_static", "x_calendar"}
        if not required <= set(z.files) or {"count", "target", "y", "target_12h"} & set(z.files):
            raise RuntimeError("predictor archive schema violates Phase A")
        return tuple(np.array(z[k]) for k in ("origin_us", "station_ids", "x_hist", "m_hist", "x_week", "x_static", "x_calendar"))

def _labels(snapshot: Path, origins: np.ndarray, stations: np.ndarray):
    with np.load(snapshot, allow_pickle=False) as z:
        if set(z.files) != {"city_id", "station_id", "origin_us", "count", "observed", "origin_feature_sha256"}:
            raise RuntimeError("sealed fit-snapshot schema drift")
        station = np.array(z["station_id"], dtype=np.int64); origin = np.array(z["origin_us"], dtype=np.int64)
        count = np.array(z["count"], dtype=np.float32); observed = np.array(z["observed"], dtype=bool)
    oi = np.searchsorted(origins, origin); si = np.searchsorted(stations, station)
    if len(oi) and (np.any(oi >= len(origins)) or np.any(si >= len(stations)) or
                    not np.array_equal(origins[oi], origin) or not np.array_equal(stations[si], station)):
        raise RuntimeError("fit rows do not align to sealed predictors")
    target = np.full((len(origins), len(stations)), np.nan, dtype=np.float32)
    mask = np.zeros(target.shape, dtype=bool)
    valid = observed & np.isfinite(count)
    target[oi[valid], si[valid]] = count[valid]; mask[oi[valid], si[valid]] = True
    if np.any(target[mask] < 0): raise RuntimeError("negative fitting target")
    return target, mask

def manifest():
    return core.read_json(core.DATA_MANIFEST)

def _target_record(city_id: int):
    return next(x for x in manifest()["targets"] if x["city_id"] == city_id)

def target_city(city_id: int, budget: str, *, evaluation: bool = False) -> CityTensorData:
    if city_id not in core.TARGETS or budget not in core.BUDGETS: raise ValueError("unregistered target/budget")
    rec = _target_record(city_id); predictor = rec["evaluation_panel" if evaluation else "fitting_panel"]
    if evaluation:
        origin, stations, hist, mhist, week, static, calendar = _load_predictors(_verify(predictor))
        target = np.full((len(origin), len(stations)), np.nan, np.float32); mask = np.zeros(target.shape, bool)
        snap = {"path": "PHASE_B_INACCESSIBLE", "bytes": 0, "sha256": "0" * 64}
    else:
        adaptation = next(x for x in rec["adaptations"] if x["request"]["budget"] == ("zero" if budget == "0" else budget))
        origin, stations, hist, mhist, week, static, calendar = _load_predictors(_verify(predictor))
        target, mask = _labels(_verify(adaptation["data"]), origin, stations); snap = adaptation["data"]
    with np.load(_verify(rec["graph"]), allow_pickle=False) as z: adjacency = np.array(z["adjacency"], np.float32)
    return CityTensorData(city_id, origin, stations, hist.astype(np.float32), mhist.astype(bool),
        week.astype(np.float32), static.astype(np.float32), calendar.astype(np.float32), target, mask,
        adjacency, predictor, snap, rec["graph"])

def source_city(city_id: int) -> CityTensorData:
    if city_id not in core.SOURCES: raise PermissionError("final target cannot enter source refit")
    dm = manifest(); snap_rec = next(x for x in dm["source_refit_partitions"] if x["request"]["city_ids"] == [city_id])
    predictor_path = core.path(core.SOURCE_PREDICTORS[city_id]); predictor = core.entry(core.SOURCE_PREDICTORS[city_id])
    origin, stations, hist, mhist, week, static, calendar = _load_predictors(predictor_path)
    target, mask = _labels(_verify(snap_rec["data"]), origin, stations)
    graph_path = core.path(core.SOURCE_GRAPHS[city_id]); graph = core.entry(core.SOURCE_GRAPHS[city_id])
    adjacency = np.load(graph_path, allow_pickle=False).astype(np.float32)
    return CityTensorData(city_id, origin, stations, hist.astype(np.float32), mhist.astype(bool),
        week.astype(np.float32), static.astype(np.float32), calendar.astype(np.float32), target, mask,
        adjacency, predictor, snap_rec["data"], graph)

def structural_job_binding(job: dict[str, Any]) -> dict[str, Any]:
    dm = manifest(); bindings: dict[str, Any] = {"final_data_manifest_sha256": core.FROZEN[core.DATA_MANIFEST]}
    if job["category"] == "source":
        bindings["source_snapshots"] = [next(x for x in dm["source_refit_partitions"] if x["request"]["city_ids"] == [c])["data"] for c in core.SOURCES]
        bindings["source_predictors"] = [core.entry(core.SOURCE_PREDICTORS[c]) for c in core.SOURCES]
        bindings["source_graphs"] = [core.entry(core.SOURCE_GRAPHS[c]) for c in core.SOURCES]
    else:
        rec = next(x for x in dm["targets"] if x["city_id"] == job["target_city_id"])
        budget = job["budget"]
        bindings["target_predictors"] = rec["fitting_panel"]
        bindings["target_graph"] = rec["graph"] if job["architecture"].startswith("GGRU") else None
        bindings["target_snapshot"] = next(x for x in rec["adaptations"] if x["request"]["budget"] == budget)["data"]
        if job["category"] == "pooled":
            bindings["source_snapshots"] = [next(x for x in dm["source_refit_partitions"] if x["request"]["city_ids"] == [c])["data"] for c in core.SOURCES]
            bindings["source_predictors"] = [core.entry(core.SOURCE_PREDICTORS[c]) for c in core.SOURCES]
            bindings["source_graphs"] = [core.entry(core.SOURCE_GRAPHS[c]) for c in core.SOURCES]
    bindings["binding_sha256"] = core.canonical_hash(bindings)
    return bindings

