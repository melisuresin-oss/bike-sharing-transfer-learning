"""Authoritative UNIVERSITY_A40 worker/validator/freezer for Stage 2 and 2B V2.2."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import uuid
from typing import Any

if os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
    raise RuntimeError("Conflicting CUBLAS_WORKSPACE_CONFIG")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch

from research.development_data_v2_2.registry import DevelopmentRegistry
from research.evaluation.metrics import compute_metrics
from research.models.graph_gru import GraphGRU
from research.models.vanilla_gru import VanillaGRU
from research.training.reproducibility import set_seed
from research.training.trainer import NeuralTrainer, TrainerConfig
from research.v2_2.contract import Contract, DEVELOPMENT, HOUR
from research.stage2_v2_2 import core as c


PACKAGE_MANIFEST = "deployment/stage2_v2_2_a40/BUNDLE_MANIFEST.json"
PACKAGE_AUTHORIZATION = "deployment/stage2_v2_2_a40/EXECUTION_AUTHORIZATION.json"
DEFAULT_WORKERS = 4
MAX_WORKERS = 6
THREADS_PER_WORKER = 2


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _allowed_output(relative: str) -> bool:
    return relative.startswith((c.STAGE2_OUT, c.STAGE2B_OUT, c.JOINT_OUT))


def output_path(relative: str) -> Path:
    if not _allowed_output(relative):
        raise PermissionError("Stage-2/2B output isolation")
    return c.repository_path(relative)


def entry(relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": c.sha256_file(relative),
        "bytes": c.repository_path(relative).stat().st_size,
    }


def write_json_atomic(relative: str, value: Any) -> dict[str, Any]:
    destination = output_path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    require(not destination.exists(), "Append-only destination exists: " + relative)
    temporary.rename(destination)
    return entry(relative)


def write_bytes_atomic(relative: str, payload: bytes) -> dict[str, Any]:
    destination = output_path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(not destination.exists(), "Append-only destination exists: " + relative)
    temporary.rename(destination)
    return entry(relative)


def save_npz_atomic(relative: str, **arrays: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return write_bytes_atomic(relative, buffer.getvalue())


def save_checkpoint_atomic(relative: str, payload: dict[str, Any]) -> dict[str, Any]:
    destination = output_path(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".pending-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    require(not destination.exists(), "Append-only checkpoint exists: " + relative)
    temporary.rename(destination)
    return entry(relative)


def execution_lock(name: str):
    import contextlib
    import msvcrt

    @contextlib.contextmanager
    def locked():
        lock_path = output_path(c.JOINT_OUT + "locks/" + name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            handle.close()
            raise RuntimeError("Execution lock is already held: " + name) from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            handle.close()
    return locked()


def _job_map_path(stage: str) -> str:
    if stage == "stage2":
        return c.STAGE2_JOB_MAP
    if stage == "stage2b":
        return c.STAGE2B_JOB_MAP
    raise ValueError("Stage must be stage2 or stage2b")


def _contract_path(stage: str) -> str:
    return c.STAGE2_CONTRACT if stage == "stage2" else c.STAGE2B_CONTRACT


def _output_root(stage: str) -> str:
    return c.STAGE2_OUT if stage == "stage2" else c.STAGE2B_OUT


def verify_package(expected_manifest_sha256: str) -> dict[str, Any]:
    require(c.sha256_file(PACKAGE_MANIFEST) == expected_manifest_sha256, "External package-manifest binding failed")
    manifest = c.read_json(PACKAGE_MANIFEST)
    require(manifest.get("protocol_version") == "2.2", "Wrong package protocol")
    require(manifest.get("execution_environment") == c.ENVIRONMENT, "Wrong package environment")
    require(manifest.get("stage2_jobs") == 144 and manifest.get("stage2b_jobs") == 48, "Package workload changed")
    for relative, digest in manifest["files"].items():
        require(c.sha256_file(relative) == digest, "Package member changed: " + relative)
    forbidden = (
        "final_labels", "sealed_final", "final_adaptation", "final_features",
        "stage3_", "stage4_", "/checkpoints/", "/predictions/",
    )
    for relative in manifest["files"]:
        lowered = relative.lower()
        require(not any(token in lowered for token in forbidden), "Forbidden package member: " + relative)
        require(not lowered.endswith(".pt"), "Fitted checkpoint in execution package")
    c.validate_authorities()
    for stage in ("stage2", "stage2b"):
        job_map = c.read_json(_job_map_path(stage))
        contract = c.read_json(_contract_path(stage))
        c.validate_job_map(stage, job_map)
        require(job_map["scientific_contract"]["sha256"] == c.sha256_file(_contract_path(stage)), "Job-map contract binding")
        require(contract["status"] == "FROZEN_VERIFIED" and contract["contradictions"] == [], "Scientific contract not frozen")
        require(contract["stage1_decision_sha256"] == c.STAGE1_DECISION_SHA256, "Stage-1 binding changed")
        require(contract["target_transform"]["output_mode"] == c.OUTPUT_MODE, "LOG1P binding changed")
    authorization = c.read_json(PACKAGE_AUTHORIZATION)
    require(authorization["stage2_authorized_after_strict_preflight_pass"] is True, "Stage 2 not authorized")
    require(authorization["stage2b_authorized_after_strict_preflight_pass"] is True, "Stage 2B not authorized")
    require(authorization["stage3_authorized"] is False and authorization["stage4_authorized"] is False, "Downstream stage authorized")
    require(authorization["final_target_label_access_authorized"] is False, "Final labels authorized")
    return manifest


def configure_runtime() -> None:
    torch.set_num_threads(THREADS_PER_WORKER)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False


def runtime() -> dict[str, Any]:
    configure_runtime()
    expected_python = Path(os.environ["USERPROFILE"]) / "bike_env/Scripts/python.exe"
    require(Path(sys.executable).resolve() == expected_python.resolve(), "Use USERPROFILE\\bike_env\\Scripts\\python.exe")
    require(sys.version_info[:2] == (3, 10), "Validated TS9 Python 3.10 required")
    require(str(torch.__version__) == "2.0.1+cu118" and np.__version__ == "1.26.4", "Validated TS9 torch/numpy versions required")
    for module in (torch, np):
        require(expected_python.parent.parent.resolve() in Path(module.__file__).resolve().parents, "Dependencies must load from bike_env")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Exactly one CUDA device required")
    require(torch.cuda.get_device_name(0) == "NVIDIA A40", "CUDA device must be NVIDIA A40")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(len(smi.splitlines()) == 1 and smi.split(",")[0].strip() == "NVIDIA A40", "nvidia-smi A40 identity")
    settings = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    require(settings == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "deterministic_algorithms": True,
        "cuda_matmul_tf32": False, "cudnn_benchmark": False,
        "cudnn_deterministic": True, "cudnn_tf32": False,
    }, "Deterministic CUDA contract")
    properties = torch.cuda.get_device_properties(0)
    return {
        "execution_environment": c.ENVIRONMENT,
        "python_executable": str(expected_python.resolve()),
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": "NVIDIA A40",
        "gpu_total_memory_bytes": int(properties.total_memory),
        "logical_cpu_count": os.cpu_count(),
        "nvidia_smi": smi,
        "settings": settings,
    }


def _host_total_memory_bytes() -> int | None:
    if os.name != "nt":
        return None
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)


def resource_check(requested_workers: int, run: dict[str, Any]) -> dict[str, Any]:
    base_cache = c.read_json(c.BASE_CACHE_MANIFEST)
    cache_bytes = sum(
        c.repository_path(array["path"]).stat().st_size
        for city in base_cache["cities"].values() for array in city["arrays"].values()
    )
    host_ram = _host_total_memory_bytes()
    logical_cpu = os.cpu_count()
    six_worker_projection = cache_bytes * 6
    six_safe = bool(
        run["gpu_total_memory_bytes"] >= 48 * 1024**3
        and host_ram is not None and host_ram >= 24 * 1024**3
        and logical_cpu is not None and logical_cpu >= 12
        and six_worker_projection <= 6 * 1024**3
    )
    if requested_workers == 6:
        require(six_safe, "Six-worker resource gate did not pass; use the proven four-worker setting")
    return {
        "requested_workers": requested_workers,
        "worker_count_is_scientific_hyperparameter": False,
        "threads_per_worker": THREADS_PER_WORKER,
        "base_cache_payload_bytes_per_worker_upper_bound": cache_bytes,
        "six_worker_cache_payload_bytes_upper_bound": six_worker_projection,
        "gpu_total_memory_bytes": run["gpu_total_memory_bytes"],
        "host_total_memory_bytes": host_ram,
        "logical_cpu_count": logical_cpu,
        "six_worker_gate_passed": six_safe,
        "prepackage_selected_workers": DEFAULT_WORKERS,
        "selection_reason": "Four workers are already proven; the package was prepared without authoritative TS9 host-RAM/CPU evidence for six.",
    }


def preflight(manifest_sha256: str, workers: int) -> dict[str, Any]:
    require(1 <= workers <= MAX_WORKERS, "Operational worker count outside 1..6")
    verify_package(manifest_sha256)
    run = runtime()
    resources = resource_check(workers, run)
    device = torch.device("cuda")
    synthetic = {
        "x_hist": torch.zeros(2, 24, 3, 2, device=device),
        "m_hist": torch.zeros(2, 24, 3, dtype=torch.bool, device=device),
        "x_week": torch.zeros(2, 3, 2, device=device),
        "x_static": torch.zeros(3, 2, device=device),
        "x_calendar": torch.zeros(2, 6, device=device),
        "adjacency": torch.eye(3, device=device),
    }
    parameter_counts: dict[str, int] = {}
    for configuration in c.graph_configurations():
        set_seed(17)
        model = GraphGRU(c.model_config(configuration)).to(device).eval()
        with torch.no_grad():
            first = model(**synthetic).count_prediction
            second = model(**synthetic).count_prediction
        require(torch.equal(first, second) and torch.isfinite(first).all(), "Graph-GRU synthetic deterministic forward")
        parameter_counts[configuration["config_id"]] = sum(parameter.numel() for parameter in model.parameters())
    vanilla_input = {
        "x_hist": synthetic["x_hist"][:, :, 0, :],
        "m_hist": synthetic["m_hist"][:, :, 0],
        "x_week": synthetic["x_week"][:, 0, :],
        "x_static": synthetic["x_static"][:2],
        "x_calendar": synthetic["x_calendar"],
    }
    for configuration in c.vanilla_configurations():
        set_seed(17)
        model = VanillaGRU(c.model_config(configuration)).to(device).eval()
        with torch.no_grad():
            first = model(**vanilla_input).count_prediction
            second = model(**vanilla_input).count_prediction
        require(torch.equal(first, second) and torch.isfinite(first).all(), "Vanilla-GRU synthetic deterministic forward")
        parameter_counts[configuration["config_id"]] = sum(parameter.numel() for parameter in model.parameters())
    require(set(parameter_counts.values()) == {3403, 12939}, "Frozen architecture parameter counts changed")
    result = {
        "status": "PASS",
        "scope": "STRICT_ZERO_TRAINING_STAGE2_STAGE2B_V2_2_A40_PREFLIGHT",
        "bundle_manifest_sha256": manifest_sha256,
        "runtime": run,
        "resource_check": resources,
        "stage1_decision_sha256": c.STAGE1_DECISION_SHA256,
        "stage1_selected_candidate": "LOG1P",
        "stage1_selected_mode": c.OUTPUT_MODE,
        "development_data_manifest_sha256": c.DATA_MANIFEST_SHA256,
        "implementation_manifest_sha256": c.IMPLEMENTATION_MANIFEST_SHA256,
        "stage2_jobs": 144,
        "stage2b_jobs": 48,
        "total_jobs": 192,
        "parameter_counts": parameter_counts,
        "scientific_training_steps": 0,
        "scientific_evaluations": 0,
        "synthetic_forward_only": True,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
        "created_utc": utc(),
    }
    saved = write_json_atomic(c.JOINT_OUT + "preflight/" + uuid.uuid4().hex + ".json", result)
    print(json.dumps({"status": "PASS", "preflight": saved}, indent=2), flush=True)
    return {"record": result, "artifact": saved}


def check_preflight(relative: str, manifest_sha256: str, workers: int, actual_runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    require(relative.startswith(c.JOINT_OUT + "preflight/"), "Preflight namespace")
    record = c.read_json(relative)
    require(record["status"] == "PASS" and record["bundle_manifest_sha256"] == manifest_sha256, "Strict preflight PASS required")
    require(record["scientific_training_steps"] == 0 and record["scientific_evaluations"] == 0, "Preflight scientific work detected")
    require(record["resource_check"]["requested_workers"] == workers, "Worker-count provenance drift")
    if actual_runtime is not None:
        require(record["runtime"] == actual_runtime, "Runtime drift after preflight")
    return record


class City:
    def __init__(self, city_entry: dict[str, Any], graph_k: int | None, device: torch.device):
        self.entry = city_entry
        for name, array in city_entry["arrays"].items():
            require(c.sha256_file(array["path"]) == array["sha256"], "Cache array changed")
            setattr(self, name, np.load(c.repository_path(array["path"]), mmap_mode="r", allow_pickle=False))
        self.static = torch.as_tensor(np.array(self.x_static, dtype=np.float32), device=device)
        self.graph = None
        if graph_k is not None:
            relative = f"tmp/stage2_graph_grid_cache_v2_1/k_{graph_k:02d}/city_{city_entry['city_id']}/adjacency.npy"
            self.graph = torch.as_tensor(np.load(c.repository_path(relative), allow_pickle=False), dtype=torch.float32, device=device)
        self.device = device

    def graph_inputs(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        require(self.graph is not None, "Graph input requested for Vanilla-GRU")
        return {
            "x_hist": torch.as_tensor(np.array(self.x_hist[indices], dtype=np.float32), device=self.device),
            "m_hist": torch.as_tensor(np.array(self.m_hist[indices], dtype=bool), device=self.device),
            "x_week": torch.as_tensor(np.array(self.x_week[indices], dtype=np.float32), device=self.device),
            "x_static": self.static,
            "x_calendar": torch.as_tensor(np.array(self.x_calendar[indices], dtype=np.float32), device=self.device),
            "adjacency": self.graph,
        }

    def graph_batch(self, indices: np.ndarray) -> dict[str, Any]:
        return {
            "model_inputs": self.graph_inputs(indices),
            "target": torch.as_tensor(np.array(self.fit_count[indices], dtype=np.float32), device=self.device),
            "mask": torch.as_tensor(np.array(self.fit_mask[indices], dtype=bool), device=self.device),
        }

    def vanilla_inputs(self, indices: np.ndarray) -> dict[str, torch.Tensor]:
        hours = len(indices)
        nodes = len(self.station_ids)
        x_hist = np.asarray(self.x_hist[indices], dtype=np.float32).transpose(0, 2, 1, 3).reshape(hours * nodes, 24, 2)
        x_week = np.asarray(self.x_week[indices], dtype=np.float32).reshape(hours * nodes, 2)
        static = np.broadcast_to(np.asarray(self.x_static, dtype=np.float32)[None], (hours, nodes, 2)).reshape(hours * nodes, 2).copy()
        calendar = np.broadcast_to(np.asarray(self.x_calendar[indices], dtype=np.float32)[:, None], (hours, nodes, 6)).reshape(hours * nodes, 6).copy()
        x_hist_tensor = torch.as_tensor(x_hist, device=self.device)
        return {
            "x_hist": x_hist_tensor,
            "m_hist": x_hist_tensor[..., 1] > 0.5,
            "x_week": torch.as_tensor(x_week, device=self.device),
            "x_static": torch.as_tensor(static, device=self.device),
            "x_calendar": torch.as_tensor(calendar, device=self.device),
        }

    def vanilla_batch(self, indices: np.ndarray) -> dict[str, Any]:
        hours = len(indices)
        nodes = len(self.station_ids)
        return {
            "model_inputs": self.vanilla_inputs(indices),
            "target": torch.as_tensor(np.asarray(self.fit_count[indices], dtype=np.float32).reshape(hours * nodes), device=self.device),
            "mask": torch.as_tensor(np.asarray(self.fit_mask[indices], dtype=bool).reshape(hours * nodes), device=self.device),
        }


def _build_model(job: dict[str, Any], device: torch.device) -> torch.nn.Module:
    configuration = job["configuration"]
    model = GraphGRU(c.model_config(configuration)) if job["stage"] == 2 else VanillaGRU(c.model_config(configuration))
    return model.to(device)


def expected_metadata(job: dict[str, Any], manifest_sha256: str, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_version": "2.2",
        "stage": job["stage"],
        "job_id": job["job_id"],
        "configuration": job["configuration"],
        "pseudo_target_city_id": job["pseudo_target_city_id"],
        "source_city_ids": job["source_city_ids"],
        "seed": job["seed"],
        "source_city_schedule_seed": job["source_city_schedule_seed"],
        "source_anchor_sampling_seed": job["source_anchor_sampling_seed"],
        "source_updates_completed": c.SOURCE_UPDATES,
        "optimizer": asdict(c.optimizer_config()),
        "batch_size_city_hours": c.BATCH_SIZE,
        "gradient_clip_global_norm": c.GRADIENT_CLIP,
        "output_mode": c.OUTPUT_MODE,
        "source_snapshot": job["source_snapshot"],
        "job_map_sha256": c.sha256_file(_job_map_path("stage2" if job["stage"] == 2 else "stage2b")),
        "scientific_contract_sha256": c.sha256_file(_contract_path("stage2" if job["stage"] == 2 else "stage2b")),
        "bundle_manifest_sha256": manifest_sha256,
        "development_data_manifest_sha256": c.DATA_MANIFEST_SHA256,
        "implementation_manifest_sha256": c.IMPLEMENTATION_MANIFEST_SHA256,
        "runtime": run,
        "v2_1_fitted_checkpoint_loaded": False,
        "final_target_labels_accessed": False,
    }


def _checkpoint_payload(model: torch.nn.Module, trainer: NeuralTrainer, metadata: dict[str, Any],
                        anchor_rng: np.random.Generator, city_updates: dict[int, int]) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "model_state": model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "anchor_rng": anchor_rng.bit_generator.state,
        "city_updates": city_updates,
    }


@torch.no_grad()
def _predict(model: torch.nn.Module, city: City, indices: np.ndarray, stage: str) -> np.ndarray:
    model.eval()
    if stage == "stage2":
        prediction = model(**city.graph_inputs(indices)).count_prediction
        return prediction.detach().cpu().numpy()
    hours, nodes = len(indices), len(city.station_ids)
    prediction = model(**city.vanilla_inputs(indices)).count_prediction
    return prediction.detach().cpu().numpy().reshape(hours, nodes)


def _metrics_for_prediction(prediction_entry: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    require(c.sha256_file(prediction_entry["path"]) == prediction_entry["sha256"], "Prediction artifact changed")
    with np.load(c.repository_path(prediction_entry["path"]), allow_pickle=False) as archive:
        require(set(archive.files) == {"origin_us", "station_ids", "prediction_count"}, "Prediction artifact contains labels or wrong schema")
        origins = archive["origin_us"]
        station_ids = archive["station_ids"]
        predictions = archive["prediction_count"]
    require(predictions.shape == (len(origins), len(station_ids)), "Prediction grid shape")
    require(np.isfinite(predictions).all() and np.all(predictions >= 0), "Invalid count-space predictions")
    require(c.canonical_hash({
        "city_id": job["pseudo_target_city_id"],
        "origin_us": list(map(int, origins)),
        "station_ids": list(map(int, station_ids)),
    }) == job["prediction_grid_key_sha256"], "Precommitted prediction grid changed")
    registry = DevelopmentRegistry(c.ROOT, c.DATA_MANIFEST, c.DATA_MANIFEST_SHA256)
    labels = registry.retrospective_labels(job["pseudo_target_city_id"])
    start = int(np.searchsorted(labels["origin_us"], origins[0]))
    stop = start + len(origins)
    require(np.array_equal(labels["origin_us"][start:stop], origins), "Development-label origins do not match committed predictions")
    require(np.array_equal(labels["station_ids"], station_ids), "Development-label roster does not match committed predictions")
    mask = labels["observed"][start:stop]
    target = labels["count"][start:stop]
    stations = np.broadcast_to(station_ids[None, :], mask.shape)
    result = compute_metrics(target[mask], predictions[mask], stations[mask]).as_dict()
    require(result["mae"] is not None and result["n"] == int(mask.sum()), "Empty retrospective development evaluation")
    actual = target[mask].astype(np.float64)
    predicted = predictions[mask].astype(np.float64)
    result.update({
        "prediction_mean": float(predicted.mean()),
        "prediction_std": float(predicted.std()),
        "target_mean": float(actual.mean()),
        "target_std": float(actual.std()),
        "observed_key_sha256": c.canonical_hash({
            "city_id": job["pseudo_target_city_id"],
            "origins": list(map(int, np.broadcast_to(origins[:, None], mask.shape)[mask])),
            "stations": list(map(int, stations[mask])),
        }),
        "target_value_sha256": hashlib.sha256(np.ascontiguousarray(actual).tobytes()).hexdigest(),
        "evaluation_label_role": "DEVELOPMENT_RETROSPECTIVE_LABELS",
        "evaluation_window": "[HD,HF)",
        "final_target_labels_accessed": False,
    })
    return result


def evaluate_and_commit(model: torch.nn.Module, city: City, job: dict[str, Any], attempt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = Contract()
    first = int((contract.boundaries["HD"] - contract.boundaries["H0"]) // HOUR)
    last = int((contract.boundaries["HF"] - contract.boundaries["H0"]) // HOUR)
    indices = np.arange(first, last, dtype=np.int64)
    parts = []
    for start in range(0, len(indices), 64):
        parts.append(_predict(model, city, indices[start:start + 64], "stage2" if job["stage"] == 2 else "stage2b"))
    predictions = np.concatenate(parts).astype(np.float32)
    prediction = save_npz_atomic(
        attempt + "prediction.npz",
        origin_us=np.array(city.origin_us[indices]),
        station_ids=np.array(city.station_ids),
        prediction_count=predictions,
    )
    # This commitment is durable before the retrospective development-label reader runs.
    prediction_manifest = write_json_atomic(attempt + "prediction_manifest.json", {
        "status": "PREDICTIONS_COMMITTED_BEFORE_DEVELOPMENT_LABEL_ACCESS",
        "job_id": job["job_id"],
        "prediction_grid_key_sha256": job["prediction_grid_key_sha256"],
        "all_prediction_keys": int(predictions.size),
        "prediction": prediction,
        "contains_target_values": False,
        "retrospective_development_labels_opened_before_commitment": False,
        "final_target_labels_accessed": False,
        "created_utc": utc(),
    })
    metrics = _metrics_for_prediction(prediction, job)
    return {"prediction": prediction, "prediction_manifest": prediction_manifest}, metrics


def _load_checkpoint(relative: str) -> dict[str, Any]:
    return torch.load(c.repository_path(relative), map_location="cpu", weights_only=True)


def validate_completed(job: dict[str, Any], manifest_sha256: str, *, recompute_metrics: bool = True) -> dict[str, Any]:
    stage = "stage2" if job["stage"] == 2 else "stage2b"
    relative = _output_root(stage) + "completed/" + job["job_id"] + ".json"
    record = c.read_json(relative)
    require(record["status"] == "COMPLETED_VALIDATED_JOB" and record["job"] == job, "Completion identity")
    require(record["bundle_manifest_sha256"] == manifest_sha256, "Completion package binding")
    require(record["source_updates_completed"] == c.SOURCE_UPDATES and record["target_adaptation_fits"] == 0, "Scientific budget drift")
    require(record["final_target_labels_accessed"] is False, "Final-label firewall failed")
    check_preflight(record["preflight"]["path"], manifest_sha256, record["operational_worker_count"])
    checkpoint_entry = record["checkpoint"]
    require(checkpoint_entry["path"].startswith(_output_root(stage) + "attempts/" + job["job_id"] + "/"), "Checkpoint namespace")
    require(c.sha256_file(checkpoint_entry["path"]) == checkpoint_entry["sha256"], "Checkpoint hash")
    payload = _load_checkpoint(checkpoint_entry["path"])
    require(payload["metadata"] == record["checkpoint_metadata"], "Checkpoint metadata")
    require(payload["metadata"]["bundle_manifest_sha256"] == manifest_sha256, "Checkpoint package binding")
    expected_counts = {city: 0 for city in job["source_city_ids"]}
    for city in c.city_sequence(tuple(job["source_city_ids"]), job["source_city_schedule_seed"]):
        expected_counts[city] += 1
    require(payload["city_updates"] == expected_counts and record["source_city_updates"] == {str(k): v for k, v in expected_counts.items()}, "Exact source exposure")
    optimizer_state = payload["optimizer_state"]
    require(optimizer_state["state"], "Final optimizer state is empty")
    require(all(int(state["step"].item()) == c.SOURCE_UPDATES for state in optimizer_state["state"].values()), "Optimizer did not reach final step")
    group = optimizer_state["param_groups"]
    require(len(group) == 1 and group[0]["lr"] == 1e-3 and group[0]["betas"] == (0.9, 0.999)
            and group[0]["eps"] == 1e-8 and group[0]["weight_decay"] == 1e-4, "Frozen AdamW configuration")
    require(record["checkpoint_roundtrip_max_abs_difference"] == 0.0 and record["optimizer_initial_state_empty"], "Checkpoint/optimizer integrity")
    prediction = record["prediction"]
    require(c.sha256_file(prediction["path"]) == prediction["sha256"], "Prediction hash")
    prediction_manifest = record["prediction_manifest"]
    require(c.sha256_file(prediction_manifest["path"]) == prediction_manifest["sha256"], "Prediction manifest hash")
    commitment = c.read_json(prediction_manifest["path"])
    require(commitment["status"] == "PREDICTIONS_COMMITTED_BEFORE_DEVELOPMENT_LABEL_ACCESS"
            and commitment["contains_target_values"] is False, "Prediction commitment firewall")
    if recompute_metrics:
        require(_metrics_for_prediction(prediction, job) == record["metrics"], "Persisted pseudo-target metrics do not recompute")
    return record


def run_job(stage: str, job_id: str, manifest_sha256: str, preflight_path: str, workers: int) -> None:
    verify_package(manifest_sha256)
    run = runtime()
    check_preflight(preflight_path, manifest_sha256, workers, run)
    job_map = c.read_json(_job_map_path(stage))
    job = next((candidate for candidate in job_map["jobs"] if candidate["job_id"] == job_id), None)
    require(job is not None, "Unregistered immutable job")
    completed_path = _output_root(stage) + "completed/" + job_id + ".json"
    if c.repository_path(completed_path).exists():
        validate_completed(job, manifest_sha256)
        print("SKIP validated completed immutable job " + job_id, flush=True)
        return
    attempt = _output_root(stage) + "attempts/" + job_id + "/" + uuid.uuid4().hex + "/"
    write_json_atomic(attempt + "started.json", {
        "status": "STARTED_FROM_UPDATE_ZERO",
        "job": job,
        "bundle_manifest_sha256": manifest_sha256,
        "preflight": entry(preflight_path),
        "runtime": run,
        "optimizer_updates_completed_at_start": 0,
        "partial_checkpoint_resumed": False,
        "v2_1_checkpoint_loaded": False,
        "final_target_labels_accessed": False,
        "created_utc": utc(),
    })
    device = torch.device("cuda")
    set_seed(job["seed"])
    model = _build_model(job, device)
    trainer = NeuralTrainer(
        model, c.optimizer_config(),
        TrainerConfig(seed=job["seed"], output_mode=c.OUTPUT_MODE,
                      gradient_clip_global_norm=c.GRADIENT_CLIP, checkpoint_mode="final"),
    )
    require(not trainer.optimizer.state and trainer.step == 0, "Fresh optimizer required")
    graph_k = job["configuration"]["graph_k"]
    base_cache = c.read_json(c.BASE_CACHE_MANIFEST)
    cities = {
        city_id: City(base_cache["cities"][str(city_id)], graph_k, device)
        for city_id in set(job["source_city_ids"] + [job["pseudo_target_city_id"]])
    }
    sequence = c.city_sequence(tuple(job["source_city_ids"]), job["source_city_schedule_seed"])
    anchor_rng = np.random.default_rng(job["source_anchor_sampling_seed"])
    updates = {city: 0 for city in job["source_city_ids"]}
    valid_targets = {city: 0 for city in job["source_city_ids"]}
    diagnostics: list[dict[str, Any]] = []
    started = time.monotonic()
    print(job_id + " START from optimizer update 0", flush=True)
    for step, city_id in enumerate(sequence, 1):
        city = cities[city_id]
        indices = anchor_rng.choice(city.source_anchors, size=c.BATCH_SIZE, replace=True).astype(np.int64)
        batch = city.graph_batch(indices) if stage == "stage2" else city.vanilla_batch(indices)
        log = trainer.train_step(batch)
        updates[city_id] += 1
        valid_targets[city_id] += log.valid_targets
        if step == 1 or step % 1000 == 0 or step == c.SOURCE_UPDATES:
            diagnostics.append(asdict(log))
        if step % 2000 == 0 and step < c.SOURCE_UPDATES:
            partial_metadata = {**expected_metadata(job, manifest_sha256, run), "source_updates_completed": step,
                                "completed_fit": False, "evaluation_completed": False}
            save_checkpoint_atomic(
                attempt + f"partial_{step:05d}.pt",
                _checkpoint_payload(model, trainer, partial_metadata, anchor_rng, dict(updates)),
            )
        if len(trainer.logs) >= 1000:
            trainer.logs.clear()
    require(trainer.step == c.SOURCE_UPDATES and sum(updates.values()) == c.SOURCE_UPDATES, "Fixed source budget")
    require(max(updates.values()) - min(updates.values()) <= 1, "Equal-city source exposure")
    metadata = expected_metadata(job, manifest_sha256, run)
    checkpoint = save_checkpoint_atomic(
        attempt + "final.pt", _checkpoint_payload(model, trainer, metadata, anchor_rng, dict(updates))
    )
    target_city = cities[job["pseudo_target_city_id"]]
    probe_index = np.asarray([int((Contract().boundaries["HD"] - Contract().boundaries["H0"]) // HOUR)], dtype=np.int64)
    before = _predict(model, target_city, probe_index, stage)
    saved = _load_checkpoint(checkpoint["path"])
    restored = _build_model(job, device)
    restored.load_state_dict(saved["model_state"], strict=True)
    after = _predict(restored, target_city, probe_index, stage)
    require(np.array_equal(before, after), "Checkpoint roundtrip changed predictions")
    artifacts, metrics = evaluate_and_commit(restored, target_city, job, attempt)
    record = {
        "status": "COMPLETED_VALIDATED_JOB",
        "protocol_version": "2.2",
        "stage": job["stage"],
        "job": job,
        "job_id": job_id,
        "config_id": job["config_id"],
        "pseudo_target_city_id": job["pseudo_target_city_id"],
        "seed": job["seed"],
        "bundle_manifest_sha256": manifest_sha256,
        "scientific_contract_sha256": c.sha256_file(_contract_path(stage)),
        "job_map_sha256": c.sha256_file(_job_map_path(stage)),
        "preflight": entry(preflight_path),
        "runtime": run,
        "operational_worker_count": workers,
        "worker_count_is_scientific_hyperparameter": False,
        "source_updates_completed": c.SOURCE_UPDATES,
        "source_city_updates": {str(key): value for key, value in updates.items()},
        "source_valid_target_exposures": {str(key): value for key, value in valid_targets.items()},
        "target_adaptation_fits": 0,
        "optimizer_initial_state_empty": True,
        "checkpoint_roundtrip_max_abs_difference": 0.0,
        "checkpoint_metadata": metadata,
        "checkpoint": checkpoint,
        "prediction": artifacts["prediction"],
        "prediction_manifest": artifacts["prediction_manifest"],
        "metrics": metrics,
        "training_diagnostics": diagnostics,
        "elapsed_seconds": time.monotonic() - started,
        "v2_1_fitted_checkpoint_loaded": False,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
        "completed_utc": utc(),
    }
    write_json_atomic(completed_path, record)
    validate_completed(job, manifest_sha256)
    print(job_id + " COMPLETE AND VALIDATED", flush=True)


def validate_stage(stage: str, manifest_sha256: str) -> dict[str, Any]:
    verify_package(manifest_sha256)
    job_map = c.read_json(_job_map_path(stage))
    expected = 144 if stage == "stage2" else 48
    completed = []
    entries = []
    for job in job_map["jobs"]:
        record = validate_completed(job, manifest_sha256)
        completed.append(record)
        entries.append(entry(_output_root(stage) + "completed/" + job["job_id"] + ".json"))
    actual = {
        path.name for path in output_path(_output_root(stage) + "completed").glob("*.json")
    }
    required = {job["job_id"] + ".json" for job in job_map["jobs"]}
    require(actual == required, "Missing, duplicate, or unregistered completed job records")
    require(len(completed) == expected, "Validated completion count")
    result = {
        "status": "PASS",
        "purpose": "READ_ONLY_POSTRUN_VALIDATION",
        "protocol_version": "2.2",
        "stage": 2 if stage == "stage2" else "2B",
        "execution_environment": c.ENVIRONMENT,
        "validated_jobs": expected,
        "source_fits": expected,
        "zero_shot_evaluations": expected,
        "adaptation_fits": 0,
        "training_steps_performed_by_validator": 0,
        "optimizer_steps_performed_by_validator": 0,
        "model_forward_calls_by_validator": 0,
        "bundle_manifest_sha256": manifest_sha256,
        "scientific_contract_sha256": c.sha256_file(_contract_path(stage)),
        "job_map_sha256": c.sha256_file(_job_map_path(stage)),
        "execution_records": entries,
        "v2_1_fitted_checkpoints_reused": False,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def record_validation(stage: str, manifest_sha256: str) -> dict[str, Any]:
    relative = _output_root(stage) + "POSTRUN_VALIDATION.json"
    result = validate_stage(stage, manifest_sha256)
    if c.repository_path(relative).exists():
        existing = c.read_json(relative)
        comparable = dict(existing)
        comparable.pop("recorded_utc", None)
        require(comparable == result, "Existing validator report differs")
        return entry(relative)
    return write_json_atomic(relative, {**result, "recorded_utc": utc()})


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    require(rows, "Cannot serialize an empty table")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                         if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return buffer.getvalue().encode("utf-8")


def freeze(stage: str, manifest_sha256: str) -> dict[str, Any]:
    decision_relative = _output_root(stage) + ("stage2_decision_manifest.json" if stage == "stage2" else "stage2b_decision_manifest.json")
    if c.repository_path(decision_relative).exists():
        decision = c.read_json(decision_relative)
        require(decision["status"] == "FROZEN_VALIDATED" and decision["bundle_manifest_sha256"] == manifest_sha256, "Existing decision invalid")
        print(json.dumps({"status": "ALREADY_FROZEN_VALIDATED", "decision": entry(decision_relative)}, indent=2))
        return decision
    validation = record_validation(stage, manifest_sha256)
    job_map = c.read_json(_job_map_path(stage))
    completed = [c.read_json(_output_root(stage) + "completed/" + job["job_id"] + ".json") for job in job_map["jobs"]]
    result = c.aggregate(stage, completed)
    individual_rows = [{
        "job_id": row["job_id"], "config_id": row["config_id"],
        "pseudo_target_city_id": row["pseudo_target_city_id"], "seed": row["seed"],
        **row["metrics"], "checkpoint_sha256": row["checkpoint"]["sha256"],
        "prediction_sha256": row["prediction"]["sha256"],
    } for row in completed]
    individual = write_bytes_atomic(_output_root(stage) + "individual_runs.csv", _csv_bytes(individual_rows))
    folds = write_bytes_atomic(_output_root(stage) + "fold_summary.csv", _csv_bytes(result["fold_rows"]))
    configurations = write_bytes_atomic(_output_root(stage) + "configuration_summary.csv", _csv_bytes(result["configuration_rows"]))
    winner = result["winner"]
    runner = result["runner_up"]
    report_text = f"""# {'Stage 2 Graph-GRU' if stage == 'stage2' else 'Stage 2B Vanilla-GRU'} V2.2 frozen decision

Status: **FROZEN_VALIDATED**  
Execution environment: **UNIVERSITY_A40**  
Source fits/evaluations: **{len(completed)}/{len(completed)}**  
Target adaptations: **0**

The selected configuration is **{winner['config_id']}**, with equal-fold count-space MAE
`{winner['equal_fold_count_space_mae']:.15f}`. The runner-up is **{runner['config_id']}**
at `{runner['equal_fold_count_space_mae']:.15f}`; the absolute margin is
`{result['margin_to_runner_up']:.15f}`. The frozen four-decimal tie rule was
{'invoked' if result['tie_break_invoked'] else 'not invoked'}.

Winner fold heterogeneity: population SD `{winner['fold_mae_population_sd']:.15f}`,
minimum `{winner['fold_mae_min']:.15f}`, maximum `{winner['fold_mae_max']:.15f}`, and
range `{winner['fold_mae_range']:.15f}`. The V2.2 winner
{'matches' if result['winner_matches_historical_v2_1'] else 'does not match'} the historical
V2.1 winner `{result['historical_v2_1_winner']}`. Historical results were not used in selection.

Selection used only the registered V2.2 pseudo-target development results and the frozen
selection hierarchy. No final-target labels were accessed. Stage 3 and Stage 4 were not started.
"""
    report = write_bytes_atomic(_output_root(stage) + "SELECTION_REPORT.md", report_text.encode("utf-8"))
    decision = {
        "schema_version": "1.0",
        "protocol_version": "2.2",
        "status": "FROZEN_VALIDATED",
        "stage": 2 if stage == "stage2" else "2B",
        "execution_environment": c.ENVIRONMENT,
        "selected_configuration": winner,
        "aggregate_count_space_mae": winner["equal_fold_count_space_mae"],
        "runner_up": runner,
        "margin_to_runner_up": result["margin_to_runner_up"],
        "tie_break_invoked": result["tie_break_invoked"],
        "selection_rule": result["selection_rule"],
        "per_pseudo_target_seed_averaged_mae": [row for row in result["fold_rows"] if row["config_id"] == winner["config_id"]],
        "fold_heterogeneity": {
            "population_sd": winner["fold_mae_population_sd"],
            "minimum": winner["fold_mae_min"],
            "maximum": winner["fold_mae_max"],
            "range": winner["fold_mae_range"],
        },
        "historical_v2_1_winner": result["historical_v2_1_winner"],
        "winner_matches_historical_v2_1": result["winner_matches_historical_v2_1"],
        "historical_v2_1_results_used_in_selection": False,
        "source_fits": len(completed),
        "zero_shot_evaluations": len(completed),
        "adaptation_fits": 0,
        "stage1_decision_sha256": c.STAGE1_DECISION_SHA256,
        "bundle_manifest_sha256": manifest_sha256,
        "scientific_contract": entry(_contract_path(stage)),
        "job_map": entry(_job_map_path(stage)),
        "postrun_validation": validation,
        "individual_runs": individual,
        "fold_summary": folds,
        "configuration_summary": configurations,
        "selection_report": report,
        "local_cpu_results_used": False,
        "v2_1_fitted_checkpoints_reused": False,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
        "frozen_utc": utc(),
    }
    saved = write_json_atomic(decision_relative, decision)
    print(json.dumps({"status": "FROZEN_VALIDATED", "decision": saved,
                      "selected_configuration": winner["config_id"],
                      "aggregate_mae": winner["equal_fold_count_space_mae"]}, indent=2), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2/Stage-2B V2.2 A40 execution")
    parser.add_argument("mode", choices=("package-check", "preflight", "job", "validate", "freeze"))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--stage", choices=("stage2", "stage2b"))
    parser.add_argument("--job-id")
    parser.add_argument("--preflight")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.mode == "package-check":
        verify_package(args.manifest_sha256)
        print(json.dumps({"status": "PASS", "scope": "PACKAGE_ONLY_NO_CUDA_CLAIM",
                          "stage2_jobs": 144, "stage2b_jobs": 48,
                          "scientific_training_steps": 0}, indent=2))
    elif args.mode == "preflight":
        preflight(args.manifest_sha256, args.workers)
    elif args.mode == "job":
        require(args.stage and args.job_id and args.preflight, "Job mode requires stage, job ID and preflight")
        with execution_lock(args.stage + "-" + args.job_id):
            run_job(args.stage, args.job_id, args.manifest_sha256, args.preflight, args.workers)
    elif args.mode == "validate":
        require(args.stage is not None, "Validate mode requires stage")
        if args.record:
            saved = record_validation(args.stage, args.manifest_sha256)
            print(json.dumps({"status": "PASS", "validator_report": saved}, indent=2))
        else:
            validate_stage(args.stage, args.manifest_sha256)
    elif args.mode == "freeze":
        require(args.stage is not None, "Freeze mode requires stage")
        freeze(args.stage, args.manifest_sha256)


if __name__ == "__main__":
    main()
