from __future__ import annotations

from typing import Any

import numpy as np
import torch

from research.data.graph_dataset import GraphCityHourLoader, GraphWindowBatch
from research.data.manifests import ProtocolArtifactRegistry
from research.graphs.geographic_graph import GeographicGraph, build_city_graph
from research.models.common import NeuralModelConfig
from research.models.model_factory import build_model
from research.training.batching import graph_batch_to_torch
from research.training.losses import raw_count_mae
from research.training.reproducibility import set_seed
from research.training.trainer import NeuralTrainer, OptimizerConfig, TrainerConfig


ENGINEERING_CITY_ID = 467
ENGINEERING_OFFSET = 168
ENGINEERING_HOURS = 32
REFERENCE_K = 8
REFERENCE_HIDDEN = 64
REFERENCE_DROPOUT = 0.10
OBSERVED_STEPS = 400
TEACHER_STUDENT_STEPS = 800
TEACHER_STUDENT_MAX_FINAL_MAE = 0.02
TEACHER_STUDENT_MIN_REDUCTION = 0.95
TRAJECTORY_ABSOLUTE_TOLERANCE = 1e-7
TRAJECTORY_CHECKPOINTS = (0, 1, 10, 25, 50, 100, 200, 400)


def fixed_real_graph_fixture(
    registry: ProtocolArtifactRegistry,
) -> tuple[GraphWindowBatch, dict[str, Any], GeographicGraph]:
    """Label-value-independent deterministic 32-hour post-week Gießen fixture."""
    loader = GraphCityHourLoader(
        registry,
        ENGINEERING_CITY_ID,
        end_utc=registry.boundaries["HD"],
        eligible_anchors_only=False,
    )
    source = loader.fetch(ENGINEERING_OFFSET, ENGINEERING_HOURS)
    if len(source.timestamps) != ENGINEERING_HOURS:
        raise RuntimeError("Fixed engineering fixture does not contain 32 hours")
    graph = build_city_graph(registry, ENGINEERING_CITY_ID, k=REFERENCE_K)
    batch = graph_batch_to_torch(source, graph.normalized_adjacency)
    return source, batch, graph


def reference_config() -> NeuralModelConfig:
    return NeuralModelConfig(
        model_type="graph_gru",
        hidden_size=REFERENCE_HIDDEN,
        dropout=REFERENCE_DROPOUT,
        output_mode="raw_count",
    )


def _masked_variance(prediction: torch.Tensor, mask: torch.Tensor) -> float:
    selected = prediction[mask]
    return float(torch.var(selected, unbiased=False)) if selected.numel() else float("nan")


def run_fixed_observed_trajectory(
    fixture_batch: dict[str, Any],
    adjacency: torch.Tensor,
    *,
    seed: int = 1703,
    steps: int = OBSERVED_STEPS,
) -> tuple[dict[str, Any], Any, NeuralTrainer]:
    """Repeated-subset optimization; observed labels are used only as training loss."""
    set_seed(seed)
    model = build_model(reference_config())
    trainer = NeuralTrainer(
        model,
        OptimizerConfig(learning_rate=1e-3),
        TrainerConfig(seed=seed, output_mode="raw_count"),
    )
    model_inputs = dict(fixture_batch["model_inputs"])
    model_inputs["adjacency"] = adjacency
    batch = {**fixture_batch, "model_inputs": model_inputs}
    model.eval()
    with torch.no_grad():
        initial_prediction = model(**model_inputs).prediction.detach().clone()
        initial_loss = float(raw_count_mae(initial_prediction, batch["target"], batch["mask"]))
    trajectory = [{"step": 0, "evaluation_loss": initial_loss}]
    maximum_gradient_sum: dict[str, float] = {}
    global_gradient_norms: list[float] = []
    finite = bool(torch.isfinite(initial_prediction).all())
    checkpoint_set = set(TRAJECTORY_CHECKPOINTS)
    for step in range(1, steps + 1):
        log = trainer.train_step(batch)
        global_gradient_norms.append(log.gradient_norm_before_clip)
        finite = finite and np.isfinite(log.loss) and np.isfinite(log.gradient_norm_before_clip)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                value = float(parameter.grad.detach().abs().sum())
                maximum_gradient_sum[name] = max(maximum_gradient_sum.get(name, 0.0), value)
        if step in checkpoint_set:
            trajectory.append({"step": step, "evaluation_loss": trainer.evaluate_loss(batch)})
    model.eval()
    with torch.no_grad():
        final_prediction = model(**model_inputs).prediction.detach().clone()
        final_loss = float(raw_count_mae(final_prediction, batch["target"], batch["mask"]))
    required = (
        "cell.weight_reset",
        "cell.weight_update",
        "cell.weight_candidate",
        "head.linear.weight",
    )
    connected = all(maximum_gradient_sum.get(name, 0.0) > 0.0 for name in required)
    finite = finite and np.isfinite(final_loss) and bool(torch.isfinite(final_prediction).all())
    result = {
        "seed": seed,
        "steps": steps,
        "initial_training_mae": initial_loss,
        "final_training_mae": final_loss,
        "percentage_reduction": 100.0 * (initial_loss - final_loss) / initial_loss,
        "prediction_variance_before": _masked_variance(initial_prediction, batch["mask"]),
        "prediction_variance_after": _masked_variance(final_prediction, batch["mask"]),
        "maximum_prediction_absolute_change": float(
            torch.max(torch.abs(final_prediction - initial_prediction))
        ),
        "valid_target_nodes": int(batch["mask"].sum()),
        "windows": int(batch["target"].shape[0]),
        "maximum_gradient_absolute_sums": maximum_gradient_sum,
        "global_gradient_norm_before_clip_min": min(global_gradient_norms),
        "global_gradient_norm_before_clip_max": max(global_gradient_norms),
        "required_parameter_groups_connected": connected,
        "finite_optimization": finite,
        "trajectory": trajectory,
        "passed": bool(
            finite
            and final_loss < initial_loss
            and connected
            and not torch.equal(initial_prediction, final_prediction)
        ),
    }
    return result, model, trainer


def compare_trajectories(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    left = [item["evaluation_loss"] for item in first["trajectory"]]
    right = [item["evaluation_loss"] for item in second["trajectory"]]
    if [item["step"] for item in first["trajectory"]] != [
        item["step"] for item in second["trajectory"]
    ]:
        raise RuntimeError("Trajectory checkpoints differ")
    maximum = max(abs(a - b) for a, b in zip(left, right))
    return {
        "absolute_tolerance": TRAJECTORY_ABSOLUTE_TOLERANCE,
        "maximum_absolute_loss_difference": maximum,
        "passed": maximum <= TRAJECTORY_ABSOLUTE_TOLERANCE,
    }


def run_teacher_student(
    fixture_batch: dict[str, Any],
    *,
    teacher_seed: int = 1701,
    student_seed: int = 1702,
    steps: int = TEACHER_STUDENT_STEPS,
) -> dict[str, Any]:
    """Recover teacher targets on real tensors/graph without using observed y."""
    set_seed(teacher_seed)
    teacher = build_model(reference_config()).eval()
    with torch.no_grad():
        teacher_target = teacher(**fixture_batch["model_inputs"]).prediction.detach().clone()
    student_batch = {**fixture_batch, "target": teacher_target}
    set_seed(student_seed)
    student = build_model(reference_config())
    trainer = NeuralTrainer(
        student,
        OptimizerConfig(learning_rate=1e-3),
        TrainerConfig(seed=student_seed, output_mode="raw_count"),
    )
    initial = trainer.evaluate_loss(student_batch)
    maximum_gradient_sum: dict[str, float] = {}
    finite = np.isfinite(initial)
    for _ in range(steps):
        log = trainer.train_step(student_batch)
        finite = finite and np.isfinite(log.loss) and np.isfinite(log.gradient_norm_before_clip)
        for name, parameter in student.named_parameters():
            if parameter.grad is not None:
                value = float(parameter.grad.detach().abs().sum())
                maximum_gradient_sum[name] = max(maximum_gradient_sum.get(name, 0.0), value)
    final = trainer.evaluate_loss(student_batch)
    reduction = (initial - final) / initial
    required = (
        "cell.weight_reset",
        "cell.weight_update",
        "cell.weight_candidate",
        "head.linear.weight",
    )
    connected = all(maximum_gradient_sum.get(name, 0.0) > 0.0 for name in required)
    tolerance_pass = final <= TEACHER_STUDENT_MAX_FINAL_MAE
    reduction_pass = reduction >= TEACHER_STUDENT_MIN_REDUCTION
    return {
        "teacher_seed": teacher_seed,
        "student_seed": student_seed,
        "steps": steps,
        "windows": int(fixture_batch["target"].shape[0]),
        "valid_target_nodes": int(fixture_batch["mask"].sum()),
        "real_observed_y_used_as_training_target": False,
        "initial_teacher_target_mae": initial,
        "final_teacher_target_mae": final,
        "percentage_reduction": 100.0 * reduction,
        "maximum_final_mae_tolerance": TEACHER_STUDENT_MAX_FINAL_MAE,
        "minimum_reduction_tolerance": TEACHER_STUDENT_MIN_REDUCTION,
        "finite_optimization": bool(finite and np.isfinite(final)),
        "required_parameter_groups_connected": connected,
        "acceptance_via_final_tolerance": tolerance_pass,
        "acceptance_via_reduction": reduction_pass,
        "passed": bool(finite and connected and (tolerance_pass or reduction_pass)),
    }


def _duplicate_summary(matrix: np.ndarray) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    _, inverse, counts = np.unique(matrix, axis=0, return_inverse=True, return_counts=True)
    duplicate_groups = counts >= 2
    return (
        {
            "rows": int(matrix.shape[0]),
            "unique_rows": int(counts.size),
            "duplicate_groups": int(duplicate_groups.sum()),
            "examples_in_duplicate_groups": int(counts[duplicate_groups].sum()),
        },
        inverse,
        counts,
    )


def observational_equivalence_audit(source: GraphWindowBatch) -> dict[str, Any]:
    batch, _, nodes, _ = source.x_hist.shape
    history = source.x_hist.transpose(0, 2, 1, 3).reshape(batch * nodes, -1)
    weekly = source.x_week.reshape(batch * nodes, -1)
    static = np.broadcast_to(source.x_static[None, :, :], (batch, nodes, 2)).reshape(
        batch * nodes, -1
    )
    calendar = np.broadcast_to(source.x_calendar[:, None, :], (batch, nodes, 6)).reshape(
        batch * nodes, -1
    )
    full = np.concatenate([history, weekly, static, calendar], axis=1)
    summaries = {}
    for name, matrix in (
        ("history_24h_value_mask", history),
        ("weekly_value_mask", weekly),
        ("static_features", static),
        ("calendar_features", calendar),
    ):
        summaries[name] = _duplicate_summary(matrix)[0]
    full_summary, inverse, counts = _duplicate_summary(full)
    target = source.y.reshape(-1)
    mask = source.m_target.reshape(-1) & np.isfinite(target)
    conflicting_groups = 0
    conflicting_examples = 0
    conflicting_pairs = 0
    for group, count in enumerate(counts):
        if count < 2:
            continue
        indices = np.flatnonzero((inverse == group) & mask)
        values = target[indices]
        if values.size >= 2 and np.unique(values).size >= 2:
            conflicting_groups += 1
            conflicting_examples += int(values.size)
            conflicting_pairs += int(
                sum(values[i] != values[j] for i in range(len(values)) for j in range(i + 1, len(values)))
            )
    full_summary.update(
        exact_duplicate_groups_with_different_observed_targets=conflicting_groups,
        observed_examples_in_conflicting_duplicate_groups=conflicting_examples,
        unequal_target_pairs_among_exact_duplicates=conflicting_pairs,
    )
    summaries["full_registered_representation"] = full_summary
    return {
        "definition": "exact float32 equality across the named registered tensor channels",
        "station_examples": batch * nodes,
        "components": summaries,
    }
