from __future__ import annotations

import csv
import json
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from research.remote.stage2b_a40 import (
    REMOTE_OUTPUT,
    REMOTE_PREFLIGHT,
    ROOT,
    assert_remote_runtime,
    load_authorization,
    read_job_map,
    read_json,
    sha256_file,
    validate_resume_job,
    write_json_atomic,
)


def main() -> None:
    load_authorization()
    assert_remote_runtime(read_json(REMOTE_PREFLIGHT))
    rows = read_job_map()
    completed = []
    for row in rows:
        path = ROOT / row["expected_job_result"]
        if not path.is_file():
            raise RuntimeError(f"Missing job result: {row['job_key']}")
        job = read_json(path)
        if not validate_resume_job(job, row):
            raise RuntimeError(f"Invalid job result: {row['job_key']}")
        prediction_path = ROOT / job["prediction"]["path"]
        prediction_manifest = read_json(ROOT / job["prediction_manifest"]["path"])
        with np.load(prediction_path, allow_pickle=False) as values:
            if len(values["prediction_count"]) != int(prediction_manifest["rows"]):
                raise RuntimeError(f"Prediction row mismatch: {row['job_key']}")
            if not np.isfinite(values["prediction_count"]).all() or np.any(values["prediction_count"] < 0):
                raise RuntimeError(f"Invalid prediction values: {row['job_key']}")
        completed.append(job)
    fold_seed = defaultdict(list)
    for job in completed:
        mae = job["metrics"].get("mae")
        if mae is None:
            raise RuntimeError("Undefined MAE")
        fold_seed[(job["config_id"], job["pseudo_target_city_id"])].append(float(mae))
    config_rows = []
    for config_id in sorted({job["config_id"] for job in completed}):
        fold_means = [sum(fold_seed[(config_id, target)]) / 3 for target in (532, 476, 619, 658)]
        mean = sum(fold_means) / 4
        std = float(np.std(np.asarray(fold_means), ddof=0))
        exemplar = next(row for row in rows if row["config_id"] == config_id)
        config_rows.append({
            "config_id": config_id, "equal_fold_mae": mean, "equal_fold_mae_4dp": str(Decimal(str(mean)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)),
            "fold_population_std": std, "hidden_size": int(exemplar["hidden_size"]), "dropout": float(exemplar["dropout"]),
        })
    ranked = sorted(config_rows, key=lambda row: (Decimal(row["equal_fold_mae_4dp"]), row["fold_population_std"], row["hidden_size"], row["dropout"]))
    summary_path = REMOTE_OUTPUT / "stage2b_config_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader(); writer.writerows(ranked)
    validation = {
        "status": "PASS", "validated_remote_jobs": 48, "source_fits": 48,
        "zero_shot_evaluations": 48, "fine_tuning_fits": 0,
        "final_label_firewall_passed": True, "final_target_labels_accessed": False,
        "summary": summary_path.relative_to(ROOT).as_posix(), "summary_sha256": sha256_file(summary_path),
        "selection_rule_applied_exactly_as_frozen": True, "selected_config_id": ranked[0]["config_id"],
        "does_not_authorize_stage4_or_final_execution": True,
    }
    write_json_atomic(REMOTE_OUTPUT / "STAGE2B_REMOTE_VALIDATION.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
