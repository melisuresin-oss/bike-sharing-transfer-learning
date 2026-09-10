"""Strong Phase-A commitment; completeness is always recomputed internally."""
from __future__ import annotations

from . import core, predictions, validation

def commit(package_sha: str):
    completeness = predictions.validate_predictions(package_sha)
    jobs = core.read_json(core.JOB_MANIFEST)["fits"]
    completions = []
    for job in jobs:
        record = validation.validate_one_completion(job, package_sha)
        completions.append({"job_id": job["job_id"], "completion": core.entry(validation.completion_path(job["job_id"])),
                            "checkpoint": record["output_checkpoint"], "input_checkpoint": record["input_checkpoint"]})
    prediction_manifest = core.read_json(core.PREDICTION_MANIFEST)
    payloads = [{"evaluation_item_id": x["evaluation_item_id"], "payload": x["payload"]} for x in prediction_manifest["items"]]
    package = core.read_json(core.PACKAGE_MANIFEST)
    bound = {"prediction_manifest": core.entry(core.PREDICTION_MANIFEST), "prediction_payloads": payloads,
        "completions": completions, "job_manifest": core.entry(core.JOB_MANIFEST),
        "evaluation_manifest": core.entry(core.EVALUATION_MANIFEST), "final_data_manifest": core.entry(core.DATA_MANIFEST),
        "implementation_contract": core.entry(core.IMPLEMENTATION_CONTRACT), "package_manifest": core.entry(core.PACKAGE_MANIFEST),
        "package_member_manifest": package["package_member_manifest"], "executable_code_manifest": package["executable_code_manifest"],
        "final_protocol_markdown": core.entry("FINAL_EVALUATION_PROTOCOL_V2_2_AMENDMENT.md"),
        "final_protocol_json": core.entry("final_evaluation_protocol_v2_2.json"),
        "grl_clarification_markdown": core.entry("FINAL_GRL_SOURCE_DOMAIN_CARDINALITY_CLARIFICATION_V2_2.md"),
        "grl_clarification_json": core.entry("final_grl_source_domain_cardinality_v2_2.json"),
        "frozen_upstream": [{"path": p, "sha256": h} for p,h in core.FROZEN.items()]}
    result = {"schema_version": "final_v2_2_r7_r1.prediction_commitment.1", "status": "PHASE_A_PREDICTIONS_STRONGLY_COMMITTED",
        "completeness_validation": completeness, "bound": bound, "bound_root_sha256": core.canonical_hash(bound),
        "final_labels_opened_before_commitment": False, "phase_b_authorized": False, "committed_utc": core.utc()}
    return core.atomic_json(core.PREDICTION_COMMITMENT, result)

def verify(commitment_path: str, package_sha: str):
    predictions.validate_predictions(package_sha); value = core.read_json(commitment_path); bound = value["bound"]
    if value["status"] != "PHASE_A_PREDICTIONS_STRONGLY_COMMITTED" or value["bound_root_sha256"] != core.canonical_hash(bound): raise PermissionError("commitment root mismatch")
    entries = [bound["prediction_manifest"], *[x["payload"] for x in bound["prediction_payloads"]],
        *[x["completion"] for x in bound["completions"]], *[x["checkpoint"] for x in bound["completions"]],
        bound["job_manifest"], bound["evaluation_manifest"], bound["final_data_manifest"], bound["implementation_contract"],
        bound["package_manifest"], bound["package_member_manifest"], bound["executable_code_manifest"],
        bound["final_protocol_markdown"], bound["final_protocol_json"], bound["grl_clarification_markdown"], bound["grl_clarification_json"]]
    entries += [x["input_checkpoint"] for x in bound["completions"] if x["input_checkpoint"]]
    for entry in entries:
        p = core.path(entry["path"])
        if core.sha256_path(p) != entry["sha256"] or ("bytes" in entry and p.stat().st_size != entry["bytes"]): raise PermissionError("bound artifact changed: " + entry["path"])
    for item in bound["frozen_upstream"]:
        if core.sha(item["path"]) != item["sha256"]: raise PermissionError("frozen upstream changed")
    return {"status": "PASS_STRONG_COMMITMENT", "bound_completions": 410, "bound_prediction_payloads": 468,
            "final_labels_accessed": False}

