"""Final state machine and safe preflight. Scientific execution requires a separate seal."""
from __future__ import annotations
import argparse, json, os, platform, sys, uuid, zipfile
from pathlib import Path
from . import core, firewall, jobs

FINAL_DATA_BINDING = "processed/protocol_v2_2/final/FINAL_DATA_BINDING.json"
PREDICTION_COMMITMENT = core.OUT + "/prediction_commitment.json"

def _read_external_binding():
    p=core.repo_path(FINAL_DATA_BINDING)
    if not p.exists(): return None
    value=json.loads(p.read_text(encoding="utf-8"))
    if value.get("protocol_version")!="2.2" or value.get("contains_v2_1_artifacts") is not False:
        raise RuntimeError("invalid V2.2 final-data binding")
    return {"path":FINAL_DATA_BINDING,"sha256":core.sha256_path(p),"bytes":p.stat().st_size}

def package_check(manifest_sha):
    core.verify_authority()
    m=core.read_json(core.PACKAGE+"/BUNDLE_MANIFEST.json")
    if core.sha(core.PACKAGE+"/BUNDLE_MANIFEST.json")!=manifest_sha:raise RuntimeError("package manifest hash mismatch")
    for e in m["files"]:
        if core.sha(e["path"])!=e["sha256"]:raise RuntimeError("package member hash mismatch: "+e["path"])
    jm=core.read_json(m["job_manifest"]["path"]);em=core.read_json(m["evaluation_manifest"]["path"])
    jobs.validate_neural_jobs(jm["fits"])
    if len(em["passes"])!=468 or sum(len(x["reuse_labels"]) for x in em["passes"])!=660:raise RuntimeError("evaluation accounting drift")
    return {"status":"PASS","package_members":len(m["files"]),"neural_fits":410,"evaluation_passes":468,
            "reporting_cells":660,"final_target_labels_accessed":False,"final_experiment_started":False,
            "optimizer_updates":0,"model_forward_calls":0}

def preflight(manifest_sha):
    result=package_check(manifest_sha)
    binding=_read_external_binding()
    return {**result,"status":"PASS_PHASE_A_NO_SCIENTIFIC_EXECUTION","runtime":{"python":sys.version,"platform":platform.platform()},
            "determinism":core.deterministic_runtime(),"final_data_binding":binding,
            "phase_b_authorized":False,"prediction_generation_authorized":False}

def dry_run(manifest_sha):
    result=preflight(manifest_sha)
    return {**result,"status":"PASS_DRY_RUN","optimizer_updates":0,"model_forward_calls":0,
            "final_target_labels_accessed":False,"final_experiment_started":False,"files_written":[]}

def commit_predictions(prediction_manifest_path, manifest_sha):
    """Phase A: hash a complete prediction manifest without opening labels."""
    package_check(manifest_sha)
    p=core.repo_path(prediction_manifest_path)
    firewall.reject_labelish_path(prediction_manifest_path)
    obj=json.loads(p.read_text(encoding="utf-8"))
    if obj.get("status")!="ALL_468_PREDICTION_STATISTIC_INPUTS_COMPLETE" or obj.get("final_labels_joined") is not False:
        raise RuntimeError("incomplete or label-contaminated prediction manifest")
    payload={"status":"PREDICTIONS_CRYPTOGRAPHICALLY_COMMITTED","package_manifest_sha256":manifest_sha,
             "prediction_manifest":{"path":prediction_manifest_path,"sha256":core.sha256_path(p),"bytes":p.stat().st_size},
             "final_labels_opened_before_commitment":False,"committed_utc":core.utc(),
             "next":"obtain separate FINAL_LABEL_SCORING_AUTHORIZED capability bound to this file hash"}
    return core.atomic_write_json(PREDICTION_COMMITMENT,payload)

def require_scientific_authorization(path):
    p=Path(path);v=json.loads(p.read_text(encoding="utf-8"))
    if v.get("status")!="FINAL_V2_2_A40_EXECUTION_AUTHORIZED" or v.get("job_manifest_sha256")!=core.sha(core.SEAL+"/final_job_manifest.json"):
        raise PermissionError("separate scientific execution authorization absent or mismatched")
    return v

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="command",required=True)
    for name in ("package-check","preflight","dry-run"):
        p=sp.add_parser(name);p.add_argument("--manifest-sha256",required=True)
    p=sp.add_parser("commit-predictions");p.add_argument("--manifest-sha256",required=True);p.add_argument("--prediction-manifest",required=True)
    args=ap.parse_args()
    if args.command=="package-check":out=package_check(args.manifest_sha256)
    elif args.command=="preflight":out=preflight(args.manifest_sha256)
    elif args.command=="dry-run":out=dry_run(args.manifest_sha256)
    else:out=commit_predictions(args.prediction_manifest,args.manifest_sha256)
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()
