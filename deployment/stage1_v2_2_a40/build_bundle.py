"""Build an explicit allowlist bundle; never enumerate label storage or old results."""
from pathlib import Path
import hashlib
import json
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE="deployment/stage1_v2_2_a40/"
OLD="research/results/stage1_scale_loss_v2_2/"
def read(p):return json.loads((ROOT/p).read_text(encoding="utf-8"))
def sha(p):
    h=hashlib.sha256()
    with (ROOT/p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def once(p,v):
    data=json.dumps(v,indent=2,allow_nan=False)+"\n"
    dest=ROOT/p
    if dest.exists():
        if dest.read_text(encoding="utf-8")!=data:raise FileExistsError(p)
    else:dest.write_text(data,encoding="utf-8")

jm=read(OLD+"job_map.json");design=read(OLD+"scientific_manifest.json")
abort=OLD+"local_cpu_runtime_migration_abort.json"
authorization={
    "protocol_version":"2.2","stage":1,"execution_environment":"UNIVERSITY_A40",
    "authorization_source":"User explicitly authorizes the complete A40 Stage-1 execution after strict A40 preflight PASS, restarting all scientific fits; stop after validated Stage-1 decision.",
    "stage1_execution_authorized_after_preflight_pass":True,
    "job_map_sha256":sha(OLD+"job_map.json"),"scientific_manifest_sha256":sha(OLD+"scientific_manifest.json"),
    "selection_rule_sha256":sha(OLD+"selection_rule.json"),
    "source_fits":24,"adaptation_fits":24,"evaluation_records":48,
    "source_initialization":"registered seed 17,29,43; optimizer update 0",
    "adaptation_initialization":"completed authoritative A40 source checkpoint; frozen derived fine-tune seed; reset optimizer update 0",
    "local_cpu_results_authoritative":False,"local_cpu_results_selection_eligible":False,
    "local_cpu_checkpoint_reuse_allowed":False,"partial_checkpoint_resume_allowed":False,
    "local_abort_provenance":{"path":abort,"sha256":sha(abort),"role":"external append-only provenance reference; not an execution input"},
    "runtime_only_supersession":{
        "original_scientific_manifest_unchanged":True,
        "overridden_fields":["runtime","precision device CPU wording only","ambiguities_resolved_from_authorities.device","code_sha256 execution adapter extension only"],
        "effective_precision":"float32 CUDA; no AMP, no TF32, no compilation or architecture rewrite",
        "effective_runner":"research/stage1_v2_2/a40.py; imports unchanged scientific configuration, model, trainer, RNG formula, and selection",
        "effective_scheduling":"one independent process per immutable source or adaptation fit; completed jobs verified before skip; partial attempt retained and restarted at optimizer step 0",
        "unchanged":"all scientific candidates, folds, seeds and seed derivation, architecture, optimizer settings, update and batch budgets, frozen V2.2 fitting cache, labels, fitting windows, evaluation keys, selection rule"
    },
    "required_runtime":{"python":"3.10","torch":"2.0.1+cu118","numpy":"1.26.4","device":"NVIDIA A40",
        "python_executable":"%USERPROFILE%\\bike_env\\Scripts\\python.exe",
        "CUBLAS_WORKSPACE_CONFIG":":4096:8","torch_deterministic_algorithms":True,
        "cuda_matmul_tf32":False,"cudnn_benchmark":False,"cudnn_deterministic":True,"cudnn_tf32":False},
    "validated_TS9_contract_reference":{"path":"research/remote/stage2b_a40.py","sha256":sha("research/remote/stage2b_a40.py")},
    "authoritative_output_directory":"research/results/stage1_scale_loss_v2_2_a40/",
    "required_finish_order":["48 immutable completed jobs","read-only postrun validator PASS","frozen Stage-1 selection rule","append-only validated decision","STOP"],
    "stage2_authorized":False,"stage2b_authorized":False,"final_target_label_access_authorized":False
}
once(BASE+"EXECUTION_AUTHORIZATION.json",authorization)

files=set(design["code_sha256"])|set(design["design_source_sha256"])
files.update({OLD+"job_map.json",OLD+"scientific_manifest.json",OLD+"selection_rule.json",
              "research/stage1_v2_2/a40.py",BASE+"EXECUTION_AUTHORIZATION.json",BASE+"run_a40.ps1",BASE+"README.md",
              "processed/protocol_v2_2/PROTOCOL_DATASET_MANIFEST.json",
              "processed/protocol_v2_2/development_labels/retrospective_label_manifest.json",
              "processed/protocol_v2_2/fit_snapshots/fit_snapshot_manifest.json",
              "research/results/causal_history_implementation_v2_2/implementation_manifest.json",
              "research/results/causal_history_audit_v2_2/development_verification_report.json",
              "research/results/causal_history_audit_v2_2/development_data_verification_seal.json",
              "research/results/causal_history_audit_v2_2/graph_static_reconciliation.json",
              "research/results/causal_history_spec_v2_2/v2_2_specification.json",
              "research/results/causal_history_spec_v2_2/causal_history_spec_seal.json",
              "research/results/causal_history_spec_v2_2/fixed_cohort_static_manifest.json",
              "RESEARCH_PROTOCOL_V2_2_AMENDMENT.md",
              "research/development_data_v2_2/__init__.py","research/development_data_v2_2/registry.py",
              "research/models/__init__.py","research/models/vanilla_gru.py",
              "research/training/__init__.py","research/evaluation/__init__.py"})
files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"research/v2_2").glob("*.py"))
cachepath="tmp/stage1_scale_loss_cache_v2_2/cache_manifest.json"
files.add(cachepath)
cache=read(cachepath)
for city in cache["cities"].values():files.update(a["path"] for a in city["arrays"].values())
for t in jm["tasks"]:
    files.add(t["source_snapshot"]["path"]);files.add(t["adaptation_snapshot"]["path"])
labels=read("processed/protocol_v2_2/development_labels/retrospective_label_manifest.json")
files.update(x["path"] for x in labels["labels"] if x["city_id"] in (532,476,619,658))
for p in files:
    assert not any(s in p.lower() for s in ("final_label","final_adaptation","sealed_final","__pycache__")),p
    assert not p.endswith(".pt"),p
    assert (ROOT/p).is_file(),p
manifest={"schema_version":"1.0","protocol_version":"2.2","stage":1,"execution_environment":"UNIVERSITY_A40",
          "job_map_sha256":sha(OLD+"job_map.json"),"scientific_manifest_sha256":sha(OLD+"scientific_manifest.json"),
          "files":{p:sha(p) for p in sorted(files)},"file_count":len(files),
          "payload_bytes":sum((ROOT/p).stat().st_size for p in files),
          "input_strategy":"Byte-identical frozen execution cache, immutable fitting seals and original artifact references; no panel rebuild or duplicate predictor/fit NPZ payloads",
          "excluded_payloads":["CPU execution/checkpoints/predictions/metrics","V2.1 predictors and results","unrelated historical outputs","original predictor and source/adaptation fit NPZ duplicated by frozen cache","unused adaptation budgets","final-target labels/features/adaptation"],
          "external_environment_dependencies":authorization["required_runtime"]}
once(BASE+"BUNDLE_MANIFEST.json",manifest)
manifest_sha=sha(BASE+"BUNDLE_MANIFEST.json")
commands=f'''# Run from the new extraction root, on TS9.
$py="$env:USERPROFILE\\bike_env\\Scripts\\python.exe"
$manifestSha="{manifest_sha}"

# Strict A40 preflight: no scientific training.
& .\\deployment\\stage1_v2_2_a40\\run_a40.ps1 -Mode preflight -ManifestSha256 $manifestSha

# Authorized detached launch. Re-running safely skips validated completed jobs.
& .\\deployment\\stage1_v2_2_a40\\run_a40.ps1 -Mode launch -ManifestSha256 $manifestSha

# After completion, read-only verification (also automatically run before freeze).
& .\\deployment\\stage1_v2_2_a40\\run_a40.ps1 -Mode validate -ManifestSha256 $manifestSha

# Output: .\\research\\results\\stage1_scale_loss_v2_2_a40\\
# Success: stage1_decision_manifest.json, status FROZEN_VALIDATED; then STOP.
'''
cmdpath=ROOT/BASE/"COMMANDS.txt"
if cmdpath.exists():assert cmdpath.read_text(encoding="utf-8")==commands
else:cmdpath.write_text(commands,encoding="utf-8")
if "--manifest-only" in sys.argv:
    print(json.dumps({"manifest_sha256":manifest_sha,"file_count":len(files),"payload_bytes":manifest["payload_bytes"]}));sys.exit(0)
bundle=BASE+"stage1_v2_2_UNIVERSITY_A40.zip"
if (ROOT/bundle).exists():raise FileExistsError(bundle)
with zipfile.ZipFile(ROOT/bundle,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(files|{BASE+"BUNDLE_MANIFEST.json",BASE+"COMMANDS.txt"}):
        z.write(ROOT/p,p)
result={"bundle":{"path":bundle,"sha256":sha(bundle),"bytes":(ROOT/bundle).stat().st_size},
        "bundle_manifest":{"path":BASE+"BUNDLE_MANIFEST.json","sha256":manifest_sha},
        "immutable_job_map":{"path":OLD+"job_map.json","sha256":sha(OLD+"job_map.json")},
        "local_abort_provenance":{"path":abort,"sha256":sha(abort)},
        "zip_member_count":len(files)+2,"scientific_training_performed_by_builder":False,
        "actual_A40_preflight_status":"REQUIRES_EXECUTION_ON_TS9"}
once(BASE+"DELIVERY_MANIFEST.json",result)
print(json.dumps(result,indent=2))
