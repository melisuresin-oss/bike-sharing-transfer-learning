"""Build the append-only Stage-1 V2.2 A40 throughput-R1 bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
BASE = "deployment/stage1_v2_2_a40/"
ORIGINAL_MANIFEST = BASE + "BUNDLE_MANIFEST.json"
ORIGINAL_BUNDLE = BASE + "stage1_v2_2_UNIVERSITY_A40.zip"
MANIFEST = BASE + "BUNDLE_MANIFEST_THROUGHPUT_R1.json"
COMMANDS = BASE + "COMMANDS_THROUGHPUT_R1.txt"
BUNDLE = BASE + "stage1_v2_2_UNIVERSITY_A40_THROUGHPUT_R1.zip"
DELIVERY = BASE + "DELIVERY_MANIFEST_THROUGHPUT_R1.json"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(relative: str, text: str) -> None:
    destination = ROOT / relative
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise RuntimeError("Append-only throughput artifact differs: " + relative)
    else:
        destination.write_text(text, encoding="utf-8")


original = read(ORIGINAL_MANIFEST)
original_sha = sha(ORIGINAL_MANIFEST)
if original_sha != "357f93afb5279099af00334b339860b557531fd2284b5e05bbedef9aa380dbe9":
    raise RuntimeError("Original sealed A40 bundle manifest changed")
if sha(ORIGINAL_BUNDLE) != "7d37884cc4401a505315c50b94895b9a74503064139ba88db5238bc151ff6c12":
    raise RuntimeError("Original A40 bundle changed")

files = set(original["files"])
files.add(ORIGINAL_MANIFEST)
files.update({
    "research/stage1_v2_2/a40_parallel.py",
    BASE + "run_a40_throughput_r1.ps1",
    BASE + "EXECUTION_AUTHORIZATION_THROUGHPUT_R1.json",
    BASE + "THROUGHPUT_AUDIT_R1.json",
    BASE + "THROUGHPUT_AUDIT_R1.md",
    BASE + "test_throughput_r1.py",
})
for relative in files:
    if not (ROOT / relative).is_file():
        raise FileNotFoundError(relative)
    lowered = relative.lower()
    if any(token in lowered for token in ("final_label", "final_adaptation", "sealed_final", "__pycache__")):
        raise RuntimeError("Forbidden bundle path: " + relative)
    if relative.endswith(".pt"):
        raise RuntimeError("Checkpoint forbidden from throughput bundle: " + relative)

manifest = {
    "schema_version": "1.0",
    "protocol_version": "2.2",
    "stage": 1,
    "revision": "THROUGHPUT_R1",
    "execution_environment": "UNIVERSITY_A40",
    "original_bundle_manifest_sha256": original_sha,
    "original_bundle_sha256": sha(ORIGINAL_BUNDLE),
    "job_map_sha256": "4709c222b799a9a2d4ef209ce80ae5ee859b5a818db5c1cb97df2954ffce3937",
    "scientific_manifest_sha256": "25f1959ccaacf49f3aa50641427739d9c291843270b7d8961c5843cb5465024b",
    "operational_worker_policy": {
        "default_workers": 4,
        "maximum_workers": 4,
        "threads_per_worker": 2,
        "one_fresh_python_process_per_immutable_fit": True,
        "scientific_hyperparameter": False,
    },
    "files": {relative: sha(relative) for relative in sorted(files)},
    "file_count": len(files),
    "payload_bytes": sum((ROOT / relative).stat().st_size for relative in files),
    "scientific_worker": {
        "entrypoint": "research.stage1_v2_2.a40 job",
        "sha256": sha("research/stage1_v2_2/a40.py"),
        "changed_from_original_bundle": False,
    },
    "scope": "operational detached concurrency, worker logging, atomic status and runtime provenance only",
    "scientific_parameters_changed": False,
    "scientific_training_performed_by_builder": False,
    "final_target_labels_accessed": False,
}
manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
write_once(MANIFEST, manifest_text)
manifest_sha = sha(MANIFEST)

commands = f'''# Run from the fresh THROUGHPUT_R1 extraction root on TS9.
$py="$env:USERPROFILE\\bike_env\\Scripts\\python.exe"
$manifestSha="{manifest_sha}"

# Strict A40 and throughput preflight; no scientific training.
& .\\deployment\\stage1_v2_2_a40\\run_a40_throughput_r1.ps1 -Mode preflight -ManifestSha256 $manifestSha -Workers 4

# Detached authoritative launch with four operational workers.
& .\\deployment\\stage1_v2_2_a40\\run_a40_throughput_r1.ps1 -Mode launch -ManifestSha256 $manifestSha -Workers 4

# Monitor completed immutable fits.
$done=@(Get-ChildItem .\\research\\results\\stage1_scale_loss_v2_2_a40\\completed -File -Filter '*.json' -ErrorAction SilentlyContinue)
[pscustomobject]@{{Completed=$done.Count; Source=@($done | Where-Object Name -Like '*_source.json').Count; Adaptation=@($done | Where-Object Name -Like '*_7d.json').Count; Remaining=48-$done.Count}}

# Monitor coordinator and isolated worker processes.
Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match 'research\\.stage1_v2_2\\.(a40_parallel|a40)' }} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine

# Monitor A40 utilization; Ctrl+C stops monitoring only.
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 2

# Read-only validator after all 48 completed records exist.
& .\\deployment\\stage1_v2_2_a40\\run_a40_throughput_r1.ps1 -Mode validate -ManifestSha256 $manifestSha -Workers 4

# Output: .\\research\\results\\stage1_scale_loss_v2_2_a40\\
# Success: stage1_decision_manifest.json status FROZEN_VALIDATED; then STOP.
'''
write_once(COMMANDS, commands)

if "--manifest-only" in sys.argv:
    print(json.dumps({"manifest_sha256": manifest_sha, "file_count": len(files), "payload_bytes": manifest["payload_bytes"]}))
    raise SystemExit(0)

archive = ROOT / BUNDLE
if archive.exists():
    raise FileExistsError(BUNDLE)
members = files | {MANIFEST, COMMANDS}
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
    for relative in sorted(members):
        handle.write(ROOT / relative, relative)

delivery = {
    "schema_version": "1.0",
    "revision": "THROUGHPUT_R1",
    "bundle": {"path": BUNDLE, "sha256": sha(BUNDLE), "bytes": archive.stat().st_size},
    "bundle_manifest": {"path": MANIFEST, "sha256": manifest_sha},
    "commands": {"path": COMMANDS, "sha256": sha(COMMANDS)},
    "original_bundle_preserved": {"path": ORIGINAL_BUNDLE, "sha256": sha(ORIGINAL_BUNDLE)},
    "immutable_job_map_sha256": manifest["job_map_sha256"],
    "zip_member_count": len(members),
    "scientific_training_performed_by_builder": False,
    "actual_A40_preflight_status": "REQUIRES_EXECUTION_ON_TS9",
}
write_once(DELIVERY, json.dumps(delivery, indent=2, allow_nan=False) + "\n")
print(json.dumps(delivery, indent=2))
