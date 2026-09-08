"""Build the tiny append-only operational R2 controller-interface overlay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "deployment/stage2_v2_2_a40/"
MANIFEST = BASE + "BUNDLE_MANIFEST_OPERATIONAL_R2.json"
COMMANDS = BASE + "COMMANDS_OPERATIONAL_R2.txt"
BUNDLE = BASE + "stage2_stage2b_v2_2_OPERATIONAL_R2.zip"
DELIVERY = BASE + "DELIVERY_MANIFEST_OPERATIONAL_R2.json"


def sha(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(relative: str, text: str) -> None:
    path = ROOT / relative
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("Append-only R2 artifact differs: " + relative)
    else:
        path.write_text(text, encoding="utf-8")


def main() -> None:
    files = {
        "research/stage2_v2_2/a40_r2.py": "complete delegated controller interface plus R1 validation overrides",
        "research/stage2_v2_2/controller_r2.py": "R2 worker commands and retained-four dispatch gate",
        BASE + "EXECUTION_AUTHORIZATION_OPERATIONAL_R2.json": "operational-only R2 authorization",
        BASE + "OPERATIONAL_R2_INCIDENT_PROVENANCE.json": "append-only second incident provenance",
        BASE + "README_OPERATIONAL_R2.md": "R2 interface and deployment explanation",
        BASE + "run_stage2_stage2b_operational_r2.ps1": "R2 strict preflight and detached launcher",
        BASE + "test_operational_r2.py": "zero-science actual-controller startup integration tests",
        BASE + "build_operational_r2_overlay.py": "explicit tiny overlay allowlist provenance",
    }
    required_existing = {
        "deployment/stage2_v2_2_a40/BUNDLE_MANIFEST_OPERATIONAL_R1.json": "2b4114de5c6d85a3217f0c4c38dbf227790f4dbd3866667e02e5b6328cf41d75",
        "research/stage2_v2_2/a40_r1.py": "3527fc15d69d0c7e6210822d4026b13ca4cd47883c2e88819fd7e5e99d6241d3",
        "research/stage2_v2_2/controller_r1.py": "eea360276f1d0faec0eac85389219543e787d0e6d4977406bf7040120ed6f2fe",
        "research/stage2_v2_2/a40.py": "847cdeb2524d88ddd2b8cd52f080d04b27782247ae15a5e55e164c2e70906559",
        "research/stage2_v2_2/controller.py": "d7f88397e8cc2b046de950c2cde9a9aaa4d7f8d68494c26e0d8745af544c0064",
        "research/results/stage2_graphgru_selection_v2_2/scientific_contract.json": "55ebfa06ca49d77e5ae0b7b2e531d6bb14c26a2b93f478da7754c75fb1aae52f",
        "research/results/stage2b_vanilla_fairness_v2_2/scientific_contract.json": "2eb1a2a14dd1412f1286248c547814a2c7e04fad94edaace2084c7d78be0cad3",
        "research/results/stage2_graphgru_selection_v2_2/job_map.json": "fd34f71b46eb6f9cfc9adb5ebfd44001a2e26df049b9bb197aff1d272e7a074a",
        "research/results/stage2b_vanilla_fairness_v2_2/job_map.json": "0cf72a4f66797bcc5625bd438046562c5fd444677676971e2697e1f9a3d0f23d",
    }
    for relative in set(files) | set(required_existing):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    for relative, digest in required_existing.items():
        if sha(relative) != digest:
            raise RuntimeError("Immutable R2 dependency changed: " + relative)
    manifest = {
        "schema_version": "1.0",
        "protocol_version": "2.2",
        "status": "OPERATIONAL_R2_CONTROLLER_INTERFACE_PATCH",
        "execution_environment": "UNIVERSITY_A40",
        "r1_overlay_manifest_sha256": required_existing[
            "deployment/stage2_v2_2_a40/BUNDLE_MANIFEST_OPERATIONAL_R1.json"
        ],
        "files": {relative: sha(relative) for relative in sorted(files)},
        "file_reasons": {relative: files[relative] for relative in sorted(files)},
        "required_existing_files": required_existing,
        "file_count": len(files),
        "payload_bytes": sum((ROOT / relative).stat().st_size for relative in files),
        "existing_base_or_r1_files_in_overlay": 0,
        "scientific_changes": [],
        "controller_interface_strategy": "explicit required exports plus __getattr__ delegation to R1 then base",
        "corrected_worker_module": "research.stage2_v2_2.a40_r2",
        "operational_workers": 4,
        "retained_jobs": 4,
        "remaining_jobs": 188,
        "scientific_training_performed_by_builder": False,
        "scientific_evaluation_performed_by_builder": False,
        "final_target_labels_accessed": False,
        "stage3_started": False,
        "stage4_started": False,
    }
    write_once(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    manifest_sha = sha(MANIFEST)
    commands = f'''# Run on TS9 from the existing V2.2 execution root.
$runRoot='C:\\Users\\go54ray\\projects\\stage2_stage2b_v2_2_a40_run'
$manifestSha='{manifest_sha}'
Set-Location -LiteralPath $runRoot

& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_operational_r2.ps1 -Mode package-check -ManifestSha256 $manifestSha -Workers 4
& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_operational_r2.ps1 -Mode preflight -ManifestSha256 $manifestSha -Workers 4
& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_operational_r2.ps1 -Mode validate-existing -ManifestSha256 $manifestSha -Workers 4
& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_operational_r2.ps1 -Mode launch -ManifestSha256 $manifestSha -Workers 4

# Acceptance A/B: controller alive and CUDA workers present.
Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match 'research.stage2_v2_2.(controller_r2|a40_r2 job)' }} | Select-Object ProcessId,CommandLine
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Acceptance C: total completion count must rise above four.
$s2=(Get-ChildItem .\\research\\results\\stage2_graphgru_selection_v2_2_a40\\completed -Filter *.json -ErrorAction SilentlyContinue).Count
$s2b=(Get-ChildItem .\\research\\results\\stage2b_vanilla_fairness_v2_2_a40\\completed -Filter *.json -ErrorAction SilentlyContinue).Count
[pscustomobject]@{{Stage2=$s2;Stage2B=$s2b;Total=$s2+$s2b;AboveRetainedFour=(($s2+$s2b)-gt 4)}}

# Acceptance D: at least one new successful exit.
Get-ChildItem .\\research\\results\\stage2_stage2b_v2_2_a40\\controller_runs -Recurse -Filter *.exit.json |
  Sort-Object LastWriteTime -Descending | ForEach-Object {{ Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json }} |
  Where-Object {{ $_.status -eq 'EXITED_ZERO_WITH_COMPLETION' -and $_.exit_code -eq 0 -and $_.completed_record_present }} |
  Select-Object -First 5 job_id,status,exit_code,completed_record_present

# Acceptance E: latest run has more launch records than retained-job skips after a success.
Get-ChildItem .\\research\\results\\stage2_stage2b_v2_2_a40\\controller_runs -Recurse -Filter *.launch.json |
  Sort-Object LastWriteTime -Descending | Select-Object -First 12 FullName,LastWriteTime
'''
    write_once(COMMANDS, commands)
    if "--manifest-only" in sys.argv:
        print(json.dumps({"manifest_sha256": manifest_sha, "files": len(files),
                          "payload_bytes": manifest["payload_bytes"]}))
        return
    if (ROOT / BUNDLE).exists():
        raise FileExistsError("Append-only R2 bundle exists: " + BUNDLE)
    members = set(files) | {MANIFEST, COMMANDS}
    with zipfile.ZipFile(ROOT / BUNDLE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(members):
            archive.write(ROOT / relative, relative)
    delivery = {
        "status": "LOCAL_R2_ISOLATED_VERIFICATION_PENDING",
        "bundle": {"path": BUNDLE, "sha256": sha(BUNDLE), "bytes": (ROOT / BUNDLE).stat().st_size},
        "r2_manifest": {"path": MANIFEST, "sha256": manifest_sha},
        "archive_member_count": len(members),
        "r1_overlay_manifest_sha256": manifest["r1_overlay_manifest_sha256"],
        "scientific_changes": [],
        "scientific_training_performed_by_builder": False,
        "scientific_evaluation_performed_by_builder": False,
        "actual_a40_post_launch_acceptance": "REQUIRES_TS9",
    }
    write_once(DELIVERY, json.dumps(delivery, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    print(json.dumps(delivery, indent=2))


if __name__ == "__main__":
    main()
