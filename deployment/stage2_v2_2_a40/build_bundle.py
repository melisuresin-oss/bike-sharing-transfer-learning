"""Build the explicit, compact Stage-2/Stage-2B V2.2 A40 package."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = "deployment/stage2_v2_2_a40/"
BUNDLE = BASE + "stage2_stage2b_v2_2_UNIVERSITY_A40.zip"
MANIFEST = BASE + "BUNDLE_MANIFEST.json"
COMMANDS = BASE + "COMMANDS.txt"
DELIVERY = BASE + "DELIVERY_MANIFEST.json"
PSEUDO_TARGETS = {476, 532, 619, 658}


def read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(relative: str, text: str) -> None:
    destination = ROOT / relative
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError("Append-only artifact differs: " + relative)
    else:
        destination.write_text(text, encoding="utf-8")


def module_path(module: str) -> str | None:
    candidate = module.replace(".", "/") + ".py"
    package = module.replace(".", "/") + "/__init__.py"
    if (ROOT / candidate).is_file():
        return candidate
    if (ROOT / package).is_file():
        return package
    return None


def add_parent_initializers(relative: str, pending: list[str], included: set[str]) -> None:
    parent = (ROOT / relative).parent
    while parent != ROOT:
        initializer = parent / "__init__.py"
        if initializer.is_file():
            rel = initializer.relative_to(ROOT).as_posix()
            if rel not in included:
                pending.append(rel)
        parent = parent.parent


def research_import_closure(seeds: set[str]) -> set[str]:
    """Resolve local research imports mechanically using Python's AST."""
    included: set[str] = set()
    pending = list(seeds)
    while pending:
        relative = pending.pop()
        if relative in included:
            continue
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        included.add(relative)
        add_parent_initializers(relative, pending, included)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names if alias.name.startswith("research"))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = Path(relative).with_suffix("").parts[:-1]
                    prefix = list(package_parts[:len(package_parts) - node.level + 1])
                    if node.module:
                        prefix.extend(node.module.split("."))
                    if prefix and prefix[0] == "research":
                        modules.add(".".join(prefix))
                        modules.update(".".join(prefix + [alias.name]) for alias in node.names)
                elif node.module and node.module.startswith("research"):
                    modules.add(node.module)
                    modules.update(node.module + "." + alias.name for alias in node.names)
        for module in modules:
            resolved = module_path(module)
            if resolved and resolved not in included:
                pending.append(resolved)
    return included


def add(files: dict[str, str], relative: str, reason: str) -> None:
    if relative in files and files[relative] != reason:
        files[relative] += "; " + reason
    else:
        files[relative] = reason


def build_file_list() -> dict[str, str]:
    files: dict[str, str] = {}
    python_seeds = {
        "research/stage2_v2_2/core.py",
        "research/stage2_v2_2/a40.py",
        "research/stage2_v2_2/controller.py",
    }
    for relative in research_import_closure(python_seeds):
        add(files, relative, "mechanically resolved local Python import closure")

    control_files = {
        BASE + "EXECUTION_AUTHORIZATION.json": "execution authorization and hard stage firewall",
        BASE + "RESOURCE_CHECK.json": "prepackage operational concurrency evidence",
        BASE + "SCIENTIFIC_DEFINITION_VERIFICATION.json": "scientific-definition verification record",
        BASE + "SCIENTIFIC_DEFINITION_REPORT.md": "human-readable definition verification",
        BASE + "prepare_scientific_definitions.py": "deterministic job-map construction provenance",
        BASE + "run_stage2_stage2b_a40.ps1": "strict preflight and detached launch wrapper",
        BASE + "README.md": "remote execution instructions and lifecycle boundaries",
        BASE + "requirements.txt": "exact validated third-party runtime versions",
        BASE + "test_package.py": "zero-training package regression tests",
        BASE + "build_bundle.py": "allowlist and import-closure package provenance",
        "research/results/stage2_graphgru_selection_v2_2/scientific_contract.json": "frozen Stage-2 scientific contract",
        "research/results/stage2_graphgru_selection_v2_2/selection_rule.json": "frozen Stage-2 selection hierarchy",
        "research/results/stage2_graphgru_selection_v2_2/job_map.json": "immutable 144-job Stage-2 map",
        "research/results/stage2b_vanilla_fairness_v2_2/scientific_contract.json": "frozen Stage-2B scientific contract",
        "research/results/stage2b_vanilla_fairness_v2_2/selection_rule.json": "frozen Stage-2B selection hierarchy",
        "research/results/stage2b_vanilla_fairness_v2_2/job_map.json": "immutable 48-job Stage-2B map",
    }
    for relative, reason in control_files.items():
        add(files, relative, reason)

    from research.stage2_v2_2 import core as c
    for relative in c.AUTHORITY_SHA256:
        add(files, relative, "authoritative frozen design/hash dependency")
    for relative, reason in {
        c.DATA_MANIFEST: "V2.2 development-data authority",
        c.IMPLEMENTATION_MANIFEST: "V2.2 implementation authority",
        c.STAGE1_DECISION: "frozen Stage-1 V2.2 LOG1P decision",
        c.FIT_MANIFEST: "V2.2 source-snapshot authority",
        c.LABEL_MANIFEST: "development retrospective-label authority",
        c.BASE_CACHE_MANIFEST: "frozen execution-cache authority",
        c.GRAPH_RECONCILIATION: "V2.2 static-graph reconciliation authority",
    }.items():
        add(files, relative, reason)

    stage2 = read(c.STAGE2_JOB_MAP)
    stage2b = read(c.STAGE2B_JOB_MAP)
    for job in stage2["jobs"] + stage2b["jobs"]:
        add(files, job["source_snapshot"]["path"], "immutable job source-snapshot binding")

    cache = read(c.BASE_CACHE_MANIFEST)
    for city in cache["cities"].values():
        for array in city["arrays"].values():
            add(files, array["path"], "frozen causal execution-cache payload")

    graphs = read(c.GRAPH_RECONCILIATION)
    for graph in graphs["graphs"]:
        add(files, graph["path"], "reconciled V2.2 static adjacency payload")

    labels = read(c.LABEL_MANIFEST)
    for label in labels["labels"]:
        if label["city_id"] in PSEUDO_TARGETS:
            add(files, label["path"], "registered pseudo-target development evaluation labels")

    return files


def main() -> None:
    sys.path[:0] = [str(ROOT / "tmp/neural_build/pydeps"), str(ROOT)]
    from research.stage2_v2_2 import core as c
    files = build_file_list()
    forbidden = ("final_label", "sealed_final", "final_adaptation", "final_features",
                 "stage3_", "stage4_", "__pycache__", "/checkpoints/", "/predictions/")
    for relative in files:
        lowered = relative.lower()
        if any(token in lowered for token in forbidden) or lowered.endswith(".pt"):
            raise RuntimeError("Forbidden package member: " + relative)
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)

    hashes = {relative: sha(relative) for relative in sorted(files)}
    manifest = {
        "schema_version": "1.0",
        "protocol_version": "2.2",
        "execution_environment": "UNIVERSITY_A40",
        "scope": "STAGE2_AND_STAGE2B_DEVELOPMENT_SELECTION_ONLY",
        "stage1_decision_sha256": "8db24acabd373fa57aa6f35708566da400646fa2ea8474fd0e656a4955828aed",
        "stage2_jobs": 144,
        "stage2b_jobs": 48,
        "total_source_fits": 192,
        "target_adaptation_fits": 0,
        "selected_operational_workers": 4,
        "maximum_operational_workers": 6,
        "worker_count_is_scientific_hyperparameter": False,
        "files": hashes,
        "file_reasons": {relative: files[relative] for relative in sorted(files)},
        "file_count": len(files),
        "payload_bytes": sum((ROOT / relative).stat().st_size for relative in files),
        "excluded": [
            "V2.1 fitted checkpoints and results", "local CPU results", "Stage-1 fitted checkpoints",
            "Stage-3/Stage-4 artifacts", "final-target labels/features/outcomes", "virtual environments",
            "vendored dependencies", "__pycache__",
        ],
        "scientific_training_performed_by_builder": False,
        "scientific_evaluation_performed_by_builder": False,
        "final_target_labels_accessed": False,
    }
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    write_once(MANIFEST, manifest_text)
    manifest_sha = sha(MANIFEST)

    commands = f'''# Run from the fresh extraction root on TS9.
$py="$env:USERPROFILE\\bike_env\\Scripts\\python.exe"
$manifestSha="{manifest_sha}"

# Strict A40 preflight: zero scientific training/evaluation.
& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_a40.ps1 -Mode preflight -ManifestSha256 $manifestSha -Workers 4

# Detached shared execution; only after preflight PASS.
& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_a40.ps1 -Mode launch -ManifestSha256 $manifestSha -Workers 4

# Monitor completed immutable jobs.
(Get-ChildItem .\\research\\results\\stage2_graphgru_selection_v2_2_a40\\completed -Filter *.json -ErrorAction SilentlyContinue).Count
(Get-ChildItem .\\research\\results\\stage2b_vanilla_fairness_v2_2_a40\\completed -Filter *.json -ErrorAction SilentlyContinue).Count

# Monitor coordinator and workers.
Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match 'research.stage2_v2_2.(controller|a40)' }} | Select-Object ProcessId,Name,CommandLine

# Monitor the A40.
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

# Inspect failures without opening metrics.
Get-ChildItem .\\research\\results\\stage2_stage2b_v2_2_a40 -Recurse -Filter *failed*.json | Select-Object FullName,LastWriteTime

# Separate read-only validators after completion (the controller also runs these before freezing).
& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_a40.ps1 -Mode validate-stage2 -ManifestSha256 $manifestSha -Workers 4
& .\\deployment\\stage2_v2_2_a40\\run_stage2_stage2b_a40.ps1 -Mode validate-stage2b -ManifestSha256 $manifestSha -Workers 4
'''
    write_once(COMMANDS, commands)

    if "--manifest-only" in sys.argv:
        print(json.dumps({"manifest_sha256": manifest_sha, "file_count": len(files),
                          "payload_bytes": manifest["payload_bytes"]}))
        return
    if (ROOT / BUNDLE).exists():
        raise FileExistsError("Append-only bundle already exists: " + BUNDLE)
    members = set(files) | {MANIFEST, COMMANDS}
    with zipfile.ZipFile(ROOT / BUNDLE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(members):
            archive.write(ROOT / relative, relative)
    delivery = {
        "status": "LOCAL_PACKAGE_VERIFICATION_PENDING",
        "bundle": {"path": BUNDLE, "sha256": sha(BUNDLE), "bytes": (ROOT / BUNDLE).stat().st_size},
        "bundle_manifest": {"path": MANIFEST, "sha256": manifest_sha},
        "stage2_scientific_contract": {"path": "research/results/stage2_graphgru_selection_v2_2/scientific_contract.json",
                                        "sha256": sha("research/results/stage2_graphgru_selection_v2_2/scientific_contract.json")},
        "stage2b_scientific_contract": {"path": "research/results/stage2b_vanilla_fairness_v2_2/scientific_contract.json",
                                         "sha256": sha("research/results/stage2b_vanilla_fairness_v2_2/scientific_contract.json")},
        "stage2_job_map": {"path": c.STAGE2_JOB_MAP, "sha256": sha(c.STAGE2_JOB_MAP)},
        "stage2b_job_map": {"path": c.STAGE2B_JOB_MAP, "sha256": sha(c.STAGE2B_JOB_MAP)},
        "archive_member_count": len(members),
        "scientific_training_performed_by_builder": False,
        "scientific_evaluation_performed_by_builder": False,
        "actual_A40_preflight_status": "REQUIRES_EXECUTION_ON_TS9",
    }
    write_once(DELIVERY, json.dumps(delivery, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    print(json.dumps(delivery, indent=2))


if __name__ == "__main__":
    main()
