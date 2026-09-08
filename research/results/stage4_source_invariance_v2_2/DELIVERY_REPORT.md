# Stage 4 V2.2 — completed implementation and A40 delivery

**STAGE4_A40_PACKAGE_READY_AFTER_PROSPECTIVE_SELECTION_CLARIFICATION**

## A. Pre-clarification firewall and resumed state

The preserved pre-execution inventory covered 27,885 paths and found no V2.2
Stage-4 scientific fit, prediction, evaluation, ranking, or winner artifact.
Historical V2.1 seals and the earlier blocked V2.2 gate were distinguished from
new scientific results. No Stage-4 candidate results were inspected.

Evidence: `research/results/stage4_selection_clarification_v2_2/pre_execution_firewall.json`
SHA256: `0c56f51a1fed9d5e34da1ea49ac7b9bd373f9bd93eebbf3f4c10a7e338a99cfb`.

At resumption, the clarification and eight ranking tests were complete. The
runner, controller, validators, launcher and synthetic suite existed; saved
reports recorded 102 and then 111 passing checks. Only `a40.py` had changed
after the latest report, to strengthen package bindings. There was no sealed
Stage-4 implementation contract, job map, manifest, ZIP, or scientific output
directory. The 111-check suite was therefore rerun against the current code.
Earlier reports and fixtures were retained.

Scoped Git inspection found the new Stage-4 files untracked. Repository-wide
Git status encountered a read-only Git LFS temporary-file restriction; scoped
status and authoritative file hashes supplied the relevant state evidence.
No Git configuration, index, or historical result was changed.

## B. Prospective clarification

Artifact: `research/results/stage4_selection_clarification_v2_2/selection_clarification.json`

SHA256: `852d8c32dd9004a8dfaa17633e285877828672d79414cf5fb67b8550c8ea7d4d`

Sealed: `2026-09-08T10:31:36.546015+00:00`. Its bytes were verified and preserved.
The previously missing definitions are explicitly prospective, not historical.

For each candidate, average post-7-day/300-update count-space MAE across seeds
17, 29, 43 within each pseudo-target fold, then equally average the four fold
means. Primary equality is exactly:

`Decimal(str(primary_unrounded)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)`.

The quantized value determines equality only. When not tied, compare unrounded
primary MAE. Ties use lower **unrounded population SD of the four seed-mean
fold MAEs, N=4**, with exact stored-number equality and no tolerance; then
smaller lambda; then constant before linear. Unrounded statistics are retained.
No secondary metric, zero-shot result, discriminator diagnostic, or final-target
performance enters selection.

All eight independent synthetic ranking cases pass, including rounding
boundaries, population versus sample SD, both remaining tie-breaks, and input
order independence. The hash-bound selection implementation was not altered.

## C. Scientific re-audit

**PASS: 16 checks; no remaining Stage-4 scientific ambiguity.**
Artifact: `research/results/stage4_source_invariance_v2_2/scientific_reaudit.json`.

The frozen design is GGRU_K04_H032_D00, LOG1P, seven source cities excluding
the pseudo-target, the six registered lambda/schedule candidates, four folds,
and seeds 17/29/43. Source fits use 12,000 updates, ordinary unit-weight
seven-class CE through GRL, and the current permitted fitting target mask at
cutoff HD. Predictors remain frozen at their original origins. Adaptation
discards the discriminator, resets AdamW, and uses exactly 300 updates over
the fixed elapsed `[HD-168h, HD)` window.

Scientific execution is authorized **only on UNIVERSITY_A40 after strict
preflight PASS**. The controller runs development jobs, then stops. Run the
read-only postrun validator, freeze the decision, and STOP before final evaluation.

## D. Implementation and preservation

New Stage-4 implementation files under `research/stage4_v2_2/`:
`__init__.py`, `core.py`, `method.py`, `sampler.py`, `selection.py`,
`verify_selection.py`, `a40.py`, `controller.py`, `synthetic_worker.py`, `tests.py`.

New deployment files under `deployment/stage4_v2_2_a40/`:
`run_stage4_a40.ps1`, `prepare.py`, `README.md`, `requirements-ts9.txt`,
`BUNDLE_MANIFEST.json`, `DELIVERY_SHA256.json`, and the ZIP below.

New governance artifacts under `research/results/stage4_source_invariance_v2_2/`:
`scientific_contract.json`, `job_map.json`, `scientific_reaudit.json`,
`implementation_test_report.json`, `execution_authorization.json`,
`provenance.json`, `package_validation_report.json`, and this report.

Only new, unsealed Stage-4 draft implementation/test files were edited during
completion. Existing V2.1 files, Stage-1/2/2B decisions, Stage-3 rebind, V2.2
data, and the prospective clarification remain unchanged. Existing
architectures, training primitives and the accepted V2.2 fitting cache are
reused by hash. The exact EqualCitySampler class is extracted from its pinned
source AST without importing legacy data readers. The sole legacy data
exception is approved, reconciled geographic adjacency.

## E. Tests and isolated package verification

**111 implementation checks PASS**, including the independent eight-case
ranking suite. The final test report binds the tested code hashes.

Actual PowerShell detached startup exercised the real controller import path
and shared four-worker dispatcher. Eight synthetic dependent jobs succeeded
with `EXITED_ZERO_WITH_COMPLETION`; separate fixtures demonstrated that a
nonzero worker exit and an exit-zero worker with no completion stop dispatch.

Checkpoint, prediction and completion hash corruptions are rejected. Rehashing
does not conceal changed metadata values, missing metadata keys, incomplete
update budgets, wrong parameter shapes or a false completed-fit flag. Valid
completed jobs skip without constructing an optimizer. Partials remain
non-completions and cannot be resumed as trained checkpoints. Half-committed
completions fail closed.

Semantic canonicalization preserves values and keys: recursive tuple/list,
NumPy/native scalar, and Path/string normalization only. No rounding, key
dropping, defaults, or ignored numerical differences is permitted.

The package was verified in staging and again after **clean ZIP extraction**:
all 211 member hashes matched; package-check and validate-existing passed;
all imported research modules resolved inside the isolated root. Strict
preflight rejected the local non-A40 Python before scientific execution.
Actual A40 preflight remains to be run on TS9.

REAL local scientific optimizer updates = **0**.
REAL local scientific evaluations = **0**.
The suite performed two explicitly synthetic toy optimizer updates.
No scientific output directory was created locally.

## F. Scientific contract

Path: `research/results/stage4_source_invariance_v2_2/scientific_contract.json`

SHA256: `94e586624d18ac9047c6b0a71922139ad00795da54e20ca05d8ef04008470e15`

## G. Immutable job map

Path: `research/results/stage4_source_invariance_v2_2/job_map.json`

SHA256: `ce84b658100d49a313fa102ea6b07d0319d197f1e23d610af53b420a8492458c`

72 source fits + 72 adaptation fits = **144 scientific fits**.
Selection uses exactly 72 post-adaptation evaluation records and zero
zero-shot selection evaluations.

## H. A40 package

ZIP: `deployment/stage4_v2_2_a40/stage4_v2_2_a40_e30caa42b3e9.zip`

ZIP SHA256: `c4f4e75bc8f5d6c90fa273efb35667ce8f86022e8a3c597b1b62b5d28165efd5`

Size: **24,831,577 bytes** (23.68 MiB); 211 members;
676,778,125 uncompressed bytes.

Manifest: `deployment/stage4_v2_2_a40/BUNDLE_MANIFEST.json`

Manifest SHA256: `e30caa42b3e92f1a80220083493b8278da31d25f1aeb8eb8744e16ed6f27b87b`

The package contains the required development cache, four pseudo-target
development-label files, fit-snapshot metadata, static graphs, governance,
implementation and dependencies. It contains no historical fitted checkpoints,
final held-out labels, or final-experiment runner. Upstream decisions are
included as provenance; their historical payloads are not bundled.

## I. Exact TS9 PowerShell commands

Run steps 1–9 in one TS9 PowerShell session. Steps 10–12 are progress checks.
Wait until all jobs finish before steps 13–18. Run step 19 on the local PC.
Do not paste the entire sequence and assume the asynchronous training has finished.

The run root resolves to `C:\Users\go54ray\projects\s4_e30caa42` for the
registered TS9 account. Expected outputs are under
`C:\Users\go54ray\projects\s4_e30caa42\research\results\stage4_source_invariance_v2_2_a40`.

```powershell
# 1. Copy the verified delivery ZIP through the redirected D: drive.
$ErrorActionPreference = 'Stop'
$py = "$env:USERPROFILE\bike_env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "Required bike_env Python missing: $py" }
$zipName = 'stage4_v2_2_a40_e30caa42b3e9.zip'
$zipSha = 'c4f4e75bc8f5d6c90fa273efb35667ce8f86022e8a3c597b1b62b5d28165efd5'
$manifestSha = 'e30caa42b3e92f1a80220083493b8278da31d25f1aeb8eb8744e16ed6f27b87b'
$deliveryRoot = Join-Path $env:USERPROFILE 'projects\s4_delivery_e30caa42'
$runRoot = Join-Path $env:USERPROFILE 'projects\s4_e30caa42'
if ((Test-Path -LiteralPath $deliveryRoot) -or (Test-Path -LiteralPath $runRoot)) { throw 'Use a fresh root; do not overwrite an earlier run' }
New-Item -ItemType Directory -Path $deliveryRoot | Out-Null
$zip = Join-Path $deliveryRoot $zipName
$redirectedPackage = "\\tsclient\D\Deep\deepest\european-bike-sharing-dataset\deployment\stage4_v2_2_a40\$zipName"
Copy-Item -LiteralPath $redirectedPackage -Destination $zip

# 2. Verify the external ZIP hash before extraction.
if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $zipSha) { throw 'ZIP SHA256 mismatch' }

# 3. Extract to the fresh run root and set the deterministic environment.
Expand-Archive -LiteralPath $zip -DestinationPath $runRoot
Set-Location -LiteralPath $runRoot
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$launcher = Join-Path $runRoot 'deployment\stage4_v2_2_a40\run_stage4_a40.ps1'
$manifest = Join-Path $runRoot 'deployment\stage4_v2_2_a40\BUNDLE_MANIFEST.json'
$out = Join-Path $runRoot 'research\results\stage4_source_invariance_v2_2_a40'
if ((Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $manifestSha) { throw 'Manifest SHA256 mismatch' }

# 4. Package-only integrity gate.
& $py -X utf8 -B -m research.stage4_v2_2.a40 package-check --manifest-sha256 $manifestSha
if ($LASTEXITCODE -ne 0) { throw 'Package check failed' }

# 5. Strict actual-A40 preflight; no scientific training.
& $py -X utf8 -B -m research.stage4_v2_2.a40 preflight --manifest-sha256 $manifestSha
if ($LASTEXITCODE -ne 0) { throw 'A40 preflight failed; do not launch' }

# 6. Read-only existing-job/dry gate. A fresh root should show completed = 0.
& $py -X utf8 -B -m research.stage4_v2_2.a40 validate-existing --manifest-sha256 $manifestSha
if ($LASTEXITCODE -ne 0) { throw 'Existing-job validation failed' }

# 7. Detach the four-worker coordinator. It repeats strict preflight itself.
$launchJson = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode launch -ManifestSha256 $manifestSha -Workers 4
if ($LASTEXITCODE -ne 0) { throw 'Detached launcher failed' }
$launch = ($launchJson -join "`n") | ConvertFrom-Json
$launch | Format-List

# Recovery only: use this same command with -Resume after the previous
# coordinator has stopped. Valid completions skip; partial attempts restart
# at update zero. Do not delete corrupt or half-committed completions.
# $launchJson = & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode launch -ManifestSha256 $manifestSha -Workers 4 -Resume
# $launch = ($launchJson -join "`n") | ConvertFrom-Json

# 8. Coordinator PID check.
Get-Process -Id $launch.CoordinatorPID -ErrorAction SilentlyContinue |
    Select-Object Id, ProcessName, StartTime, CPU

# 9. Confirm the actual device and active GPU processes.
nvidia-smi

# 10. Progress counts are informational; the validator establishes authority.
$completed = @(Get-ChildItem -LiteralPath (Join-Path $out 'completed') -Filter '*.json' -ErrorAction SilentlyContinue |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
[pscustomobject]@{
    SourceCompleted = @($completed | Where-Object { $_.job.phase -eq 'source' -and $_.completed_fit }).Count
    AdaptationCompleted = @($completed | Where-Object { $_.job.phase -eq 'adaptation' -and $_.completed_fit }).Count
    TotalCompleted = $completed.Count
    Expected = 144
}
Get-ChildItem -LiteralPath (Join-Path $out 'controller_runs') -Filter '*.stdout.log' -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 0 } | Sort-Object LastWriteTime -Descending | Select-Object -First 4 |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 2 }

# 11. Coordinator and worker stderr.
Get-Content -LiteralPath $launch.StandardError -Tail 40
Get-ChildItem -LiteralPath (Join-Path $out 'controller_runs') -Filter '*.stderr.log' -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 0 } |
    ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Tail 20 }

# 12. Worker exits: only EXITED_ZERO_WITH_COMPLETION is successful.
$exits = @(Get-ChildItem -LiteralPath (Join-Path $out 'controller_runs') -Filter '*.exit.json' -Recurse -ErrorAction SilentlyContinue |
    ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json })
$exits | Group-Object status | Select-Object Name, Count
@($exits | Where-Object { $_.status -eq 'EXITED_ZERO_WITH_COMPLETION' } |
    Select-Object -ExpandProperty job_id -Unique).Count

# 13. After training ends, require exactly 144 completed immutable fits.
$dryJson = & $py -X utf8 -B -m research.stage4_v2_2.a40 validate-existing --manifest-sha256 $manifestSha
if ($LASTEXITCODE -ne 0) { throw 'Completion validation failed' }
$dry = ($dryJson -join "`n") | ConvertFrom-Json
if ($dry.status -ne 'PASS' -or $dry.completed -ne 144 -or $dry.remaining -ne 0) { throw 'Not yet 144/144 validated fits' }
$dry

# 14. Read-only postrun validation; capture stdout as a separate report.
$postrunJson = & $py -X utf8 -B -m research.stage4_v2_2.a40 validate --manifest-sha256 $manifestSha
if ($LASTEXITCODE -ne 0) { throw 'Postrun validation failed; do not freeze' }
$postrun = ($postrunJson -join "`n") | ConvertFrom-Json
if ($postrun.status -ne 'PASS' -or $postrun.completed_fits -ne 144) { throw 'Postrun gate not PASS' }
$postrunReport = Join-Path $out ('postrun_stdout_' + [guid]::NewGuid().ToString('N') + '.json')
$postrunJson | Set-Content -LiteralPath $postrunReport -Encoding UTF8

# 15. Apply the sealed selection rule, freeze the winner, then STOP.
& $py -X utf8 -B -m research.stage4_v2_2.a40 freeze --manifest-sha256 $manifestSha
if ($LASTEXITCODE -ne 0) { throw 'Stage-4 freeze failed' }

# 16. Print the frozen decision and its hash.
$decisionPath = Join-Path $out 'stage4_decision_manifest.json'
Get-Content -LiteralPath $decisionPath -Raw
Get-FileHash -LiteralPath $decisionPath -Algorithm SHA256

# 17. Archive all Stage-4 outputs, including logs and partial-attempt provenance.
if (Get-Process -Id $launch.CoordinatorPID -ErrorAction SilentlyContinue) {
    Wait-Process -Id $launch.CoordinatorPID -Timeout 30
}
$archive = Join-Path $deliveryRoot ('stage4_results_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.zip')
if (Test-Path -LiteralPath $archive) { throw 'Archive destination exists' }
Compress-Archive -LiteralPath $out -DestinationPath $archive -CompressionLevel Optimal
$archiveSha = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
$archiveHashFile = $archive + '.sha256'
($archiveSha + '  ' + (Split-Path -Leaf $archive)) | Set-Content -LiteralPath $archiveHashFile -Encoding ASCII
Get-Content -LiteralPath $archiveHashFile

# 18. Copy the archive and hash back to a separate local return directory.
$returnRoot = '\\tsclient\D\Deep\deepest\european-bike-sharing-dataset\deployment\stage4_v2_2_a40\returns\e30caa42'
New-Item -ItemType Directory -Path $returnRoot -Force | Out-Null
$returnArchive = Join-Path $returnRoot (Split-Path -Leaf $archive)
$returnHash = Join-Path $returnRoot (Split-Path -Leaf $archiveHashFile)
if ((Test-Path -LiteralPath $returnArchive) -or (Test-Path -LiteralPath $returnHash)) { throw 'Return artifact already exists' }
Copy-Item -LiteralPath $archive -Destination $returnArchive
Copy-Item -LiteralPath $archiveHashFile -Destination $returnHash
if ((Get-FileHash -LiteralPath $returnArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $archiveSha) { throw 'Return transfer hash mismatch' }

# 19. Run this block ON THE LOCAL PC, not on TS9.
$localReturnRoot = 'D:\Deep\deepest\european-bike-sharing-dataset\deployment\stage4_v2_2_a40\returns\e30caa42'
$returnedArchives = @(Get-ChildItem -LiteralPath $localReturnRoot -Filter '*.zip' -File)
if ($returnedArchives.Count -eq 0) { throw 'No returned archive found' }
foreach ($returned in $returnedArchives) {
    $expected = ((Get-Content -LiteralPath ($returned.FullName + '.sha256') -Raw).Trim() -split '\s+')[0]
    $actual = (Get-FileHash -LiteralPath $returned.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { throw "Returned archive SHA256 mismatch: $($returned.Name)" }
    [pscustomobject]@{ Archive = $returned.FullName; SHA256 = $actual; Status = 'PASS' }
}
```

## J. Final-target firewall

`final_target_labels_accessed = false`

`final_experiment_started = false`

No final-label access for Mannheim 195, Innsbruck 199, Glasgow 237 or Split 617.
No real scientific training or evaluation was run locally.

## K. Unresolved downstream issues

`FINAL_SEED_COUNT_UNRESOLVED_3_VS_5`

`FINAL_SEED_BOOTSTRAP_AGGREGATION_UNRESOLVED`

These remain open for final evaluation; neither changes the frozen three-seed
Stage-4 development workload.

**STAGE4_A40_PACKAGE_READY_AFTER_PROSPECTIVE_SELECTION_CLARIFICATION**
