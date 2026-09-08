param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('package-check','preflight','validate-existing','launch','validate-stage2','validate-stage2b')][string]$Mode,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ManifestSha256,
    [ValidateRange(4,4)][int]$Workers=4
)
$ErrorActionPreference='Stop'
$py="$env:USERPROFILE\bike_env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "Required bike_env Python is missing: $py" }
$repo=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if ($env:CUBLAS_WORKSPACE_CONFIG -and $env:CUBLAS_WORKSPACE_CONFIG -ne ':4096:8') {
    throw 'Conflicting CUBLAS_WORKSPACE_CONFIG'
}
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
if ($Mode -eq 'launch') {
    $logs=Join-Path $repo 'research\results\stage2_stage2b_v2_2_a40\operational_r2\logs'
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    $stamp=[guid]::NewGuid().ToString('N')
    $stdout=Join-Path $logs "controller_r2_$stamp.stdout.log"
    $stderr=Join-Path $logs "controller_r2_$stamp.stderr.log"
    $process=Start-Process -FilePath $py `
        -ArgumentList @('-X','utf8','-B','-m','research.stage2_v2_2.controller_r2','launch','--manifest-sha256',$ManifestSha256,'--workers','4') `
        -WorkingDirectory $repo -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    [pscustomobject]@{
        CoordinatorPID=$process.Id
        OperationalWorkers=4
        StandardOutput=$stdout
        StandardError=$stderr
        Acceptance='PID alone is insufficient; require live controller/CUDA worker, completion count above 4, a new zero exit, and subsequent dispatch'
    }
} else {
    Push-Location -LiteralPath $repo
    try {
        if ($Mode -eq 'package-check') {
            & $py -X utf8 -B -m research.stage2_v2_2.a40_r2 package-check --manifest-sha256 $ManifestSha256 --workers 4
        } elseif ($Mode -eq 'preflight') {
            & $py -X utf8 -B -m research.stage2_v2_2.a40_r2 preflight --manifest-sha256 $ManifestSha256 --workers 4
        } elseif ($Mode -eq 'validate-existing') {
            & $py -X utf8 -B -m research.stage2_v2_2.a40_r2 validate-existing --manifest-sha256 $ManifestSha256 --workers 4
        } elseif ($Mode -eq 'validate-stage2') {
            & $py -X utf8 -B -m research.stage2_v2_2.a40_r2 validate --manifest-sha256 $ManifestSha256 --workers 4 --stage stage2
        } else {
            & $py -X utf8 -B -m research.stage2_v2_2.a40_r2 validate --manifest-sha256 $ManifestSha256 --workers 4 --stage stage2b
        }
        if ($LASTEXITCODE -ne 0) { throw "$Mode failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}
