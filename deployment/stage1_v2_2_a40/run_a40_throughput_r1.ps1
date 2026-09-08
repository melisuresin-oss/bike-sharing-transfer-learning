param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('preflight','launch','validate')][string]$Mode,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ManifestSha256,
    [ValidateRange(1,4)][int]$Workers=4
)
$ErrorActionPreference = 'Stop'
$py="$env:USERPROFILE\bike_env\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "Required bike_env Python is missing: $py" }
$stage1Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
if ($env:CUBLAS_WORKSPACE_CONFIG -and $env:CUBLAS_WORKSPACE_CONFIG -ne ':4096:8') {
    throw 'Conflicting CUBLAS_WORKSPACE_CONFIG'
}
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
if ($Mode -eq 'launch') {
    $stage1Logs = Join-Path $stage1Root 'research\results\stage1_scale_loss_v2_2_a40\logs'
    New-Item -ItemType Directory -Path $stage1Logs -Force | Out-Null
    $stage1Stamp = [guid]::NewGuid().ToString('N')
    $stage1Stdout = Join-Path $stage1Logs "throughput_$stage1Stamp.stdout.log"
    $stage1Stderr = Join-Path $stage1Logs "throughput_$stage1Stamp.stderr.log"
    $stage1Process = Start-Process -FilePath $py `
        -ArgumentList @('-X','utf8','-B','-m','research.stage1_v2_2.a40_parallel','launch','--manifest-sha256',$ManifestSha256,'--workers',[string]$Workers) `
        -WorkingDirectory $stage1Root -WindowStyle Hidden `
        -RedirectStandardOutput $stage1Stdout -RedirectStandardError $stage1Stderr -PassThru
    [pscustomobject]@{
        CoordinatorPID=$stage1Process.Id
        OperationalWorkers=$Workers
        StandardOutput=$stage1Stdout
        StandardError=$stage1Stderr
        OutputDirectory=(Join-Path $stage1Root 'research\results\stage1_scale_loss_v2_2_a40')
        DetachmentScope='Survives invoking PowerShell closure and RDP disconnect; not Windows logoff, reboot, or host shutdown'
    }
} else {
    Push-Location -LiteralPath $stage1Root
    try {
        & $py -X utf8 -B -m research.stage1_v2_2.a40_parallel $Mode --manifest-sha256 $ManifestSha256 --workers $Workers
        if ($LASTEXITCODE -ne 0) { throw "Stage-1 throughput $Mode failed with exit code $LASTEXITCODE" }
    } finally { Pop-Location }
}
