param(
 [ValidateSet('package-check','preflight','validate-existing','launch','validate','freeze')][string]$Mode='package-check',
 [string]$ManifestSha256,
 [ValidateSet(4)][int]$Workers=4,
 [switch]$Resume,
 [string]$StartupFixture,
 [string]$FixturePython
)
$ErrorActionPreference='Stop'
$stage4Root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$py="$env:USERPROFILE\bike_env\Scripts\python.exe"
if ($StartupFixture) {
 if ($Mode -ne 'launch' -or $ManifestSha256 -or $Resume -or -not $FixturePython) { throw 'Synthetic startup requires launch, explicit FixturePython, no production hash or resume' }
 $stage4Fixture=Get-Content -LiteralPath $StartupFixture -Raw | ConvertFrom-Json
 if ($stage4Fixture.purpose -ne 'NON_SCIENTIFIC_CONTROLLER_STARTUP_FIXTURE') { throw 'Invalid synthetic fixture' }
 $py=$FixturePython
} elseif ($FixturePython) { throw 'FixturePython is prohibited for scientific execution' }
if (-not (Test-Path -LiteralPath $py)) { throw "Required executable missing: $py" }
if (-not $StartupFixture -and $ManifestSha256 -notmatch '^[0-9a-f]{64}$') { throw 'Explicit manifest SHA256 required' }
if ($env:CUBLAS_WORKSPACE_CONFIG -and $env:CUBLAS_WORKSPACE_CONFIG -ne ':4096:8') { throw 'Conflicting CuBLAS workspace configuration' }
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONNOUSERSITE='1';$env:PYTHONDONTWRITEBYTECODE='1'
if ($Mode -eq 'launch') {
 $stage4RunId=[guid]::NewGuid().ToString('N')
 if ($StartupFixture) {
  $stage4Logs=Join-Path $stage4Root "tmp\stage4_v2_2_synthetic_tests\launcher_$stage4RunId"
  $stage4Args=@('-X','utf8','-B','-m','research.stage4_v2_2.controller','--workers','4','--startup-fixture',('"'+$StartupFixture+'"'))
 } else {
  $stage4Logs=Join-Path $stage4Root 'research\results\stage4_source_invariance_v2_2_a40\launcher_logs'
  $stage4Args=@('-X','utf8','-B','-m','research.stage4_v2_2.controller','--manifest-sha256',$ManifestSha256,'--workers','4')
  if ($Resume) { $stage4Args+='--resume' }
 }
 New-Item -ItemType Directory -Path $stage4Logs -Force | Out-Null
 $stage4Stdout=Join-Path $stage4Logs "$stage4RunId.stdout.log"
 $stage4Stderr=Join-Path $stage4Logs "$stage4RunId.stderr.log"
 $stage4Process=Start-Process -FilePath $py -ArgumentList $stage4Args -WorkingDirectory $stage4Root -WindowStyle Hidden -RedirectStandardOutput $stage4Stdout -RedirectStandardError $stage4Stderr -PassThru
 $stage4Launch=[pscustomobject]@{CoordinatorPID=$stage4Process.Id; StandardOutput=$stage4Stdout; StandardError=$stage4Stderr; Workers=4; ManifestSha256=$ManifestSha256; StartupFixture=$StartupFixture}
 $stage4Launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage4Logs "$stage4RunId.launch.json") -Encoding UTF8
 $stage4Launch | ConvertTo-Json
} else {
 Push-Location -LiteralPath $stage4Root
 try {
  $stage4Args=@('-X','utf8','-B','-m','research.stage4_v2_2.a40',$Mode,'--manifest-sha256',$ManifestSha256)
  if ($Resume) { $stage4Args+='--resume' }
  & $py @stage4Args
  if ($LASTEXITCODE -ne 0) { throw "Stage-4 $Mode failed with exit code $LASTEXITCODE" }
 } finally { Pop-Location }
}
