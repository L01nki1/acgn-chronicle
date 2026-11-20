# Run one arm of an answer-line A/B.
#
#   .\experiments\run_focus_arm.ps1 -Arm off -RunId 20261120-fc-off-r1
#   .\experiments\run_focus_arm.ps1 -Arm on  -RunId 20261120-fc-on-r1
#
# Every arm sets BOTH switches explicitly rather than relying on defaults: the rules are
# default-on, so `Remove-Item Env:\...` would silently leave a rule enabled and the "control"
# arm would not be a control. The effective flags are printed and recorded, and the child's
# exit code is propagated so a failed interpreter start or a provider error cannot look like a
# successful run.

param(
    [Parameter(Mandatory = $true)][ValidateSet('off', 'on')][string]$Arm,
    [Parameter(Mandatory = $true)][string]$RunId,
    # Chunking variant. A (512/64) and A' (256/0) are built on the same snapshot and encoder.
    [ValidateSet('A', "A'")][string]$Config = 'A'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. "$PSScriptRoot\env_common.ps1"
Set-Location $RepoRoot

switch ($Arm) {
    'on' {
        $env:ACGN_ANSWER_FOCUS = '1'
        $env:ACGN_ANSWER_QUOTE_DISCIPLINE = '1'
    }
    'off' {
        $env:ACGN_ANSWER_FOCUS = '0'
        $env:ACGN_ANSWER_QUOTE_DISCIPLINE = '0'
    }
}

$args_ = Add-ConfigArg -Arguments $CommonArgs -Config $Config
$args_ += @('--run-id', $RunId, '--out', $OutRoot)
$logDir = Join-Path $RepoRoot 'experiments\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "$RunId.log"

Write-Host ("=== ARM={0}  CONFIG={1}  RUN={2}  FOCUS={3}  QUOTE_DISCIPLINE={4} ===" -f `
    $Arm, $Config, $RunId, $env:ACGN_ANSWER_FOCUS, $env:ACGN_ANSWER_QUOTE_DISCIPLINE)

if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "=== FAILED: interpreter not found at $Py ==="
    exit 1
}

& $Py @args_ *>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE

# The runner exits 1 for "not reportable", which is a verdict, not a crash: the archive is
# written. Anything else that is nonzero is a real failure and must not be reported as success.
$archive = Join-Path $OutRoot $RunId
$hasArchive = Test-Path -LiteralPath (Join-Path $archive 'metrics.json')
Write-Host ("=== EXIT={0}  ARCHIVE={1}  log={2} ===" -f $code, $(if ($hasArchive) { 'written' } else { 'MISSING' }), $log)

if ($code -ne 0 -and -not $hasArchive) {
    Write-Host '=== FAILED: the run did not produce an archive ==='
    exit $code
}
exit 0
