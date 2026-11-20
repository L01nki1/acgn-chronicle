# Run one arm of an answer-line A/B.
#
#   .\experiments\run_arm.ps1 -Arm off -RunId 20261120-qd-off-r1
#
# Both prompt switches are set explicitly. They are default-on, so clearing the variable would
# not disable a rule and the "off" arm would not be a control. The effective flags are printed,
# and a run that produces no archive is reported as a failure rather than as success.

param(
    [Parameter(Mandatory = $true)][ValidateSet('off', 'on')][string]$Arm,
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$QaSet = 'eval/qa_set_delivery.jsonl'
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

$args_ = @($CommonArgs)
if ($QaSet -ne 'eval/qa_set_delivery.jsonl') {
    $idx = [array]::IndexOf($args_, '--qa')
    $args_[$idx + 1] = $QaSet
}
$args_ += @('--run-id', $RunId, '--out', $OutRoot)

Write-Host ("=== ARM={0}  RUN={1}  QA={2}  FOCUS={3}  QUOTE_DISCIPLINE={4} ===" -f `
    $Arm, $RunId, $QaSet, $env:ACGN_ANSWER_FOCUS, $env:ACGN_ANSWER_QUOTE_DISCIPLINE)

if (-not (Test-Path -LiteralPath $Py)) {
    Write-Host "=== FAILED: interpreter not found at $Py ==="
    exit 1
}

$logDir = Join-Path $RepoRoot 'experiments\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "$RunId.log"

& $Py @args_ *>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE

# Exit 1 means "not reportable": a verdict on a written archive, not a crash.
$archive = Join-Path $OutRoot $RunId
$hasArchive = Test-Path -LiteralPath (Join-Path $archive 'metrics.json')
Write-Host ("=== EXIT={0}  ARCHIVE={1}  log={2} ===" -f $code, $(if ($hasArchive) { 'written' } else { 'MISSING' }), $log)

if ($code -ne 0 -and -not $hasArchive) {
    Write-Host '=== FAILED: the run did not produce an archive ==='
    exit $code
}
exit 0
