# Run one arm of the quote-discipline A/B experiment.
#
#   .\experiments\run_arm.ps1 -Arm off -RunId 20261120-qd-off-r1
#   .\experiments\run_arm.ps1 -Arm on  -RunId 20261120-qd-on-r1
#
# off = shipped SYSTEM_PROMPT (default).  on = same prompt + appended quote-discipline paragraph.
# Every other knob is identical in both arms.

param(
    [Parameter(Mandatory = $true)][ValidateSet('off', 'on')][string]$Arm,
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$QaSet = 'eval/qa_set_delivery.jsonl'
)

. "$PSScriptRoot\env_common.ps1"
Set-Location $RepoRoot

if ($Arm -eq 'on') { $env:ACGN_ANSWER_QUOTE_DISCIPLINE = '1' }
else { Remove-Item Env:\ACGN_ANSWER_QUOTE_DISCIPLINE -ErrorAction SilentlyContinue }

$args_ = @($CommonArgs)
# allow an alternate QA set without changing anything else
if ($QaSet -ne 'eval/qa_set_delivery.jsonl') {
    $idx = [array]::IndexOf($args_, '--qa')
    $args_[$idx + 1] = $QaSet
}
$args_ += @('--run-id', $RunId, '--out', $OutRoot)

Write-Host "=== ARM=$Arm  RUN=$RunId  QA=$QaSet  QUOTE_DISCIPLINE=$(if ($env:ACGN_ANSWER_QUOTE_DISCIPLINE) {'1'} else {'<unset>'}) ==="
$logDir = Join-Path $RepoRoot 'experiments\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "$RunId.log"

& $Py @args_ *>&1 | Tee-Object -FilePath $log
$code = $LASTEXITCODE
Write-Host "=== EXIT=$code  log=$log ==="
exit $code
