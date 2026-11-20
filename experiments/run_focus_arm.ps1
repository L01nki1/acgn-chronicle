# Run one arm of the answer-focus A/B (quote discipline is on in both arms).
#
#   .\experiments\run_focus_arm.ps1 -Arm off -RunId 20261120-fc-off-r1
#   .\experiments\run_focus_arm.ps1 -Arm on  -RunId 20261120-fc-on-r1

param(
    [Parameter(Mandatory = $true)][ValidateSet('off', 'on')][string]$Arm,
    [Parameter(Mandatory = $true)][string]$RunId
)

. "$PSScriptRoot\env_common.ps1"
Set-Location $RepoRoot

if ($Arm -eq 'on') {
    $env:ACGN_ANSWER_FOCUS = '1'
    Remove-Item Env:\ACGN_ANSWER_QUOTE_DISCIPLINE -ErrorAction SilentlyContinue
} else {
    Remove-Item Env:\ACGN_ANSWER_FOCUS -ErrorAction SilentlyContinue
    Remove-Item Env:\ACGN_ANSWER_QUOTE_DISCIPLINE -ErrorAction SilentlyContinue
}

$args_ = @($CommonArgs) + @('--run-id', $RunId, '--out', $OutRoot)
$logDir = Join-Path $RepoRoot 'experiments\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "$RunId.log"

Write-Host "=== ARM=$Arm  RUN=$RunId  FOCUS=$(if ($env:ACGN_ANSWER_FOCUS) {'1'} else {'<unset>'}) ==="
& $Py @args_ *>&1 | Tee-Object -FilePath $log
Write-Host "=== EXIT=$LASTEXITCODE  log=$log ==="
exit 0
