# Shared environment for the answer-line A/B runs. Dot-source from a launcher:
#   . .\experiments\env_common.ps1
#
# The launchers set ACGN_ANSWER_FOCUS and ACGN_ANSWER_QUOTE_DISCIPLINE explicitly for each arm;
# both rules are default-on, so clearing a variable is NOT the same as disabling a rule.

$env:ACGN_DATA_DIR = 'D:\ACGNdaily-Database\.bge-index'
$env:ACGN_EMBED_MODEL = 'BAAI/bge-m3'
$env:ACGN_EMBED_DEVICE = 'cuda'
$env:ACGN_EMBED_BATCH = '16'
$env:HF_HOME = 'D:\ACGNdaily-Database\.hf-home'
$env:HF_HUB_OFFLINE = '1'
$env:HF_HUB_DISABLE_SYMLINKS = '1'
$env:SQLITE_TMPDIR = 'D:\ACGNdaily-Database\.sqltmp'
$env:TMP = 'D:\ACGNdaily-Database\.sqltmp'
$env:TEMP = 'D:\ACGNdaily-Database\.sqltmp'
$env:PYTHONIOENCODING = 'utf-8'

$Script:RepoRoot = 'D:\ACGNdaily-Database'
$Script:Py       = 'D:\ACGNdaily-Database\.evalvenv\Scripts\python.exe'
$Script:OutRoot  = 'D:\ACGNdaily-Database\evalout'

# Every A/B run invokes this fixed set; the arm switches and --config are injected by the launcher.
$Script:CommonArgs = @(
    'scripts/run_eval.py',
    '--snapshot', '20261105-d797b5f0ae8e',
    '--qa', 'eval/qa_set_delivery.jsonl',
    '--answerer', 'live',
    '--translate-queries', 'zh',
    '--official-bonus', '1.08',
    '--window-start', '2023-11-05',
    '--window-end', '2026-11-05'
)

# Insert `--config <name>` at the front, so the launcher's -Config choice is the only place the
# chunking variant is decided. A and A' are both built on the same snapshot and encoder.
function Add-ConfigArg {
    param([string[]]$Arguments, [string]$Config = 'A')
    return @('scripts/run_eval.py', '--config', $Config) + ($Arguments | Where-Object { $_ -ne 'scripts/run_eval.py' })
}
