param(
    [string]$Question,
    [ValidateSet('a0','best','both')] [string]$System = 'both',
    [string]$DataDir,
    [string]$ModelPath,
    [switch]$RecoverNames
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'environment.ps1')
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Arguments = @('-m', 'kgqa.demo', '--system', $System)
if ($Question) { $Arguments += @('--question', $Question) }
if ($DataDir) { $Arguments += @('--data-dir', $DataDir) }
if ($ModelPath) { $Arguments += @('--model-path', $ModelPath) }
if ($RecoverNames) { $Arguments += '--recover-names' }
& $Python @Arguments
exit $LASTEXITCODE
