param(
    [ValidateRange(1, 65535)] [int]$Port = 7860,
    [string]$DataDir,
    [string]$ModelPath,
    [switch]$NoBrowser,
    [switch]$Legacy,
    [switch]$RecoverNames
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'environment.ps1')
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Project .venv is missing. Follow the setup instructions in README.md.'
}
$Arguments = @('-m', 'kgqa.web', '--port', $Port)
if ($DataDir) { $Arguments += @('--data-dir', $DataDir) }
if ($ModelPath) { $Arguments += @('--model-path', $ModelPath) }
if ($Legacy) { $Arguments += '--legacy' }
if ($RecoverNames) { $Arguments += '--recover-names' }
if (-not $NoBrowser) { $Arguments += '--open' }
& $Python @Arguments
exit $LASTEXITCODE
