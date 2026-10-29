param(
    [ValidateSet('smoke','full')] [string]$Mode = 'smoke',
    [int]$Target = 50,
    [ValidateRange(1,2147483647)] [int]$BatchSize = 16,
    [ValidateRange(1,2147483647)] [int]$GradientAccumulation = 1
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'environment.ps1')
$Python = Join-Path $Root '.venv\Scripts\python.exe'
Set-Location $Root

function Invoke-ExperimentPython {
    & $Python @args
    if ($LASTEXITCODE -ne 0) { throw "Python experiment step failed with exit code $LASTEXITCODE" }
}

Invoke-ExperimentPython -m kgqa.fetch --target $Target
Invoke-ExperimentPython -m kgqa.prepare --snapshot (Join-Path $Root "data\raw\snapshot_$Target.json")
Invoke-ExperimentPython -m kgqa.pipeline train
Invoke-ExperimentPython -m kgqa.pipeline predict --split test
Invoke-ExperimentPython -m kgqa.metrics A0
Invoke-ExperimentPython -m kgqa.transe --epochs $(if ($Mode -eq 'smoke') { 2 } else { 100 })
Invoke-ExperimentPython -m kgqa.robustness --limit 100
Invoke-ExperimentPython -m kgqa.pipeline robust
Invoke-ExperimentPython -m kgqa.metrics A0 --robustness

$Runs = @(
    @{Id='B0'; Lr='1e-4'; Extra=@()},
    @{Id='B1'; Lr='3e-5'; Extra=@()},
    @{Id='B2'; Lr='3e-4'; Extra=@()},
    @{Id='B3'; Lr='1e-3'; Extra=@()},
    @{Id='C0'; Lr='1e-4'; Extra=@('--no-embeddings')}
)
foreach ($Run in $Runs) {
    $RunArguments = @('-m','kgqa.neural','--run-id',$Run.Id,'--learning-rate',$Run.Lr,
                     '--batch-size',"$BatchSize",'--gradient-accumulation',"$GradientAccumulation") + $Run.Extra
    if ($Mode -eq 'smoke') { $RunArguments += '--smoke' }
    Invoke-ExperimentPython @RunArguments
    Invoke-ExperimentPython -m kgqa.metrics $Run.Id
}
Invoke-ExperimentPython -m kgqa.finalize
Invoke-ExperimentPython -m kgqa.plots
Invoke-ExperimentPython -m kgqa.environment_info
