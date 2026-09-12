param(
    [ValidateSet('smoke','full')] [string]$Mode = 'smoke',
    [int]$Target = 50
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'environment.ps1')
$Python = Join-Path $Root '.venv\Scripts\python.exe'
Set-Location $Root

& $Python -m kgqa.fetch --target $Target
& $Python -m kgqa.prepare --snapshot (Join-Path $Root "data\raw\snapshot_$Target.json")
& $Python -m kgqa.pipeline train
& $Python -m kgqa.pipeline predict --split test
& $Python -m kgqa.metrics A0
& $Python -m kgqa.transe --epochs $(if ($Mode -eq 'smoke') { 2 } else { 100 })
& $Python -m kgqa.robustness --limit 100
& $Python -m kgqa.pipeline robust
& $Python -m kgqa.metrics A0 --robustness

$Runs = @(
    @{Id='B0'; Lr='1e-4'; Extra=@()},
    @{Id='B1'; Lr='3e-5'; Extra=@()},
    @{Id='B2'; Lr='3e-4'; Extra=@()},
    @{Id='B3'; Lr='1e-3'; Extra=@()},
    @{Id='C0'; Lr='1e-4'; Extra=@('--no-embeddings')}
)
foreach ($Run in $Runs) {
    $Args = @('-m','kgqa.neural','--run-id',$Run.Id,'--learning-rate',$Run.Lr) + $Run.Extra
    if ($Mode -eq 'smoke') { $Args += '--smoke' }
    & $Python @Args
    & $Python -m kgqa.metrics $Run.Id
}
& $Python -m kgqa.finalize
& $Python -m kgqa.plots
& $Python -m kgqa.environment_info
