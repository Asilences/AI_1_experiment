param([ValidateSet('local', 'server')][string]$Target = 'local')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'
Write-Output ('Local project: ' + $projectRoot)
if ($Target -eq 'local') {
    & (Join-Path $projectRoot '.venv/Scripts/python.exe') -c "import sys,platform,importlib.metadata as m; print('Python executable:',sys.executable); print('Python:',sys.version.split()[0]); print('Platform:',platform.platform()); [print(p+':',m.version(p)) for p in ['torch','transformers','numpy','scikit-learn','pykeen','pytest']]; import torch; print('CUDA build:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
} else {
    $probe = @'
import sys, os, platform, importlib.metadata as m, subprocess
os.chdir('/home/junjiex/AI1experiment')
print('Remote project:', os.getcwd())
print('Python executable:', sys.executable)
print('Python:', sys.version.split()[0])
print('Platform:', platform.platform())
for p in ['torch','transformers','numpy','scikit-learn']:
 print(p+':', m.version(p))
import torch
print('PyTorch CUDA build:', torch.version.cuda)
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv'], text=True))
'@
    $probe | ssh -o BatchMode=yes -p 50012 junjiex@222.20.94.68 '/home/junjiex/AI1experiment/.venv/bin/python -'
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
