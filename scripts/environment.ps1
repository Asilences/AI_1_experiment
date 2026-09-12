$ExperimentRoot = Split-Path -Parent $PSScriptRoot
$env:PIP_CACHE_DIR = Join-Path $ExperimentRoot '.work\pip-cache'
$env:HF_HOME = Join-Path $ExperimentRoot '.work\huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:TORCH_HOME = Join-Path $ExperimentRoot '.work\torch'
$env:PYSTOW_HOME = Join-Path $ExperimentRoot '.work\pystow'
$env:XDG_DATA_HOME = Join-Path $ExperimentRoot '.work\xdg-data'
$env:TEMP = Join-Path $ExperimentRoot '.work\temp'
$env:TMP = $env:TEMP
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
$env:HF_HUB_ETAG_TIMEOUT = '60'
foreach ($ExperimentCache in @($env:PIP_CACHE_DIR, $env:HF_HOME, $env:TORCH_HOME, $env:PYSTOW_HOME, $env:XDG_DATA_HOME, $env:TEMP)) {
    New-Item -ItemType Directory -Force -Path $ExperimentCache | Out-Null
}
