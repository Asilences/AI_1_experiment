@echo off
setlocal
chcp 65001 >nul
set "PROJECT=%~dp0.."
set "RUNTIME=%PROJECT%\.work\trainfree_5000_runtime_20261031_final\AI_1_experiment"
set "CATALOG=%PROJECT%\data\processed\movies_v2_20261030"
if not exist "%RUNTIME%\kgqa\guarded.py" (
  echo 尚未准备独立5000电影运行目录，请先按 docs/无需训练增强与报告补充_20261031.md 解包。
  exit /b 1
)
if not exist "%CATALOG%\manifest.json" (
  echo 未找到十二属性扩展目录，请准备 data/processed/movies_v2_20261030 后再启动。
  exit /b 1
)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTHONIOENCODING=utf-8"
cd /d "%RUNTIME%"
"%PROJECT%\.venv\Scripts\python.exe" -m kgqa.web --legacy --catalog-dir "%CATALOG%" --port 7863 --open
