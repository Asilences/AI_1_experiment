# Data directory

The experiment data is intentionally not included in the code archive.

For a quick offline code check without the formal dataset, install
`requirements-test.txt` and run `python -m kgqa.acceptance`. It generates a tiny
synthetic fixture in an isolated output directory. See
[submission instructions](../docs/提交验收说明.md) for expected results and demo paths.

To rebuild it, activate the project-local environment and run:

```powershell
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe -m kgqa.fetch --target 1000
.\.venv\Scripts\python.exe -m kgqa.prepare --snapshot .\data\raw\snapshot_1000.json
```

The frozen formal run used snapshot SHA-256:
`eb85268c3359238b7eeed2438e3693c5e42ab2de09ce4a32d4137a5196613739`.
