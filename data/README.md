# 数据目录与版本

完整实验数据和模型不包含在 GitHub 或代码压缩包中。没有正式数据时，可安装 `requirements-test.txt` 并运行 `python -m kgqa.acceptance`，生成独立的小型虚构夹具；该夹具仅用于代码验收，操作见[提交验收说明](../docs/提交验收说明.md)。

原实验数据位于 `data/raw/snapshot_1000.json` 和 `data/processed` 根目录，包含 1000 部电影、6194 条三元组、12267 道问答。历史冻结快照 SHA-256：

`eb85268c3359238b7eeed2438e3693c5e42ab2de09ce4a32d4137a5196613739`。

重新在线采集会形成新的数据版本，不能将它当成原冻结实验。原流程为：

```powershell
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe -m kgqa.fetch --target 1000
.\.venv\Scripts\python.exe -m kgqa.prepare --snapshot .\data\raw\snapshot_1000.json
```

2026 年 10 月 29 日已在独立目录扩展到 **5000 部电影、35485 条三元组、58994 道问答**，并训练了配套 A0。原始数据、划分和模型保留；原 TransE/B2 没有与扩展库混用。来源、偏差、缓存、哈希、目录和复现命令见[电影数据集扩展记录](../docs/电影数据集扩展记录.md)。
