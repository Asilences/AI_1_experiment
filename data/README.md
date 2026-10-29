# 数据目录与版本

仓库现在提供 **5000 部电影的完整冻结快照、处理数据和配套 A0 模型**，共 38.55 MiB。[分发说明与 manifest](../datasets/movies-5000-20261029/README.md)列出全部文件、SHA-256 和启动步骤；安装 `requirements-test.txt` 后运行 `python -m kgqa.dataset activate --smoke` 即可校验并激活。历史 1000 部及独立 200 部实验的完整数据、神经权重仍需单独恢复。

只检查软件流程时，可运行 `python -m kgqa.acceptance` 生成独立的小型虚构夹具；该夹具仅用于代码验收，操作见[提交验收说明](../docs/提交验收说明.md)。

原实验数据位于 `data/raw/snapshot_1000.json` 和 `data/processed` 根目录，包含 1000 部电影、6194 条三元组、12267 道问答。历史冻结快照 SHA-256：

`eb85268c3359238b7eeed2438e3693c5e42ab2de09ce4a32d4137a5196613739`。

仓库另有独立的 200 部电影完整实验，冻结快照 SHA-256 为 `5b02443928ef5503c280c9422376cf5efb8849f13e02ee92f03aca917c47ed38`，数据规模及配套 A0/TransE/B/C 记录见[实验归档](../verification/2026-10-29/README.md)。`data/processed` 是默认运行路径，不绑定特定规模；恢复某一版本时，必须同时恢复其数据、模型及选模记录。5000 部扩展库使用单独目录和配套 A0，不改变该 200 部实验的结果。

重新在线采集会形成新的数据版本，不能将它当成原冻结实验。原流程为：

```powershell
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe -m kgqa.fetch --target 1000
.\.venv\Scripts\python.exe -m kgqa.prepare --snapshot .\data\raw\snapshot_1000.json
```

2026 年 10 月 29 日已在独立目录扩展到 **5000 部电影、35485 条三元组、58994 道问答**，并训练了配套 A0。原始数据、划分和模型保留；原 TransE/B2 没有与扩展库混用。来源、偏差、缓存、哈希、目录和复现命令见[电影数据集扩展记录](../docs/电影数据集扩展记录.md)。
