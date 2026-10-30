# 5000 部电影完整数据与使用说明

本版本随 Git 仓库提供完整冻结快照、处理数据及配套 A0，克隆后无需重新请求 Wikidata、训练模型或下载神经模型。共 **11 个资产文件、40,422,065 字节（38.55 MiB，十进制 40.42 MB）**，不计源代码、Git 历史和本页等说明文件。

## 直接使用

需要 Python 3.11；以下在 Windows PowerShell 的项目根目录运行，使用包含本数据版本的 `main`。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m kgqa.dataset activate --smoke
.\scripts\run_ui.cmd
```

打开 http://127.0.0.1:7860/ 后，可搜索电影、中文标签/别名及 QID，并输入英文问题。例如 `Who directed The Avengers?` 应回答 `The Avengers was directed by Joss Whedon.`。激活命令先核对清单中全部文件的大小、SHA-256、数据/模型配对与实际规模，并验证两道示例问答；成功后才写入本机 `.work/ui-dataset.json`。修改数据后应生成新版本，不能继续沿用本清单和旧模型。

只校验、不切换默认界面：

```powershell
.\.venv\Scripts\python.exe -m kgqa.dataset verify --smoke
```

Linux/macOS 下使用 `.venv/bin/python`；激活后用 `.venv/bin/python -m kgqa.web --open` 启动同一页面。可通过 `--manifest <清单路径>` 指定其他符合相同格式的数据版本。

## 数据、模型与清单

| 位置（相对项目根目录） | 内容 | 字节数 |
|---|---|---:|
| `data/raw/expansion_5000_20261029/snapshot_5000.json` | 完整冻结快照：电影、实体、事实、别名与来源元数据 | 8,940,504 |
| `data/processed/movies_5000_20261029/` | 实体、三元组、全部 58994 道 QA、划分及原处理 manifest | 27,724,804 |
| `models/pipeline_5000_20261029/` | A0 权重、配置、验证阈值、8917 道测试预测及指标 | 3,756,757 |
| [manifest.json](manifest.json) | 11 个文件的路径、大小、SHA-256、版本、来源、规模、环境与使用入口 | 清单自身不纳入哈希库存 |

5000 部电影对应 7864 个实体、35485 条三元组。按电影划分训练/验证/测试为 3500/750/750，问答为 41220/8857/8917，随机种子 42。清单保留每个文件的原始字节哈希；`.gitattributes` 对这些冻结资产关闭换行转换，保证不同系统检出后校验一致。

新分发清单提供全套文件库存与使用入口；原 `data/processed/.../manifest.json` 继续保留原提取时间、来源快照哈希与划分记录，两者用途不同。

## 来源与实验范围

Wikidata 结构化事实按 [CC0](https://www.wikidata.org/wiki/Wikidata:Licensing) 提供。保留历史 1000 部电影，新增 4000 部来自官方实体 API；新记录保留修订号与提取来源，这是明确记录的混合版本。采集原始 HTTP 缓存约 1.18 GiB，继续保留在采集者本地，未包含于本次分发；冻结快照、全部处理数据和可运行 A0 已包含。来源查询、采集统计与哈希见[扩展记录](../../docs/电影数据集扩展记录.md)及[来源摘要](../../verification/2026-10-29/expansion_summary.json)。

本版本仅包含配套 A0。原 1000 部电影及独立 200 部电影上的 TransE/B/C 不能与本版本新实体混用；其数据与神经权重不在这份分发中。本次 5000 部片库包含原电影，不是第二阶段计划中的独立 D2 数据集。模板测试的 100% 分数只说明固定模板和可查询冻结知识库的流程通过。
