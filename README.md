# Wikidata 电影知识图谱问答

《人工智能 I》课程项目：输入英文电影问题，识别片名、查询本地知识图谱，输出答案与事实证据。支持 **导演 P57、类型 P136、制作国家 P495、原始语言 P364** 四类单跳问题，同时评测别名、错拼、改写与拒答。

[快速启动](#快速启动-cpu-离线验收) · [浏览器界面](#浏览器问答界面) · [数据与进度](#数据版本与目前进度) · [两人分工](#双人分工与报告入口) · [人工评价](#人工评价填表与统计) · [完整实验](#完整数据与神经实验) · [文件索引](#仓库目录与文件索引)

| 系统 | 实现与比较目的 |
|---|---|
| A0 管道系统 | 实体链接 → TF-IDF 与逻辑回归关系分类 → 图谱查询 → 模板答案。 |
| B0–B3 神经系统 | FLAN-T5-small 接收事实文本及冻结的 TransE 向量投影成的结构前缀；比较学习率 `1e-4 / 3e-5 / 3e-4 / 1e-3`。 |
| C0 消融系统 | 使用事实文本，去掉结构前缀；学习率为 `1e-4`，与 B0 对照。 |

## 快速启动 CPU 离线验收

在解压或克隆后的项目根目录执行。以下为 **Windows PowerShell、Python 3.11** 命令；安装依赖需要网络，随后验收不下载数据或模型，也不需要 GPU。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m kgqa.acceptance --output-dir output/acceptance/review
.\.venv\Scripts\python.exe -m kgqa.demo --system a0 `
  --data-dir output/acceptance/review/data/processed `
  --model-path output/acceptance/review/runs/A0_acceptance/model.pkl `
  --question "Who directed Juniper Trail?"
```

验收成功时终端显示 `PASS synthetic offline acceptance`，演示显示电影、关系、事实证据与完整句答案。去掉 `--question` 及其内容可进入交互模式，输入 `exit` 退出。Linux/macOS 下将解释器路径换为 `.venv/bin/python`，并使用对应 shell 的换行方式。

验收使用 **10 部虚构电影、150 道题，电影划分 6/2/2、题目划分 90/30/30**，检查四关系、多答案、别名、拒答、模型重载及五组配置各三次推理。这是软件流程验收，不是 Wikidata 实验成绩。

| 生成位置 | 查看内容 |
|---|---|
| `output/acceptance/review/summary.json` | `status: PASS`、13 项检查，以及数据和模型的相对路径。 |
| `output/acceptance/review/examples.json` | 四关系、多答案、别名与拒答的实际输出。 |
| `output/acceptance/review/data/processed/` | 合成知识库、划分、问答与别名配对题。 |
| `output/acceptance/review/runs/A0_acceptance/` | 模型、验证选阈值、测试预测、指标与参数对照。 |

已有验收目录会被拒绝覆盖；再次运行请把 `review` 换成新名字，或省略 `--output-dir` 自动生成新目录。更多现场操作见 [提交验收说明](docs/提交验收说明.md)。

运行单元测试和手工链接回归：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe benchmarks/run_linker_benchmark.py --output output/verification
```

## 浏览器问答界面

项目共享界面支持搜索电影、别名和 QID，点击片名填入问题，切换 A0/神经系统或双系统对比，查看答案、实体链接、关系和图谱事实；也可复制答案、导出本次结果 JSON、重新填写最近提问。无需 Node.js 或额外 Web 框架。

完成上面的 CPU 离线验收后，可直接用其数据和模型打开 A0 页面：

```powershell
.\scripts\run_ui.cmd -DataDir output/acceptance/review/data/processed -ModelPath output/acceptance/review/runs/A0_acceptance/model.pkl
```

启动后自动打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)。保留启动终端，按 `Ctrl+C` 停止服务。该服务只绑定本机地址；这里使用的是合成电影，不能将页面结果写成 Wikidata 实验成绩。

已有真实实验数据与模型时，可以启动默认双系统界面、更换端口，或指定个人 A0 模型：

```powershell
.\scripts\run_ui.cmd
.\scripts\run_ui.cmd -Port 7861 -NoBrowser
.\scripts\run_ui.cmd -DataDir data/processed -ModelPath runs/A0_personal_20261019/model.pkl
```

默认读取 `data/processed`、`models/pipeline/model.pkl` 及 `output/selected_model.json` 指定的本地神经检查点。缺少神经模型仍可使用 A0；缺少数据或 A0 模型时页面给出提示。指定独立数据或模型路径时只开放 A0，避免与默认神经模型混用。

也可在相应 Python 环境中跨平台执行 `python -m kgqa.web --open`，使用 `--data-dir`、`--model-path`、`--port` 指定配置。页面不下载数据或模型，神经模型首次使用时才加载；首次处理耗时包含加载，UI 耗时只供交互观察。神经卡片展示的是输入候选事实，不代表每条都支持生成答案。

合并共享界面后的完整集成测试实测为 **220 项通过，14.93 秒，退出码 0**。当前 200 部电影数据的 `QAService` 已实际调用 A0 与 B0，均回答出《The Big Red One》的导演 Samuel Fuller；见 [集成测试日志](verification/2026-10-29/integration_pytest.log) 和 [真实双系统服务检查](verification/2026-10-29/ui_two_system_demo.json)。

## 数据版本与目前进度

| 数据与记录 | 用途和状态 | 入口 |
|---|---|---|
| 历史 1000 部 Wikidata 电影 | 2285 实体、6194 三元组、12267 问答；保留 A0、B0–B3、C0 的第一阶段结果。 | [历史结果摘要](提交材料/实验结果摘要.md) |
| 新 200 部电影，10 月 19 日与 29 日 | 552 实体、1264 三元组、2492 问答；电影 140/30/30，问答 1749/373/370。保留原 A0 个人实验，现已补齐同快照 A0、TransE、B0–B3、C0、选模及鲁棒性。 | [原 A0 证据](verification/2026-10-19/personal/) · [10 月 29 日完整证据](verification/2026-10-29/README.md) |
| 2026-10-24 合成 10 部电影 | 独立的 CPU 验收夹具，不与前两组实验合并计分。 | [验收记录](verification/2026-10-24/README.md) |

新 200 部电影的候选来自缓存的 **1980—1989 发行日期区间**查询，不保证每部电影最早上映年份属于该区间。快照 SHA-256 为 `5b02443928ef5503c280c9422376cf5efb8849f13e02ee92f03aca917c47ed38`。完整数据、模型和原始运行目录不随代码仓库分发；在线重新采集会产生新版本。

新 200 部电影的 A0 测试为 370/370；遮盖片名后的测试题全部属于训练集已有的 15 种问题模板，推理可访问完整本地知识库。改写为 75/100、错拼为 85/100、别名为 47/47，别名仅与同源 47 道原题比较。阈值 0.30 由验证集选出；0.30、0.70、0.90、0.99 为固定诊断配置，各三次推理属于同一模型重复推理。以上结果说明受控模板流程的表现，不代表开放域问答能力。

2026-10-24 的记录包含 181 项通过的测试与离线验收；既有 10 月 19 日模型在新链接代码下回归，370 道测试、381 条鲁棒性题的非时间指标与逐题输出均保持一致，未重训。[查看正式数据回归](verification/2026-10-24/formal_regression.json)。

### 10 月 29 日同快照完整实验

已在 RTX 3050 Laptop GPU 4 GB 上完成 100 轮 TransE，以及 B0–B3、C0 各 5 轮训练。神经训练使用随机种子 42、batch 4、梯度累积 4、bf16，名义有效 batch 为 16；A0 在同一冻结数据上重新拟合。全部系统覆盖 370 道测试题，配置、训练曲线、逐题预测和指标见 [真实运行汇总](verification/2026-10-29/neural_summary.json)。

| 系统 | 学习率 | 验证实体 Macro-F1 % | 测试集合准确率 % | 测试实体 Macro-F1 % |
|---|---|---:|---:|---:|
| A0 | — | 100.00 | 100.00 | 100.00 |
| B0 | 1e-4 | 99.20 | 98.65 | 98.86 |
| B1 | 3e-5 | 97.03 | 97.84 | 98.27 |
| B2 | 3e-4 | 99.20 | 98.92 | 99.10 |
| B3 | 1e-3 | 99.20 | 98.65 | 99.01 |
| C0 | 1e-4 | 99.20 | 98.92 | 98.92 |

B0、B2、B3 的验证 F1 同为 99.195710%，按预定的较小编号规则选中 **B0**；没有按测试集改选 B2。C0 是消融对照，不参与 B0–B3 的选模。B0 的测试 F1 略低于 C0，本次单种子结果未显示结构前缀带来提升。

选中 B0 的鲁棒性集合准确率为：原题 99/100、改写 88/100、错拼 88/100、别名 47/47、拒答 34/34。各扰动与其同源原题配对；完整分母和 A0/B0 对比见 [配对结果](verification/2026-10-29/paired_system_comparison.json)。神经延迟为批量生成按题分摊，A0 为单题完整管道耗时，不能直接相除为端到端加速倍数。合并界面前的 [个人部分 207 项测试及失败即停记录](verification/2026-10-29/local_checks.json) 保留原值；合并界面后的完整测试为 220 项，见 [集成检查](verification/2026-10-29/upstream_ui_integration.json)。

## 双人分工与报告入口

| 成员 | 本轮负责范围 | 文档 |
|---|---|---|
| 周新永 A | 实体链接、A0 问答与评测、参数与鲁棒性分析、离线验收。 | [个人实验报告](reports/个人实验报告.docx) · [提交验收说明](docs/提交验收说明.md) |
| 夏俊杰 B | 数据版本复核、TransE、FLAN-T5、学习率实验与 C0 消融，整理个人贡献及证据。 | [夏俊杰实验报告](reports/夏俊杰_实验报告.docx) |
| 两人共同 | 在同一数据版本上比较系统、独立人工评分、核对差异与答辩。 | [分工计划](reports/双人分工与复现实验计划.docx) · [提交审查](reports/提交审查.md) |

历史仓库作为共同基础，不按新分工追认历史独立贡献。课程提交仍需本人补齐真实终端环境截图、至少两张运行截图，以及两人的独立人工评分；报告身份字段和完成日期按实际情况填写。人工评分不能由自动指标替代。

## 人工评价填表与统计

本轮已从同一数据版本的 A0 与 B0 预测中抽取 100 道题，覆盖 30 部电影、每关系 25 题。两份独立空表各含 200 条匿名答案，已准备于 [人工评价目录](reports/人工评价/)：

- [周新永评分表](reports/人工评价/周新永_评分表.csv)
- [夏俊杰评分表](reports/人工评价/夏俊杰_评分表.csv)
- [填写说明](reports/人工评价/填写说明.md)

各人下载或复制自己的表独立填写。若更换数据或模型版本，先另存已有评分，再用同源的共享 A0 和选中神经预测重新导出；不要将新答案套入本轮旧表：

```powershell
$SelectedNeuralRun = (Get-Content output/selected_model.json -Raw | ConvertFrom-Json).selected_run
.\.venv\Scripts\python.exe -m kgqa.human_eval --neural-run $SelectedNeuralRun --limit 100 --seed 42
Copy-Item output/human_evaluation.csv reports/人工评价/周新永_评分表.csv
Copy-Item output/human_evaluation.csv reports/人工评价/夏俊杰_评分表.csv
Copy-Item output/human_evaluation_key.csv reports/人工评价/对照密钥.csv
```

重新导出时，评分表、密钥与 `output/human_evaluation_manifest.json` 必须作为同一次抽样共同保存；不要混用其他版本的密钥。

两人分别填写自己的文件，只修改四个评分列与 `notes`，保留题号、答案位置、原题、参考答案和答案文本。准确性 `accuracy_0_2`、完整性 `completeness_0_2` 为 0–2，流畅度 `fluency_1_5` 为 1–5，无依据断言 `unsupported_claim_0_1` 为 0/1。**0 是真实评分，空白表示未评。** 组织者保管 key 与 manifest 中的系统映射，独立评分结束后再解码和讨论。

两份表真实填完后执行：

```powershell
.\.venv\Scripts\python.exe -m kgqa.human_eval `
  --score-file reports/人工评价/周新永_评分表.csv `
  --score-file reports/人工评价/夏俊杰_评分表.csv `
  --key-file reports/人工评价/对照密钥.csv `
  --summary-output verification/2026-10-29/human_evaluation_statistics.json
```

统计要求至少两份不同文件，每份与 key 的全部 `(question_id, answer_position)` 一一对应；重复、未知、缺失行以及空白、非整数或越界评分会报错。JSON 按 key 的系统标签汇总题数、评分数、四项均值与无依据断言百分比，并记录两人逐项一致率和分歧数量。原始 CSV 保持不变，讨论后结果另存。目前人工分数仍待两人实际填写。

## 完整数据与神经实验

### 准备完整环境

完整依赖使用 PyTorch 2.6.0 的 CUDA 12.4 构建，并下载 FLAN-T5-small。GPU 训练需要相容的 NVIDIA 显卡和驱动；无可用 CUDA 时神经代码会使用 CPU，训练速度会明显降低。

在项目根目录准备环境；若已按快速启动创建 `.venv`，直接安装完整依赖即可：

```powershell
python -m venv .venv
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`environment.ps1` 把下载缓存和临时文件放入项目的 `.work/`。如果当前 PowerShell 禁止执行脚本，可仅对当前进程执行 `Set-ExecutionPolicy -Scope Process Bypass` 后再运行；演示的 `.cmd` 入口已为其子进程设置此选项。

### 先小规模再完整运行

**先备份现有 `data/raw/`、`data/processed/`、`models/`、`runs/`、`output/`，或使用另一份项目目录。** 完整脚本会采集或复用在线缓存、重建数据，并写入固定的 A0、B0–B3、C0 目录；smoke 与 full 应分开保存，避免覆盖既有数据、模型和结果。

在独立的小规模实验目录中执行：

```powershell
.\scripts\run_experiment.ps1 -Mode smoke -Target 50 -BatchSize 4 -GradientAccumulation 4
```

环境与流程确认后，在备份后的独立完整实验目录中执行：

```powershell
.\scripts\run_experiment.ps1 -Mode full -Target 200 -BatchSize 4 -GradientAccumulation 4
```

脚本依次执行 Wikidata 采集、问答构建、A0 训练与测试、TransE、鲁棒性题生成、B0–B3 与 C0 训练，再按验证集选神经模型、生成评价表及图表。smoke 使用 2 轮 TransE，神经训练每个划分最多 32 题、1 轮；full 使用 100 轮 TransE、各神经配置默认 5 轮。smoke 输出只用于流程检查。在线采集和模型下载需要网络，运行时长取决于数据、设备与下载条件。

`-Target 200` 对应本轮规模，但在线重采集不能保证得到相同电影或相同哈希；精确复核应恢复已留存的冻结快照及处理数据，核对 [source_manifest.json](verification/2026-10-29/source_manifest.json)，再从训练阶段逐模块执行，避免采集和预处理改写数据。将 `-Target` 改为 `1000` 可开展历史规模的新实验，同样会生成新快照，不是恢复本轮 200 部电影的命令。

上述示例使用本轮实际训练配置 batch 4、梯度累积 4，名义有效训练 batch 为 16。`-BatchSize` 传入神经训练过程的训练、验证及测试批大小，`-GradientAccumulation` 控制训练更新前累积的微批次数；省略时分别为 16 和 1。显存更小时可改为 2/8，但这是另一运行配置，需要单独记录；相同有效 batch 不等于逐字节相同的实验。比较学习率与消融时各组应使用同一设置，并记录实际批大小与延迟口径。任一 Python 步骤以非零状态退出时，脚本会立即停止。

| 输出目录 | 主要产物 |
|---|---|
| `data/raw/`、`data/processed/` | 原始响应、快照、知识图谱、电影划分与 QA。 |
| `models/pipeline/`、`models/transe/` | 共享 A0 模型；TransE 嵌入、映射、模型与损失记录。 |
| `runs/A0/`、`runs/B0/`–`runs/B3/`、`runs/C0/` | 配置、日志、验证历史、检查点、逐题预测与测试指标。 |
| `output/selected_model.json`、`output/figures/` | 验证集选模结果与比较图表。 |
| `output/human_evaluation*.csv`、`output/human_evaluation_manifest.json` | 盲评表、单独的系统密钥及抽样清单；评分列留空。 |
| `output/environment.json` | 该次实际软硬件环境记录。 |

已有完整实验产物后，可补充错误分析、记录摘要与报告草稿：

```powershell
.\.venv\Scripts\python.exe -m kgqa.error_analysis
.\.venv\Scripts\python.exe -m kgqa.evidence
.\.venv\Scripts\python.exe -m kgqa.report
```

对应输出为 `output/error_analysis.json`、`output/evidence/run_summary.json` 与 `.txt`、`output/人工智能I_实验报告.docx`。报告生成器读取共享 A0/B/C 产物，不会更新 `reports/` 的个人报告，也不会完成人工评分或真实截图。

### 已有真实数据时单独复核 A0

确认 `data/processed/` 与待复核模型属于同一数据版本后执行；新运行名可避免覆盖历史目录：

```powershell
.\.venv\Scripts\python.exe -m kgqa.robustness --limit 100 --seed 42
.\.venv\Scripts\python.exe -m kgqa.personal_experiment --run-id A0_personal_review
.\.venv\Scripts\python.exe -m kgqa.demo --system a0 `
  --data-dir data/processed --model-path runs/A0_personal_review/model.pkl
```

个人运行保存于 `runs/A0_personal_review/`，不覆盖共享 `models/pipeline/`；已有同名运行会拒绝覆盖。`robustness` 会重写当前处理数据中的鲁棒性文件，应先保存原版本。恢复历史快照与在线生成新数据的区别见 [数据说明](data/README.md) 和 [复现指南](docs/双人分工与复现指南.md)。

默认双系统演示需要共享数据、`models/pipeline/model.pkl`、神经检查点及 `output/selected_model.json`：

```powershell
.\scripts\run_demo.cmd -System both -Question "Who directed The Big Red One?"
```

片名需存在于当前知识库，示例是否可答取决于所用快照。`-System a0` 只加载 A0；`-System best` 只显示验证选中的神经系统，但现有演示初始化仍需要共享 A0 模型。自定义 `--data-dir` / `--model-path` 只适用于 `--system a0`。

本轮已实际启动 A0/B0 双系统并回答上述问题，见 [双系统演示日志](verification/2026-10-29/two_system_demo.log)。[演示与评分入口检查](verification/2026-10-29/demo_and_rating_checks.json) 同时记录：未填写的两张评分表被统计 CLI 拒绝，未生成任何人工统计成绩。

## 仓库目录与文件索引

### 根目录

| 文件或目录 | 用途 |
|---|---|
| [README.md](README.md) | 项目介绍、启动命令与完整文件导航。 |
| [requirements-test.txt](requirements-test.txt) | CPU 测试、链接回归、离线验收与 A0 演示的最小固定依赖。 |
| [requirements-ci.txt](requirements-ci.txt) | CI 核心依赖，另含 Ruff；不安装神经训练库。 |
| [requirements.txt](requirements.txt) | 完整训练、绘图及报告依赖，含 CUDA PyTorch、Transformers 和 PyKEEN。 |
| [.gitignore](.gitignore) | 排除环境、缓存、完整数据、模型、运行和自动输出，保留选定的历史提交材料。 |
| [.github/](.github/) | [ci.yml](.github/workflows/ci.yml)：Ubuntu/Windows、Python 3.11 的语法、致命错误、CLI（含 `kgqa.web --help`）、测试、离线验收及 PowerShell 语法检查。 |
| [kgqa/](kgqa/) | 数据、问答、训练、评价与报告的 Python 包，以及 [ui/](kgqa/ui/) 页面资源，逐文件见下表。 |
| [scripts/](scripts/) | Windows 环境设置与实验、演示启动脚本。 |
| [tests/](tests/) | 自动测试，逐文件见下表。 |
| [benchmarks/](benchmarks/) | 冻结旧链接器与手工回归对照脚本。 |
| [data/](data/) | [README.md](data/README.md) 说明数据重建和历史快照；实际 raw/processed 文件在本地生成。 |
| [docs/](docs/) | [双人分工与复现指南](docs/双人分工与复现指南.md)、[提交验收说明](docs/提交验收说明.md)。 |
| [reports/](reports/) | [README.md](reports/README.md) 为报告说明；[周新永个人报告](reports/个人实验报告.docx) 保留原路径；另有[夏俊杰个人报告](reports/夏俊杰_实验报告.docx)、[双人分工计划](reports/双人分工与复现实验计划.docx)、[提交审查](reports/提交审查.md) 和[本轮双人空白评分表](reports/人工评价/填写说明.md)。 |
| [verification/](verification/) | 按日期保存的实际测试、采集、运行摘要与回归证据，命名见后表。 |
| [提交材料/](提交材料/) | 历史[报告 Word](提交材料/人工智能I_实验报告.docx)、[报告 PDF](提交材料/人工智能I_实验报告.pdf)、[结果摘要](提交材料/实验结果摘要.md)、[待填写人工评价表](提交材料/人工评价表_待填写.csv)。 |
| [人工智能I-课程实验.pdf](人工智能I-课程实验.pdf) | 教师的课程实验说明。 |
| [人工智能I_实验报告要求.docx](人工智能I_实验报告要求.docx) | 教师的报告内容与提交要求。 |
| [实验设计与参考资料.md](实验设计与参考资料.md) | 第一阶段设计路线、实验设置和参考资料。 |
| [项目实施总结与后续改进.md](项目实施总结与后续改进.md) | 第一阶段实现、结果及改进方向。 |
| [第二阶段实验详细设计.md](第二阶段实验详细设计.md) | 第二阶段模型、融合方法与对照设计，属于计划。 |

本地运行还会产生以下目录：

| 目录 | 用途 |
|---|---|
| `.venv/`、`.work/` | Python 环境；下载、模型与临时缓存。 |
| `data/raw/` | 请求缓存、候选片名 ID 和 `snapshot_<数量>.json`；`--candidate-cache` 可读原采集器保存的 SPARQL 响应。 |
| `data/processed/` | `entities.json` 实体与事实；`splits.json` 电影划分；`qa.jsonl` 问答；`triples.tsv` 三元组；`manifest.json` 来源与统计；`robustness.jsonl` 及 `robustness_manifest.json` 为扰动题和配对清单。 |
| `models/`、`runs/`、`output/` | 共享模型、逐次实验、图表/盲评/验收等产物，具体路径见运行说明。 |
| `.pytest_cache/`、`__pycache__/` | 测试与 Python 字节码缓存，可重新生成。 |
| `.git/` | Git 克隆的版本管理目录，下载 ZIP 时不包含。 |

### kgqa 模块

| 文件 | 职责 |
|---|---|
| [__init__.py](kgqa/__init__.py) | Python 包入口。 |
| [common.py](kgqa/common.py) | 项目根目录、四关系常量、JSON/JSONL、日志、哈希与答案模板。 |
| [fetch.py](kgqa/fetch.py) | Wikidata 候选查询、实体请求缓存、事实筛选和快照；支持 `--target`、`--candidate-cache`。 |
| [prepare.py](kgqa/prepare.py) | 按电影划分 train/valid/test，生成 QA、三元组与数据清单；入口 `--snapshot`。 |
| [linker.py](kgqa/linker.py) | 标题/别名精确匹配、片段模糊匹配、歧义拒答和片名遮蔽。 |
| [pipeline.py](kgqa/pipeline.py) | A0 分类器训练、加载、图谱查询与答案证据；CLI 为 `train / predict / robust`。 |
| [personal_experiment.py](kgqa/personal_experiment.py) | 隔离的 A0 运行器：输入校验、训练、验证选阈值、测试、重复推理与配对鲁棒性。 |
| [acceptance.py](kgqa/acceptance.py) | 构造合成小知识库，端到端运行 A0 并写出验收检查和示例。 |
| [demo.py](kgqa/demo.py) | 交互或单题演示；`--system a0 / best / both`，A0 支持指定数据和模型路径。 |
| [web.py](kgqa/web.py) | 用标准库提供本机 HTTP 页面与状态、电影搜索、问答 API；延迟加载模型，支持 A0 或双系统及自定义 A0 路径。 |
| [transe.py](kgqa/transe.py) | 用三元组训练 TransE，保存嵌入、实体/关系映射与损失曲线。 |
| [neural.py](kgqa/neural.py) | 图谱事实与结构前缀、FLAN-T5 训练、验证选检查点、测试及指定文件推理；训练支持 `--batch-size` 和 `--gradient-accumulation`。 |
| [metrics.py](kgqa/metrics.py) | 题号和覆盖校验、实体集合准确率/Macro-F1、BLEU、ROUGE-L、拒答及延迟；按 run 评估。 |
| [robustness.py](kgqa/robustness.py) | 从测试集抽样生成原题、改写、错拼、别名和拒答题，保存来源与配对分母。 |
| [human_eval.py](kgqa/human_eval.py) | 对齐 A0 与神经 run，导出盲评表、密钥与抽样清单；读取多份独立评分 CSV，校验覆盖/分数并统计均值、一致率和分歧。 |
| [finalize.py](kgqa/finalize.py) | 从 B0–B3 的验证指标选最佳系统，执行其鲁棒性推理、评价及盲评导出。 |
| [error_analysis.py](kgqa/error_analysis.py) | 对共享 A0 和选中神经系统统计链接、关系、拒答、漏答和额外实体等错误。 |
| [plots.py](kgqa/plots.py) | 读取已保存结果，绘制学习率、系统指标、结构消融和鲁棒性比较图。 |
| [environment_info.py](kgqa/environment_info.py) | 采集完整实验的 Python、依赖、CPU、内存和 NVIDIA GPU 环境。 |
| [evidence.py](kgqa/evidence.py) | 将已有环境、训练及指标记录汇总为 JSON/TXT；不生成终端截图。 |
| [report.py](kgqa/report.py) | 从共享实验产物、图表与已有截图生成团队 Word 报告草稿。 |

### 界面资源

| 文件 | 职责 |
|---|---|
| [index.html](kgqa/ui/index.html) | 中文问答页结构：知识库检索、问题输入、系统选择和答案/证据区域。 |
| [app.js](kgqa/ui/app.js) | 调用本机 API，处理电影检索、提问、答案展示、复制/JSON 导出和本机最近提问记录。 |
| [styles.css](kgqa/ui/styles.css) | 页面布局、颜色、组件、响应式与交互状态样式。 |

### 脚本与测试

| 脚本 | 用途 |
|---|---|
| [environment.ps1](scripts/environment.ps1) | 设置项目内缓存、临时目录和 UTF-8 环境变量。 |
| [run_experiment.ps1](scripts/run_experiment.ps1) | `-Mode smoke/full -Target 数量 -BatchSize 批大小 -GradientAccumulation 累积次数`；串行执行，步骤失败即停。 |
| [run_demo.ps1](scripts/run_demo.ps1) | 演示包装器，接受 `-System`、`-Question`、`-DataDir`、`-ModelPath`。 |
| [run_demo.cmd](scripts/run_demo.cmd) | 调用 PowerShell 演示脚本并透传参数和退出码。 |
| [run_ui.ps1](scripts/run_ui.ps1) | 浏览器界面启动器，使用 `.venv`，支持 `-Port`、`-DataDir`、`-ModelPath`、`-NoBrowser`。 |
| [run_ui.cmd](scripts/run_ui.cmd) | 调用 UI PowerShell 启动器，传递参数与退出码；默认自动打开本机页面。 |

| 测试文件 | 验证内容 |
|---|---|
| [test_core.py](tests/test_core.py) | 数据划分、QA 模板、基本实体链接、指标及证据行为。 |
| [test_acceptance.py](tests/test_acceptance.py) | 无网络/神经依赖的隔离验收、产物、失败状态和 CLI。 |
| [test_demo.py](tests/test_demo.py) | A0 演示无需神经组件、自定义数据与模型，以及路径参数约束。 |
| [test_evidence.py](tests/test_evidence.py) | 保存记录摘要和缺失项，确认不生成模拟终端图片。 |
| [test_fetch.py](tests/test_fetch.py) | 候选缓存读取、格式与来源、目标数量及采集批次。 |
| [test_human_eval.py](tests/test_human_eval.py) | 盲评抽样、预测对齐、空表导出，以及真实 0 分、完整评分、错误 ID/分数拒绝、系统均值与评审一致率。 |
| [test_linker.py](tests/test_linker.py) | 标题、别名、错拼、嵌套片名、未知数字续集、遮蔽与管道拒答。 |
| [test_metrics.py](tests/test_metrics.py) | ID 唯一性、完整覆盖、实体抽取、分组指标与 smoke/鲁棒性评估。 |
| [test_personal_experiment.py](tests/test_personal_experiment.py) | 个人运行输入、划分、验证选阈值、重复推理、配对与拒绝覆盖。 |
| [test_robustness.py](tests/test_robustness.py) | 扰动生成、标题边界、抽样均衡、来源配对与分母记录。 |
| [test_web.py](tests/test_web.py) | 合成数据上的真实 A0 HTTP 问答、状态/搜索、输入错误、请求边界、忙碌/失败处理及静态资源范围。 |

### benchmarks 与 verification

| 入口 | 用途 |
|---|---|
| [legacy_linker.py](benchmarks/legacy_linker.py) | 冻结自提交 `9b85f49` 的旧链接器，只供对照。 |
| [run_linker_benchmark.py](benchmarks/run_linker_benchmark.py) | 对 10 个手工实体的 44 条查询进行新旧链接与阈值对照；每设置三次，不训练模型。 |
| [verification/2026-10-18/](verification/2026-10-18/) | 首轮手工链接回归与测试：`fixture.json` 为用例，`linker_benchmark.json` 为汇总，`predictions.csv` 为逐题结果，`trials.csv` 为各次参数对照；`benchmark.log`、`pytest.log` 为日志。 |
| [verification/2026-10-19/](verification/2026-10-19/) | 新 200 部电影采集与个人实验。`acquisition.log` / `acquisition_provenance.json` 记录来源；`data_manifest.json` / `data_robustness_manifest.json` 记录数据和扰动分布。 |
| `2026-10-19/linker_*`、`pytest.log` | 当日链接回归的用例、汇总、逐题预测、三次参数表与日志；`linker_linker_benchmark.json` 是该次保留的汇总文件名。 |
| [2026-10-19/personal/](verification/2026-10-19/personal/) | `config.json`、`environment.json`、`hardware_observation.json` 记录配置和环境；`data_manifest.json`、`source_hashes.json`、`status.json` 记录输入、源码版本与运行状态。 |
| `personal/validation_thresholds.json`、`predictions_*.jsonl`、`metrics_*.json` | 验证阈值选择、完整逐题预测和测试/鲁棒性指标；`metrics_robustness_matched.json`、`robustness_summary.json` 保存配对比较与各组分母。 |
| `personal/sensitivity_summary.json`、`sensitivity_comparison.csv` | 五配置三次推理的汇总与表格。`sensitivity/<baseline 或 threshold_百分数>/repeat_<1–3>/metrics_test.json` 为各次指标，避免逐一列出重复路径。 |
| [verification/2026-10-24/](verification/2026-10-24/README.md) | `commands.json` 为命令/退出码；`pytest.log`、`syntax.log`、`ruff.log`、`powershell.log` 为自动检查，成功的编译日志可为空。 |
| `2026-10-24/acceptance*`、`demo.log` | 离线验收日志、环境、汇总、示例和指定模型演示；只保存摘要，不附整套生成数据/模型。 |
| [2026-10-24/formal_regression.json](verification/2026-10-24/formal_regression.json) | 当前代码加载 10 月 19 日旧模型的只评估回归，记录数据与源码哈希及结果对照。 |
| [verification/2026-10-29/](verification/2026-10-29/README.md) | 同一冻结 200 部电影的完整实验归档；[neural_summary.json](verification/2026-10-29/neural_summary.json) 为真实训练与评价汇总，[local_checks.json](verification/2026-10-29/local_checks.json) 为合并界面前个人部分 207 项测试及失败即停记录。 |
| `2026-10-29/runs/<A0、B0–B3、C0>/` | A0 参数对照，以及各神经 run 的 `config.json`、`history.json`、`selection.json`、`status.json`、验证/测试逐题预测和指标；鲁棒性记录保存于实际评价的 A0、B0。 |
| `2026-10-29/data/`、`transe/`、`logs/` | 数据统计与鲁棒性配对清单；TransE 配置、100 轮损失与曲线；实际运行日志。完整数据与神经权重不在归档中。 |
| `2026-10-29/output/` | 验证选模、六系统指标、错误分析、人工评价抽样清单及 [比较图表](verification/2026-10-29/output/figures/)。 |
| `2026-10-29/source_manifest.json`、`collection_manifest.json`、`model_revision.json`、`environment_freeze.txt`、`requirements.txt`、`suite_status.json` | 源码与冻结数据哈希、收集前后校验、预训练模型修订、实际依赖及套件完成状态。文本中的原运行路径已转为相对说明。 |
| [two_system_demo.log](verification/2026-10-29/two_system_demo.log)、[demo_and_rating_checks.json](verification/2026-10-29/demo_and_rating_checks.json) | 已安装的真实 A0/B0 双系统演示，以及空白评分表被正确拒绝的退出码记录。 |
| [integration_pytest.log](verification/2026-10-29/integration_pytest.log)、[upstream_ui_integration.json](verification/2026-10-29/upstream_ui_integration.json)、[ui_two_system_demo.json](verification/2026-10-29/ui_two_system_demo.json) | 合并共享界面后的完整 220 项测试、集成来源及真实 200 部电影上 A0/B0 的服务层双系统问答验证。 |
| [reports/人工评价/](reports/人工评价/) | `周新永_评分表.csv`、`夏俊杰_评分表.csv`、`对照密钥.csv` 与 `填写说明.md`。当前评分为空；真实填完后才生成 `human_evaluation_statistics.json`。 |

真实截图、人工评价、数据版本复核与最终可提交项集中见 [提交审查](reports/提交审查.md)。
