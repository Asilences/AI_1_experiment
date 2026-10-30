# Wikidata 电影知识图谱问答

《人工智能 I》课程项目：输入英文电影问题，识别电影实体，查询冻结知识图谱并返回答案与事实证据。支持导演P57、类型P136、制作国家P495、原始语言P364四类单跳问题，以及别名、错拼、改写和拒答诊断。

**本轮主实验为5000部电影，五组神经实验已全部完成，原始验证选中B0。** 仓库随附完整冻结快照、7864个实体、35485条三元组、58994道问答及配套A0，可直接校验并启动管道问答。完整配置、逐题预测、曲线与指标见[本轮实验汇总](verification/2026-10-30/neural_summary.json)；双系统完整运行包见[验收包发布入口](https://github.com/Asilences/AI_1_experiment/releases/tag/movies-5000-20261030)。

[源码快速启动](#源码快速启动5000部电影a0) · [双系统运行包](#独立双系统运行包) · [完整实验复现](#完整实验复现) · [实际结果与名称恢复](#实际结果评分与名称恢复) · [报告与人工评价](#报告与人工评价) · [代码检查](#代码检查) · [文件索引](#文件索引)

## 源码快速启动：5000部电影A0

克隆仓库或通过GitHub **Code → Download ZIP** 下载源码，进入包含`kgqa`的根目录。Windows PowerShell、Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m kgqa.dataset activate --smoke
.\.venv\Scripts\python.exe -m kgqa.web --open
```

依赖安装需要网络，随后A0使用本地数据与模型，无需重新采集或训练。`activate`先核对文件大小、SHA-256、数据/模型配对及示例回答，成功后写入本机`.work/ui-dataset.json`。**此激活入口为5000部电影的A0模式。** 只验证资产、不切换界面时执行`python -m kgqa.dataset verify --smoke`。来源、逐文件清单与使用说明见[数据说明](datasets/movies-5000-20261029/README.md)和[manifest](datasets/movies-5000-20261029/manifest.json)。

浏览器访问[http://127.0.0.1:7860/](http://127.0.0.1:7860/)，可搜索片名、中文标签/别名和QID，选择示例英文问题，查看实体链接、关系与事实证据，复制答案或导出本次结果JSON。保留启动终端，按`Ctrl+C`停止；端口占用时使用`--port 7861`。Windows也可使用`scripts/run_ui.cmd`；其他平台将解释器路径换为`.venv/bin/python`。

## 独立双系统运行包

[验收包发布入口](https://github.com/Asilences/AI_1_experiment/releases/tag/movies-5000-20261030)用于分发独立运行包，内容为同一5000部电影数据、配套A0、TransE和验证选中的B0权重。下载该包后无需重新下载基础模型或训练，依赖安装仍需网络。

解压后进入运行包根目录，使用完整环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime-cpu.txt
.\.venv\Scripts\python.exe -m kgqa.web --legacy --open
```

`--legacy`绕过已激活的A0专用配置，读取运行包默认的`data/processed`、`models/pipeline`和`output/selected_model.json`；它是路径选择参数，不表示返回旧电影版本。源码随附的发布数据在独立版本目录，不能把源码克隆直接当作含默认路径和神经权重的运行包。自定义`--data-dir`或`--model-path`会进入仅A0模式，不用于双系统验收。

页面可切换A0、神经模型和双系统，首次神经回答加载本地检查点。上述CPU部署已在独立Python 3.12、PyTorch 2.6.0 CPU环境完成四项离线验收。需要GPU时，另建环境安装`requirements.txt`（CUDA 12.4）并使用相同启动命令；完整神经训练也使用该完整依赖。两种PyTorch构建分别安装在独立环境中。可选附加方法启动为：

```powershell
.\.venv\Scripts\python.exe -m kgqa.web --legacy --recover-names --open
```

启用后标注“+ 名称恢复”，保留`raw_answer`、替换证据及独立恢复耗时。恢复不重新训练、不补漏答，不替代原始指标或原始验证选模。神经卡片中的事实是输入候选，不保证每条都支持生成答案；界面响应时间也不等于正式生成延迟。

现场核对`GET /api/status`的`movie_count=5000`、`custom_data=false`、A0/神经可用状态和实际`run_id`；再向`POST /api/answer`发送`{"question":"实际片库中的英文问题","system":"both"}`，逐项检查`a0`、`best`结果。完整PowerShell示例、截图与提交要求见[提交验收说明](docs/提交验收说明.md)。

## 完整实验复现

| 系统 | 本轮方法 | 学习率 |
|---|---|---:|
| A0 | 冻结实体链接、TF-IDF/逻辑回归关系分类、图谱查询与模板答案；阈值0.30 | 不重新拟合 |
| B0 | FLAN-T5-small + 冻结TransE经可训练投影与归一化形成的结构前缀 | 1e-4 |
| B1 | 同B0 | 3e-5 |
| B2 | 同B0 | 3e-4 |
| B3 | 同B0 | 1e-3 |
| C0 | 保留相同事实文本，移除结构前缀，与原始B0作消融比较 | 1e-4 |

冻结快照SHA-256为`486faa982e84fd5781f62872fd02db11100fad82a013b37f454bbe4da520d867`。电影划分3500/750/750，问答划分41220/8857/8917；同一电影不跨集合。各组推理可查询同一完整知识库，各划分共享模板，因此本实验不是未知边预测或开放域问答。

先安装`requirements.txt`，准备[Google FLAN-T5-small模型仓库](https://huggingface.co/google/flan-t5-small)的完整本地缓存。本轮固定[修订0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab](https://huggingface.co/google/flan-t5-small/tree/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab)，来源记录见[model_revision.json](verification/2026-10-29/model_revision.json)。将下面占位内容换为实际缓存目录，在源码根目录执行：

```powershell
$pretrained = '<本地完整预训练缓存目录>'
.\.venv\Scripts\python.exe -m kgqa.expanded_suite --work-dir output/expanded-neural --model-name $pretrained --batch-size 8 --eval-batch-size 64 --input-max-length 512
```

runner在新工作目录建立隔离`project`，复制冻结数据/A0并记录哈希。TransE训练100轮、维度100；五组神经配置各训练5轮，从同一预训练来源独立开始。训练batch8、累积2，名义有效batch16；输入512、目标/生成上限96、最多24条事实，seed42及独立shuffle生成器42。训练BF16，生成float32、greedy，评价batch64，CPU线程4。

每轮完整验证8857题，实体F1严格提升才更新检查点，轮次同分保留较早轮。B0—B3按原始验证F1选模型，同分取较小B编号，C0不参与选模；各组恢复自己的最佳检查点后完整测试8917题。测试成绩不反选学习率。`--resume`只复用状态完整且哈希一致的阶段；不要覆盖已有失败/半成品或修改正在运行的冻结项目。

协议、解析规则及复核范围见[5000部电影神经实验方案](docs/5000部电影神经实验方案.md)。在线重新采集属于新数据版本；历史`run_experiment.ps1`采集流程保留用于新实验，不能替代本轮冻结复现。各配置只有一次seed42训练，未估计多种子方差，不据此声称统计显著优越。

## 实际结果、评分与名称恢复

五组神经配置均完成5轮训练、每轮8857题验证及恢复最佳检查点后的8917题测试。B0/B1/B2的原始验证F1同为99.778706%，按较小B编号规则选B0（最佳第3轮）；B3为99.765158%，C0只作消融。以下为完整同源测试集的**原始生成输出、统一评分v2**，指标保留两位小数：

| 系统 | 题数 | 集合准确率 % | 实体Macro-F1 % | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| A0 | 8917 | 100.00 | 100.00 | 100.00 | 100.00 |
| B0（选中） | 8917 | 99.53 | 99.53 | 99.78 | 99.89 |
| B1 | 8917 | 99.53 | 99.53 | 99.78 | 99.89 |
| B2 | 8917 | 99.46 | 99.46 | 99.76 | 99.89 |
| B3 | 8917 | 99.46 | 99.52 | 99.57 | 99.80 |
| C0（无结构前缀） | 8917 | 99.53 | 99.53 | 99.77 | 99.89 |

相同学习率1e-4下，原始B0与C0的测试实体F1均为99.528990%，本轮未观察到该指标的结构前缀收益。A0满分对应共享模板、可查询完整已知图谱的受控问答；各神经配置仅一次seed42训练，未估计多种子方差。

**评分修订v2：** 测试后发现，正确文本`Will Speck and Josh Gordon`被旧解析器识别成组合实体。修正通用保留事实中的并列名称解析后，对五组已保存验证/测试输出及A0、B0扰动输出统一复评；训练、参数选择、生成文本和固定名称恢复方法保持原值，原始验证选中B0的结论不变。112625条规范参考答案审计为零差异；356个原始v1文件（含权重）保持原样。[v2证据](verification/2026-10-30/decoder_revision_v2/summary.json)与[原始v1摘要](verification/2026-10-30/neural_summary_frozen_v1.json)分别保留。

主比较保留A0/B0—B3/C0的原始输出。实体集合准确率、按题Macro-F1、BLEU、ROUGE-L、有答案与拒答分别记录；当前神经解析只从答案部分提取实体，使用实际检索事实消解同名歧义，保留无关额外实体的误报。A0单题完整管道耗时与神经batch生成按题分摊的耗时范围不同，不直接相除为端到端加速倍数。

A0的0.30/0.70/0.90/0.99是同一冻结分类器的四种预设**推理配置**，不是四次训练，也不按本轮测试成绩重选阈值；正式默认仍为0.30。实际完整8917题记录见[A0阈值检查](verification/2026-10-30/A0_thresholds/summary.json)。

名称恢复原样迁移既有200部电影实验的固定`NameRecovery`方法，另由[run_frozen_recovery.py](benchmarks/run_frozen_recovery.py)使用当前`AnswerEntityIndex`评价。候选只来自实际链接电影最多24条保留事实的尾实体，恢复时不读取标准关系/答案，不重新训练或调整阈值；原始基线保留已保存的实体ID。先完整验证并冻结方法、输入、分词器和适配程序协议，五组结束后仅对原始验证选定B模型复核验证，再用匹配报告对测试/鲁棒性各评价一次，结果单列。

固定方法已完成匹配验证门控后的测试和扰动评价，附加结果单列：

| B0附加名称恢复 | 题数 | 原始实体F1 % | 恢复后实体F1 % | 修复 | 回退 | 文本变化 |
|---|---:|---:|---:|---:|---:|---:|
| [验证集](verification/2026-10-30/decoder_revision_v2/name_recovery_comparison.json) | 8857 | 99.77871 | 100.00000 | 24 | 0 | 39 |
| [测试集](verification/2026-10-30/decoder_revision_v2/name_recovery_comparison.json) | 8917 | 99.52899 | 100.00000 | 42 | 0 | 54 |
| [扰动诊断集](verification/2026-10-30/decoder_revision_v2/name_recovery_comparison.json) | 439 | 83.97874 | 83.97874 | 0 | 0 | 4 |

统一评分v2下，恢复后测试集合准确率与实体F1均为100.00000%，BLEU为99.90827、ROUGE-L为99.97876。原始v1检查中三个划分的ID复提取差异均为0；v2为同一批保存文本重新解析实体。名称恢复只使用实际链接电影的保留事实名称，与原始生成输出分别评价；原始验证选模和B0—C0消融设计保持不变。

439题诊断的原始集合准确率如下；别名仅与同源89道原题配对，错拼100题中实际改变99题。该覆盖导向抽样用于诊断，外推范围限于本轮条件，配对结果见[paired_robustness.json](verification/2026-10-30/paired_robustness.json)。

| 类型 | 题数 | A0准确率 % | B0准确率 % |
|---|---:|---:|---:|
| 原题 | 100 | 100.00 | 100.00 |
| 改写 | 100 | 75.00 | 68.00 |
| 别名 | 89 | 98.88 | 98.88 |
| 错拼 | 100 | 53.00 | 61.00 |
| 拒答 | 50 | 100.00 | 100.00 |

## 报告与人工评价

| 成员 | 本轮负责范围 | 报告 |
|---|---|---|
| 周新永，M202674502 | A0、实体链接、当前实体解码/恢复评价接口、严格评价、配对鲁棒性和验收 | [个人实验报告](reports/个人实验报告.docx) |
| 夏俊杰，学号留空 | 数据版本核验、TransE、神经建模训练、学习率与结构前缀消融 | [夏俊杰实验报告](reports/夏俊杰_实验报告.docx) |

历史仓库作为共同基础，不按新分工追认历史独立贡献。报告保留原文件路径，最终提交前核对主实验已更新为5000部、5.1—5.5节及图表来自本轮实际记录；班级和完成日期按实际情况填写。

本轮已从完整同源A0/B0预测生成[匿名空白答案表、密钥和抽样清单](reports/人工评价/5000部电影_20261030/)。两位真实评价者分别保存、独立填写自己的评分表，完成后才解码系统、讨论分歧；密钥由组织者单独保管。旧200部电影表格保留为历史材料。

准确性、完整性为0—2，流畅度1—5，无依据断言为0/1；0是真实评分，空白表示未评。`kgqa.human_eval --score-file`支持两份原始评分表配对统计，要求题号/答案位置与同次密钥完整对应；空白、缺失、重复或越界评分会被拒绝。原始评分保留，讨论结果另存；自动实体F1、BLEU或ROUGE不能代填主观评价。

真实浏览器模型测试已核对5000部数据、A0/B0双系统及固定名称恢复，[检查记录](verification/2026-10-30/live_ui/checks.json)为PASS；[一般导演问题](verification/2026-10-30/live_ui/general.png)和[名称恢复问题](verification/2026-10-30/live_ui/name_recovery.png)两张实际运行截图已留档。

**待补材料：** 两位评价者的独立人工评分，以及每人1张真实终端环境截图。环境图须在操作系统终端显示实际Python、依赖和项目目录。两份报告各16页，完整32页已通过版面检查，具体见[提交验收说明](docs/提交验收说明.md)。

## 代码检查

源码环境安装测试依赖后执行：

```powershell
python -m pytest -q
python -m kgqa.acceptance --output-dir output/acceptance/review
python benchmarks/run_linker_benchmark.py --output output/verification
```

v2修订后的[完整测试记录](verification/2026-10-30/decoder_revision_v2/provenance/source_checks.json)为344项通过、17.03秒，定向测试32项通过。原始v1[源码检查](verification/2026-10-30/source_checks/checks.json)保留19/19项通过、pytest330项通过及18.12秒记录，包括Python编译、Ruff、CLI、分发资产校验、PowerShell解析和合成验收。合成离线验收使用10部虚构电影，只检查代码流程，已有验收目录拒绝覆盖，复跑需换新目录。[CI配置](.github/workflows/ci.yml)在Windows/Ubuntu运行核心检查，云端结论按最终提交的实际日志记录。

## 文件索引

### 目录、材料与分发

| 入口 | 内容 |
|---|---|
| [kgqa/](kgqa/)、[tests/](tests/) | 主实现与自动测试；模块清单见下表。 |
| [benchmarks/](benchmarks/)、[scripts/](scripts/) | 固定对照评价与Windows启动/实验脚本。 |
| [datasets/](datasets/)、[5000部数据说明与清单](datasets/movies-5000-20261029/README.md) | 公开冻结资产清单、哈希、来源与使用入口。 |
| [data/README.md](data/README.md)、[data/](data/)、[models/](models/) | 数据版本说明；发布数据在`data/processed/movies_5000_20261029/`，配套A0在`models/pipeline_5000_20261029/`。 |
| [GPU与训练依赖](requirements.txt)、[CPU推理依赖](requirements-runtime-cpu.txt)、[测试依赖](requirements-test.txt)、[CI依赖](requirements-ci.txt) | 分别用于GPU训练、CPU双系统推理、CPU流程检查和CI。 |
| [协议](docs/5000部电影神经实验方案.md)、[验收指南](docs/提交验收说明.md)、[扩展记录](docs/电影数据集扩展记录.md)、[双人指南](docs/双人分工与复现指南.md) | 本轮复现/交付入口与来源；双人指南中的历史命令按其原数据版本理解。 |
| [reports/](reports/README.md)、[分工计划](reports/双人分工与复现实验计划.docx)、[提交审查](reports/提交审查.md)、[改进记录](reports/实验改进补充记录.md) | 两份个人报告、分工与检查材料；人工评价按本轮新目录隔离。 |
| [本轮verification/2026-10-30/](verification/2026-10-30/) | 已归档环境、输入/源码哈希、配置、逐题预测、指标、曲线与B0选模；附加恢复单列`name_recovery/`，主摘要为`neural_summary.json`。 |
| [教师实验说明](人工智能I-课程实验.pdf)、[教师报告要求](人工智能I_实验报告要求.docx) | 课程原件。 |
| [提交材料/](提交材料/)、[第一阶段设计](实验设计与参考资料.md)、[实施总结](项目实施总结与后续改进.md)、[第二阶段计划](第二阶段实验详细设计.md) | 历史报告/结果与设计背景，计划不等于已实现结果。 |
| `output/`、`runs/`、`.work/`、`.venv/` | 本地运行结果、检查点、缓存和环境；按各入口保存，不将不同数据/运行混用。 |
| `outputs/`交付产物（随最终交付提供） | 课程代码ZIP排除完整数据/权重；独立验收运行包包含5000部数据和选中B权重，两者分开提供。GitHub源码下载包本身已含发布数据+A0，范围不同。 |
| [.gitignore](.gitignore)、[.gitattributes](.gitattributes)、[.github/](.github/) | 本地产物排除、冻结资产换行保护和跨平台CI配置。 |

### kgqa模块与界面资源

| 模块 | 职责 |
|---|---|
| [__init__](kgqa/__init__.py)、[common](kgqa/common.py) | 包入口、关系/模板常量、JSON、日志与哈希。 |
| [fetch](kgqa/fetch.py)、[expand](kgqa/expand.py)、[prepare](kgqa/prepare.py)、[dataset](kgqa/dataset.py) | 采集/扩展、事实清理、电影划分/QA生成、发布资产校验和激活。 |
| [linker](kgqa/linker.py)、[pipeline](kgqa/pipeline.py)、[personal_experiment](kgqa/personal_experiment.py) | 实体链接与标题遮蔽、A0分类查询、隔离个人实验及参数对照。 |
| [expanded_suite](kgqa/expanded_suite.py)、[transe](kgqa/transe.py)、[neural](kgqa/neural.py) | 冻结5000套件、TransE向量、结构前缀模型与训练/验证/推理。 |
| [neural_preflight](kgqa/neural_preflight.py) | 真实分词器往返、输入/目标长度、事实条数与目标覆盖预检；本轮输入配置512。 |
| [metrics](kgqa/metrics.py)、[robustness](kgqa/robustness.py)、[expanded_evaluation](kgqa/expanded_evaluation.py) | 严格题号/覆盖与答案解析、扰动与同源配对、冻结A0复评。 |
| [name_recovery](kgqa/name_recovery.py)、[recovery_experiment](kgqa/recovery_experiment.py) | 固定名称恢复；后者保留历史解码器评价流程，本轮新基准使用下表适配入口。 |
| [finalize](kgqa/finalize.py)、[human_eval](kgqa/human_eval.py)、[error_analysis](kgqa/error_analysis.py) | 原始验证选模/鲁棒性、匿名盲评与真实评分统计、错误归因。 |
| [plots](kgqa/plots.py)、[environment_info](kgqa/environment_info.py)、[evidence](kgqa/evidence.py)、[report](kgqa/report.py) | 曲线、环境、已有记录摘要及团队报告草稿。 |
| [acceptance](kgqa/acceptance.py)、[demo](kgqa/demo.py)、[web](kgqa/web.py) | 合成离线验收、命令行演示、本机页面/API。 |
| [index.html](kgqa/ui/index.html)、[app.js](kgqa/ui/app.js)、[styles.css](kgqa/ui/styles.css) | 页面结构、检索/问答与结果导出、样式。 |

### 对照脚本、启动器与测试

| 文件 | 用途 |
|---|---|
| [run_frozen_thresholds.py](benchmarks/run_frozen_thresholds.py) | 同一冻结A0四预设阈值推理，保留预测及哈希，不重新训练。 |
| [run_frozen_recovery.py](benchmarks/run_frozen_recovery.py) | `--project --run-id --split --output`附加名称恢复评价；测试/鲁棒性还需`--validation-report`，不覆盖已有输出。 |
| [run_linker_benchmark.py](benchmarks/run_linker_benchmark.py)、[legacy_linker.py](benchmarks/legacy_linker.py) | 手工链接回归与冻结旧实现对照，非正式自然问题总体。 |
| [environment.ps1](scripts/environment.ps1)、[run_experiment.ps1](scripts/run_experiment.ps1) | 项目缓存/UTF-8设置与历史采集实验流程；本轮冻结训练优先使用`expanded_suite`。 |
| [run_ui.cmd](scripts/run_ui.cmd)、[run_ui.ps1](scripts/run_ui.ps1)、[run_demo.cmd](scripts/run_demo.cmd)、[run_demo.ps1](scripts/run_demo.ps1) | Windows界面与命令行演示包装器，透传配置和退出码。 |
| [tests/](tests/) | 数据与资产校验、链接/Unicode、实体解码/上下文、训练配置/套件、恢复、人工评分、HTTP、合成验收等回归。 |

历史1000部结果见[提交摘要](提交材料/实验结果摘要.md)；独立200部完整实验及旧名称恢复见[2026-10-29归档](verification/2026-10-29/README.md)；早期代码检查见[2026-10-24归档](verification/2026-10-24/README.md)。历史数字、数据和分工保留原上下文，不与本轮5000部结果合并。
