# Wikidata 电影知识图谱问答

[![CI](https://github.com/Asilences/AI_1_experiment/actions/workflows/ci.yml/badge.svg)](https://github.com/Asilences/AI_1_experiment/actions/workflows/ci.yml)

本项目为《人工智能 I》课程实验，实现并比较两类基于 Wikidata 电影知识图谱的单跳问答系统：

- **A0 管道式系统**：实体链接、TF-IDF 关系分类、图谱事实查询和模板化答案生成。
- **B0–B3 神经生成系统**：使用图谱事实与冻结的 TransE 结构向量作为条件，基于 FLAN-T5-small 直接生成答案。
- **C0 消融系统**：保留图谱事实文本，但不使用 TransE 向量。

虚拟环境、缓存、模型和临时文件均保存在项目目录中，便于迁移与复现。

## 个人工作提交与验收

个人报告：[周新永 M202674502 的实验报告](reports/个人实验报告.docx)。本次在已合并的代码基础上补齐离线验收入口、个人模型演示路径，并修复未知数字续集误命中原片的问题。

在项目根目录使用 Python 3.11 或 3.12，首次安装 CPU 依赖后执行：

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
python -m kgqa.acceptance --output-dir output/acceptance/review
```

最后一条成功时输出 `PASS` 并退出 0；在 `output/acceptance/review/summary.json` 查看逐项检查，在 `examples.json` 查看四种关系、多答案、别名和拒答。再次执行请换一个新目录，或省略 `--output-dir` 自动生成新目录。

验收过程使用 **10 部虚构电影**，无需下载 Wikidata、神经模型或使用 GPU。它检查代码能否完成训练、验证选阈值、预测、评测和模型重载；正式 Wikidata 实验结果仍见下文与个人报告。完整操作、交互式问答和真实截图步骤见 [提交验收说明](docs/提交验收说明.md)。

## 实验文档与报告

| 文档 | 内容 |
|---|---|
| [第一阶段实验设计与参考资料](实验设计与参考资料.md) | 第一阶段实施前的实验路线、参数计划和参考资料。 |
| [项目实施总结与后续改进](项目实施总结与后续改进.md) | 第一阶段实现情况、已有结果、问题与后续改进方向。 |
| [第二阶段实验详细设计](第二阶段实验详细设计.md) | Qwen3、图嵌入、问题引导融合网络与对照实验的设计方案，尚未实施。 |
| [第一阶段实验报告 PDF](提交材料/人工智能I_实验报告.pdf) | 第一阶段实验报告的原始导出版本，便于在线查看。 |
| [第一阶段实验报告 Word](提交材料/人工智能I_实验报告.docx) | 对应的可编辑报告文件。 |
| [第一阶段实验结果摘要](提交材料/实验结果摘要.md) | 正式数据规模和各实验的已保存指标。 |
| [人工评价表（待填写）](提交材料/人工评价表_待填写.csv) | 已抽样的人工评价表；评分列仍需实验者填写。 |
| [课程实验说明](人工智能I-课程实验.pdf) | 教师提供的课程实验说明原件。 |
| [实验报告要求](人工智能I_实验报告要求.docx) | 教师提供的报告格式和提交要求原件。 |

第一阶段报告与结果摘要保留原始记录。神经系统中的延迟是批处理摊销生成耗时，单题端到端响应需按第二阶段设计重新测量；相关评测局限见第二阶段实验详细设计第 3 节。第二阶段设计中的模型和规模属于后续计划，不代表已经完成的实验结果。

## 本次优化与双人协作

2026-10-18 的优化修复了实体模糊链接、别名遮蔽、评测 ID 校验和人工评价抽样，并使 A0 演示无需加载神经依赖。责任规划、复现命令和提交前待完成事项见 [双人分工与复现指南](docs/双人分工与复现指南.md)。历史工作作为小组共同基础，本次分工不追溯认定个人贡献。

不下载模型即可运行单元测试和手工链接回归实验：

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
python benchmarks/run_linker_benchmark.py --output output/verification
```

`benchmarks/legacy_linker.py` 冻结自优化前提交 `9b85f49`，仅用于对照。该实验包含 44 条手工链接用例，不是正式 Wikidata 测试集；阈值敏感性实验的每个设置重复运行三次，准确率的重复不等同于多随机种子模型训练。已保存本次记录位于 `verification/2026-10-18/`。

`kgqa.evidence` 只导出保存记录的 JSON/TXT 摘要，不再绘制终端样式图片。课程要求的真实终端截图需本人采集，放入 `output/screenshots/environment.png`、`training_b0.png`、`evaluation.png`；历史报告内的日志渲染图片不能作为真实截图使用。人工评分必须由两位实验者独立填写。

### 个人 A0 实验已完成的自动部分

2026-10-19 在新采集的 **200 部电影、552 个实体、1264 条三元组、2492 道问答**上完成 A0 训练、验证选阈值、370 题测试、四组预设阈值各三次推理以及 381 条鲁棒性评测。电影划分 140/30/30，问答划分 1749/373/370，种子 42。

候选来自已缓存的 1980—1989 发行日期区间 SPARQL 查询，不保证每部电影最早上映年份位于该区间；不代表全部 Wikidata，也不是原 1000 部电影实验的复现。新快照 SHA-256：`5b02443928ef5503c280c9422376cf5efb8849f13e02ee92f03aca917c47ed38`。

验证集选出的关系阈值为 0.30。测试实体集合准确率、Macro-F1、BLEU 和 ROUGE-L 均为 100；**测试题遮盖电影名称后全部属于训练集已有的 15 种问题模板**，且检索使用完整本地知识库。这说明受控模板流程在该划分上通过，不能解释为任意新问法都能回答。鲁棒性中同义改写为 75/100、错拼为 85/100、别名为 47/47；别名只与相同 47 个源问题的 clean 版本比较。

运行器在独立目录保存模型与结果，拒绝覆盖已有 run；阈值只在验证集选择，测试参数对照不得反向改变基线。0.30/0.70/0.90/0.99 是预先固定的诊断阈值，每组重复三次推理，不是三个随机种子的重新训练。

```powershell
python -m kgqa.fetch --target 200 --candidate-cache data/raw/candidates_decade_1980.json
python -m kgqa.prepare --snapshot data/raw/snapshot_200.json
python -m kgqa.robustness --limit 100 --seed 42
python -m kgqa.personal_experiment --run-id A0_personal_20261019
```

上述候选缓存由原采集器的首次成功查询产生；缺失时需按 `kgqa.fetch` 的查询说明重新取得，不能凭空生成。在线重新采集会改变数据版本。完整数据和模型保留在本地且不进入代码压缩包；本次评测摘要、逐题预测、参数表和来源记录保存在 `verification/2026-10-19/`。个人报告与分工计划位于 `reports/`。报告已填写周新永与学号，班级按本人要求留空；真实截图与主观人工评分仍需补齐，神经实验由队友继续完成。

## 环境配置

在已克隆的项目根目录运行：

```powershell
python -m venv .venv
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 交互式问答演示

已训练的模型和处理后的数据位于本地项目目录中。进入项目目录后，运行：

```powershell
.\scripts\run_demo.cmd
```

`.cmd` 启动器只会在本次运行中临时绕过 PowerShell 脚本执行策略，不会修改系统策略。

出现 `Question>` 后输入英文电影问题，输入 `exit` 退出。示例：

```text
Who directed Shutter Island?
What genre is Boss Level?
```

默认的 `both` 模式会并排展示 A0 与验证集选出的最佳神经模型 B2，并显示电影实体链接、关系预测、知识图谱证据、最终答案和回答延迟。

只运行一个系统：

```powershell
.\scripts\run_demo.cmd -System a0
.\scripts\run_demo.cmd -System best
```

直接运行一道题：

```powershell
.\scripts\run_demo.cmd -Question "Who directed Shutter Island?" -System both
```

系统当前支持四类英文单跳问题：导演、类型、制作国家和原始语言。问题中的电影必须存在于冻结的本地知识库；实体无法链接或缺少事实时，系统会明确拒答。

## 持续集成检查（CI）

推送到 `main`、创建或更新 Pull Request 时，GitHub Actions 会自动在 Ubuntu 和 Windows 的 Python 3.11 环境中检查 Python 语法、致命代码错误、数据准备与管道系统的命令行入口、现有核心测试以及 PowerShell 脚本语法。也可以在仓库的 **Actions → CI → Run workflow** 中手动运行。

本次另加入离线训练与问答验收步骤；打开对应运行的日志，检查测试通过以及 `PASS synthetic offline acceptance`。

CI 使用独立的轻量依赖文件 `requirements-ci.txt`，无需 GPU、实验数据或已训练模型。完整训练和神经模型效果仍需按实验流程单独验证。

在本地复现 Python 检查（可使用独立的检查环境，所有文件保存在项目内）：

```powershell
python -m venv .work/ci-venv
.\.work\ci-venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.work\ci-venv\Scripts\python.exe -m compileall -q kgqa tests
.\.work\ci-venv\Scripts\python.exe -m ruff check --no-cache --select E9,F63,F7,F82 kgqa tests
.\.work\ci-venv\Scripts\python.exe -m kgqa.prepare --help
.\.work\ci-venv\Scripts\python.exe -m kgqa.pipeline --help
.\.work\ci-venv\Scripts\python.exe -m pytest -q
```

## 小规模验证实验

```powershell
.\scripts\run_experiment.ps1 -Mode smoke -Target 50
```

## 完整实验

```powershell
.\scripts\run_experiment.ps1 -Mode full -Target 1000
```

Wikidata 采集器会缓存成功请求并支持断点续跑。完整实验需要下载模型并执行多次微调，耗时可能较长。运行日志保存在 `runs/`，图表和评估表保存在 `output/`。

自动生成的 `human_evaluation.csv` 人工评分列保持为空，必须由实验者填写；自动指标不作为人工事实核查的替代。
