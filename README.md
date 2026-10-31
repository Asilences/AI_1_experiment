# Wikidata 电影知识图谱问答

《人工智能 I》课程项目：输入中文或英文电影问题，识别电影实体，查询冻结知识图谱并返回答案与事实证据。支持导演P57、类型P136、制作国家P495、原始语言P364四类单跳问题，以及别名、错拼、改写和拒答诊断。中文入口支持中文片名、英文片名与中文问句混合、年份消歧和中文答案展示，离线运行，无需调用外部问答服务。

**本轮主实验为5000部电影，五组神经实验已全部完成，原始验证选中B0。** 仓库随附完整冻结快照、7864个实体、35485条三元组、58994道问答及配套A0，可直接校验并启动管道问答。完整配置、逐题预测、曲线与指标见[本轮实验汇总](verification/2026-10-30/neural_summary.json)；双系统完整运行包见[验收包发布入口](https://github.com/Asilences/AI_1_experiment/releases/tag/movies-5000-zh-20261031)。

[源码快速启动](#源码快速启动5000部电影a0) · [双系统运行包](#独立双系统运行包) · [中文问答](#中文问答与演示) · [自然问法与扩展事实](#自然问法与扩展事实) · [完整实验复现](#完整实验复现) · [实际结果与名称恢复](#实际结果评分与名称恢复) · [报告与人工评价](#报告与人工评价) · [代码检查](#代码检查) · [文件索引](#文件索引)

## 源码快速启动：5000部电影A0

克隆仓库或通过GitHub **Code → Download ZIP** 下载源码，进入包含`kgqa`的根目录。Windows PowerShell、Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m kgqa.dataset activate --smoke
.\.venv\Scripts\python.exe -m kgqa.web --open
```

依赖安装需要网络，随后A0使用本地数据与模型，无需重新采集或训练。`activate`先核对文件大小、SHA-256、数据/模型配对及示例回答，成功后写入本机`.work/ui-dataset.json`。**此激活入口为5000部电影的A0模式。** 只验证资产、不切换界面时执行`python -m kgqa.dataset verify --smoke`。来源、逐文件清单与使用说明见[数据说明](datasets/movies-5000-20261029/README.md)和[manifest](datasets/movies-5000-20261029/manifest.json)。

浏览器访问[http://127.0.0.1:7860/](http://127.0.0.1:7860/)，可搜索片名、中文标签/别名和QID，输入中英文问题，查看实体链接、关系与事实证据，复制答案或导出本次结果JSON。保留启动终端，按`Ctrl+C`停止；端口占用时使用`--port 7861`。Windows也可使用`scripts/run_ui.cmd`；其他平台将解释器路径换为`.venv/bin/python`。

## 独立双系统运行包

[验收包发布入口](https://github.com/Asilences/AI_1_experiment/releases/tag/movies-5000-zh-20261031)用于分发独立运行包，内容为同一5000部电影数据、配套A0、TransE和验证选中的B0权重。下载该包后无需重新下载基础模型或训练，依赖安装仍需网络。

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

## 中文问答与演示

页面默认提供中文问题，选择电影后可点击导演、类型、制作国家、原始语言四个按钮生成问题，也可直接输入自己的问句。示例：

- `《盗梦空间》的导演是谁？`
- `《泰坦尼克号》的原始语言是什么？`
- `Inception 是哪个国家制作的？`
- `2011年版《美国队长》的导演是谁？`

中文输入先由本地片名与属性识别模块转为四类英文模型问题，再调用原 A0 或选中的 B0。A0按实际预测实体生成中文答案；神经答案只转换已识别的原始生成句式和名称，保留错误或未知内容，原始英文生成可展开核对。神经权重和正式英文实验数据均保持原值，本功能不声称完成了中文模型训练。

优势体现在本地中英文交互、两种方法的同题对比、可追踪的片名与关系、事实证据和原始生成。中文片名缺失时可使用英文片名或QID，中文标签缺失的答案实体保留原名。同名电影会要求补充年份或QID；当前每次支持一部电影的一个属性，范围外问题会明确提示。

中文受控验收、首次歧义诊断和8917题英文回归分别留档，不能将模板验收成绩视为任意中文问句准确率。协议、结果和复现命令见[中文问答说明](docs/中文问答说明.md)。中文改进作为独立扩展记录，原英文主实验结果继续保留；周新永的个人报告另纳入中文接口验收和A0补充阈值分析。

## 自然问法与扩展事实

新增无需训练的“自由问答增强 · 证据核验”（页面默认开启）：支持范围检查、模糊片名确认、关系证据筛选和显式工具回退，保留原始模型输出。取消勾选可回到原入口。它属于独立系统增强，未更新神经权重，不替代原始A0/B0实验分数。实现、开发验收及报告补充见[无需训练增强说明](docs/无需训练增强与报告补充_20261031.md)。

“自然问法”模式支持已有四类关系的英文自然表达和有限中文问法，先显示标准问题转换再查询；答案保持英文，无法确定时澄清或拒答。它是单独的规则解析入口，不代表已训练模型自动学习了新问句。完整使用、新数据规模和同伴训练交接见[自由问法与训练交接](docs/自由问法与训练交接.md)。

独立的“扩展事实问答”含5000电影、170516条带限定条件的声明，可查询演员、编剧、日期、地点、多属性、共同演员及导演作品。冻结目录已另发[复现资产Release](https://github.com/Asilences/AI_1_experiment/releases/tag/submission-assets-20261031)，用`python -m kgqa.submission_assets install --asset catalog_v2`恢复，清单见[manifest](configs/submission-assets-20261031.json)。它不在旧四关系包内；部分名称限流后保留QID。覆盖和边界见[V2实施与验收](docs/V2工具问答实施与验收.md)。

另有独立试验模式 **P0自然语言图谱问答**：冻结Qwen2.5-3B-Instruct解释中英文问题，输出受控JSON计划，经候选绑定和校验后查询V2目录，再依据证据组织回答；必要时明确标注模板回退。此模式需另行启动模型服务，模型权重未随仓库或旧运行包发布；不覆盖A0/B0。部署、CLI和UI方法见[P0使用说明](docs/P0冻结语义问答使用说明.md)。

P0已跑通真实链路，但原20道程序编写开发题计划匹配7/16，低于D0的12/16；状态匹配14/20，高于D0的12/20。原回退11/20，本轮受控回答保护对保存输出离线重放为13/20；不算新模型实验。最终答案始终展示受控证据句，原始模型文本另存。尚无独立自然问题或人工文本评分，不宣称理解任意问题。[原开发输出](verification/2026-10-31/semantic_p0/dev_v1/)与[本轮审计修订](docs/PR12审计修订说明.md)分别保留。

自由问法v2原6904题F1 69.4081%为混合测试成绩：738题模板已见（F1 100%），其余6166题F1 65.7466%，由原预测真实重算。冻结数据不覆盖，新检查点与完整预测已随上述Release发布。人工评价为任务书必需项，目前无完整真人评分；可运行`python -m kgqa.human_review_web --port 7865`，访问 http://127.0.0.1:7865/ 。[评分说明](docs/人工评价操作说明.md)及[当前提交审查](reports/提交审查.md)明确未完成项。

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
| 周新永，M202674502，硕2609班 | A0、实体链接、P0执行与评价、阈值敏感性和鲁棒性分析 | [个人实验报告](reports/个人实验报告.docx) |
| 夏俊杰，M202674534，硕2610班 | 数据版本核验、TransE、神经建模训练、学习率与结构前缀消融、自由问法微调 | [夏俊杰实验报告](reports/夏俊杰_实验报告.docx) |

历史仓库作为共同基础，不按新分工追认历史独立贡献。两份报告采用统一中文课程模板，当前各15页，[双人分工与复现实验计划](reports/双人分工与复现实验计划.docx)5页；最终文件SHA及35页逐页核对见[本轮版面记录](verification/2026-10-31/pr12_audit_fixes/reports_qa.json)。主实验完成日期按本人提供内容填写为2026年10月30日，10月31日扩展与审计单列，不改写历史运行日期。

[补充阈值运行记录](reports/补充阈值运行记录/README.md)保留同一冻结A0在8917题上的四组实际预测、指标和原始脚本：0.30/0.995/0.998/0.999的实体F1分别为100.00%/100.00%/84.12%/9.09%，误拒答数为0/0/1416/8106。这是置信度诊断后的推理敏感性补充，不重新训练，也不按测试成绩重选默认阈值0.30。

关于编号和满分原因，见[A0、B0及评测口径核查](docs/A0与B0含义及满分现象说明.md)。本次已加载正式5000部A0并实际复现8917题；四个原阈值均低于可回答题最小置信度0.9973376，因此不改变决策。高阈值0.998/0.999诊断分别为84.12%/9.09%；原默认阈值与历史结果不变。

自由问法v2已完成旧B0验证基线、3轮独立微调及6904题混合测试，原69.4081%保留为混合成绩；738题已见模板的F1为100%，其余6166题为65.7466%，由保存预测实际重计分。完整说明见[训练记录](docs/自由问法训练启动记录_20261031.md)与[审计勘误](docs/PR12审计修订说明.md)。拒答退化，未替换界面默认模型。英文四关系改写与12属性V2目录是不同版本；演员等工具能力不能算成四关系生成模型已学会的能力。

2026-10-24 的记录包含 181 项通过的测试与离线验收；既有 10 月 19 日模型在新链接代码下回归，370 道测试、381 条鲁棒性题的非时间指标与逐题输出均保持一致，未重训。[查看正式数据回归](verification/2026-10-24/formal_regression.json)。
课程任务书要求人工评价准确性、完整性和流畅度，教师未要求必须两位评价者。本轮已从完整同源A0/B0预测抽取相同100题，生成[200份匿名答案、密钥和抽样清单](reports/人工评价/5000部电影_20261030/)，尚无完整真人评分。运行`python -m kgqa.human_review_web --port 7865`，访问 http://127.0.0.1:7865/ ，使用本人代号实际评分；支持续填和导出，保存至`output/human_review/`。[操作说明](docs/人工评价操作说明.md)提供评分标准与汇总命令。至少一名真人完整评分后才统计，多人应独立填写，完成后才解码系统、讨论分歧；密钥由组织者单独保管。旧200部电影表格保留为历史材料。

准确性、完整性为0—2，流畅度1—5，无依据断言为0/1；0是真实评分，空白表示未评。`kgqa.human_eval --score-file`支持一名完整评价者的统计，多人时另报一致性，要求题号/答案位置与同次密钥完整对应；空白、缺失、重复或越界评分会被拒绝。原始评分保留，讨论结果另存；自动实体F1、BLEU或ROUGE不能代填主观评价。

真实浏览器模型测试已核对5000部数据、A0/B0双系统及固定名称恢复，[检查记录](verification/2026-10-30/live_ui/checks.json)为PASS；[一般导演问题](verification/2026-10-30/live_ui/general.png)和[名称恢复问题](verification/2026-10-30/live_ui/name_recovery.png)两张实际运行截图已留档。

**尚未完成的课程材料：** 真人评分和真实终端环境截图。截图须在真实操作系统终端显示解释器路径、Python与依赖版本、项目目录，标清实验环境归属；两台机器的版本可以不同。本机复核及V100补充运行的环境文本已经保存，不能冒称同伴原主实验环境，也不能代替截图。历史版面检查仅对应当时文件；当前报告以[本轮核对](verification/2026-10-31/pr12_audit_fixes/reports_qa.json)为准。具体要求见[提交验收说明](docs/提交验收说明.md)。

已发布的中英文运行包与同页源码ZIP是其发布时的验收快照，资产与标签保持不变；当前报告和本轮修复以当前仓库为准。本轮另发布[冻结V2图谱与自由问法v2检查点](https://github.com/Asilences/AI_1_experiment/releases/tag/submission-assets-20261031)，含下载位置、归档及逐文件SHA的[manifest](configs/submission-assets-20261031.json)；该tag指向旧实验基准，其源码ZIP不包含本轮修复。

## 代码检查

本轮PR12审计修订的[源码检查](verification/2026-10-31/pr12_audit_fixes/source_checks.json)为完整pytest740项通过、Ruff与编译通过；[报告源检查](verification/2026-10-31/pr12_audit_fixes/report_source_checks.json)验证最终DOCX与版面记录的SHA一致，并实际检查旧输入重新生成后的勘误、图像和原生公式。下列历史检查保留其对应版本。

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
| [中文适配器](kgqa/chinese.py)、[中文展示名称](kgqa/locales/zh_entities.json)、[中文验收](benchmarks/run_chinese_acceptance.py) | 中文片名与属性归一化、官方标签及别名、固定题目接口验收；原始事实和神经权重保持原值。 |
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
| [freeform](kgqa/freeform.py)、[freeform_data](kgqa/freeform_data.py) | 显式自然问法解析、标准问题转换和拒答；另存继承冻结电影划分的英文合成改写。 |
| [candidates](kgqa/candidates.py)、[explorer](kgqa/explorer.py)、[v2_data](kgqa/v2_data.py) | 中英电影候选、扩展事实只读查询与独立目录构建，保留来源和限定条件。 |
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
