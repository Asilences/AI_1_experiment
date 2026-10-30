# 5000部电影完整实验与验收记录

[本机最终验收](final_acceptance.json)通过。五组各5轮完整训练，每轮8857题验证，各组8917题测试；按原始验证F1及较小编号同分规则选B0。两份报告各16页，32页均完成版面检查。

| 记录 | 内容 |
|---|---|
| [neural_summary.json](neural_summary.json) | 当前主摘要、完整参数、两人分工、v2结果和分析。 |
| [suite_summary.json](suite_summary.json)、[input_manifest.json](input_manifest.json)、[stages/](stages/) | 冻结训练输入、源码、五组配置、选模与检查点哈希。 |
| [runs/](runs/)、[logs/](logs/)、[transe/](transe/) | 原始v1逐题预测、训练/验证记录及100轮TransE。权重另随运行包分发。 |
| [decoder_revision_v2/](decoder_revision_v2/summary.json) | 测试后发现的组合实体解析修正；全部模型统一复评，验证选模、权重和生成文本不变。 |
| [v2源码检查与参考句核验](decoder_revision_v2/provenance/) | 344项完整测试、32项定向测试；112625个参考句无v2解析错配。 |
| [name_recovery/](name_recovery/) | 既有固定名称恢复方法的原评价，保留v1实体解析口径。当前统一v2对照见[名称恢复复评](decoder_revision_v2/name_recovery_comparison.json)。 |
| [A0_thresholds/](A0_thresholds/summary.json) | 同一冻结分类器的四种预设推理阈值，各8917题。 |
| [paired_robustness.json](paired_robustness.json) | 改写、别名、错拼各与同源原题配对；别名分母89。 |
| [source_checks/](source_checks/checks.json) | 修订前19项本机检查、330项测试的历史记录；后续v2源码测试另存上表。 |
| [report_layout.json](report_layout.json)、[live_ui/](live_ui/checks.json) | 32页报告检查及两张实际浏览器模型测试截图。 |
| [runtime_package_checks.json](runtime_package_checks.json) | 独立CPU环境四项离线行为验收、包内文件哈希与最终运行包校验。 |

当前统一实体F1：A0为100%；原始B0/B1/C0为99.52899%，B2为99.46170%，B3为99.51553%。同一B0检查点经固定名称恢复后，测试实体准确率与F1均为100%，BLEU为99.90827、ROUGE-L为99.97876。所有结果限于本轮冻结知识库和模板划分；扰动诊断单独列出。

v2复评期间356份原文件保持字节一致。发布时仅将A0阈值摘要的三个本机路径字段改为相对路径，数值、输入、模型及源码哈希保持原样；[路径归一化记录](publication_path_normalization.json)保存原始/发布哈希。原始主摘要见[neural_summary_frozen_v1.json](neural_summary_frozen_v1.json)。

两份人工评分表仍为空，终端环境截图仍待本人补充，见[填写说明](../../reports/人工评价/5000部电影_20261030/填写说明.md)。
