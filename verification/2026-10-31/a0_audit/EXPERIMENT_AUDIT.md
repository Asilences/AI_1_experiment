# A0满分完整性审阅

日期：2026-10-31。项目：AI_1_experiment。审阅会话：/root/a0_integrity_audit。下文为新鲜上下文只读审阅原文；精确运行模型身份unknown，同系列暂定结论。

Overall：**WARN**。5000 部电影版本的 A0 满分可由真实逐题预测复算，未发现伪造 GT、分数自归一化或把准确率写死的证据。主要限制是测试与训练共享模板、推理可查完整知识库，以及 200 部电影版本完整原始数据当前缺失。审阅只读，未改文件、模型或分数。

审阅身份：requested model 为 `gpt-6-astra`、requested effort 为 `ultra`；无工具返回可验证的实际运行模型 ID，**observed model = unknown**。同 GPT-6 系列审阅，`review_independence=same-family`、`acceptance_status=provisional`。审阅期间 HEAD 从 `b8fadde` 变为 `aa5b142`，已确认本次八个核心代码文件相对 `b8fadde` 无差异。

**A. GT 来源：PASS，附范围限定。**

- [prepare.py:65](H:/AI1experiment/kgqa/prepare.py:65) 从冻结快照电影 `facts` 取得 `answer_ids`，并在第 73 行调用 `answer_text` 生成参考文本；拒答 GT 来自第 79–81 行的固定 `UNKNOWN` 模板。没有从 A0/B0 的预测生成 GT。
- [pipeline.py:36](H:/AI1experiment/kgqa/pipeline.py:36) 推理也从同一个电影实体的 `facts` 查答案，第 39 行使用同一 `answer_text` 格式化。[common.py:76](H:/AI1experiment/kgqa/common.py:76) 定义四类固定答案句式。
- 独立核对完整 5000 版本：**58994/58994 个 gold 答案集合与可见图谱一致，58994/58994 条第一参考文本等于程序格式化结果**。所以这是有来源的程序化 GT，但不是独立人工事实核验；结果不能验证 Wikidata 本身的真实性或完整性。

**B. 评分、归一化和对齐：PASS。**

- [metrics.py:67](H:/AI1experiment/kgqa/metrics.py:67) 以 `question_id` 严格校验重复、未知、缺失预测；[metrics.py:101](H:/AI1experiment/kgqa/metrics.py:101) 对实体集合计算 exact 与逐题 F1，第 116–125 行对真实题数求均值并乘 100。没有用模型自身最大值、均值等归一化。
- A0 使用查出的实体 ID；B0 使用生成文本抽取出的实体 ID。BLEU、ROUGE 分别比较实际答案文本与数据集参考文本。[metrics.py:96](H:/AI1experiment/kgqa/metrics.py:96)、[neural.py:118](H:/AI1experiment/kgqa/neural.py:118)。
- 代码中的 `prediction_coverage=100.0` 是完成完整预测及严格评分后的覆盖率元数据，不是硬编码准确率：[personal_experiment.py:347](H:/AI1experiment/kgqa/personal_experiment.py:347)、[expand.py:408](H:/AI1experiment/kgqa/expand.py:408)。
- `BLEU=100.00000000000004` 是浮点舍入，不是超越满分的有效改进。
- 补充阈值预测独立复评确实降分，直接否定“评分恒为 100”的猜测。

**C. 产物存在、分母及可复算性：WARN。5000 与历史 1000 核对通过；200 版有原始输入缺口。**

| 版本/配置 | 独立核对结果 |
|---|---|
| 5000 A0 默认阈值 | 8917 条唯一预测，缺失/额外均为 0；**8917/8917** 正确；8106 可答＋811 拒答；8917 条文本均匹配某条参考 |
| 历史 1000 A0 | 1855 条唯一完整预测；**1831/1855＝98.706199%**；1686 可答＋169 拒答 |
| 同伴补充阈值 0.300 / 0.995 | 各 **8917/8917＝100%**，0 次误拒答 |
| 同伴补充阈值 0.998 | **7501/8917＝84.120220%**，1416 次误拒答 |
| 同伴补充阈值 0.999 | **811/8917＝9.094987%**，8106 次误拒答 |
| 同伴 5000 B0，统一评分 v2 | 8917 条唯一完整预测；重算 exact/F1 **99.52898957048335%**、BLEU **99.77551051994016**、ROUGE-L **99.89321992114122** |

5000 数据 SHA-256 与 [config.json:9](H:/AI1experiment/models/pipeline_5000_20261029/config.json:9) 完全一致；原始指标见 [metrics_test.json:2](H:/AI1experiment/models/pipeline_5000_20261029/metrics_test.json:2)。独立核算也与执行者新增 [recomputed.json:14](H:/AI1experiment/verification/2026-10-31/a0_audit/recomputed.json:14) 一致。

补充阈值直接读取并复评了 `origin/feature/expanded-neural-full@c71ff0e:reports/补充阈值运行记录/threshold_0.{300,995,998,999}.jsonl`，没有仅信汇总。相应汇总原字节副本见 [teammate_threshold_summary.json:163](H:/AI1experiment/verification/2026-10-31/a0_audit/teammate_threshold_summary.json:163)。同伴 B0 使用 `verification/2026-10-30/decoder_revision_v2/runs/B0/predictions_test.jsonl`；没有把原始 v1 汇总与 v2 当作同一评分版本。

200 版：

- A0 保存预测为 370 条唯一测试预测、381 条唯一鲁棒性预测；B0 测试预测为 370 条唯一记录。
- 保存指标为 A0 370/370，B0 365/370；鲁棒性 A0 clean 100/100、改写 75/100、错拼 85/100、别名 47/47、拒答 34/34。[A0 metrics_test.json:2](H:/AI1experiment/verification/2026-10-29/runs/A0/metrics_test.json:2)、[metrics_robustness.json:76](H:/AI1experiment/verification/2026-10-29/runs/A0/metrics_robustness.json:76)、[B0 metrics_test.json:2](H:/AI1experiment/verification/2026-10-29/runs/B0/metrics_test.json:2)。
- 但 [data_manifest.json:3](H:/AI1experiment/verification/2026-10-29/runs/A0/data_manifest.json:3) 所列完整 200 版 QA、entities、splits 当前不在给定路径。当前 `data/processed` 是 1000 版，哈希不同。**200 版上述分数只能确认保存产物和分母一致，不能声称此次完成从匹配 GT 的独立复算。**

**D. 指标是否实际调用：PASS。**

- A0 个人实验调用并保存测试、鲁棒性评分：[personal_experiment.py:348](H:/AI1experiment/kgqa/personal_experiment.py:348)、[personal_experiment.py:379](H:/AI1experiment/kgqa/personal_experiment.py:379)。
- 5000 构建流程对完整测试预测调用评分并保存：[expand.py:414](H:/AI1experiment/kgqa/expand.py:414)。
- 神经验证调用评分：[neural.py:234](H:/AI1experiment/kgqa/neural.py:234)；测试评分入口是 [metrics.py:131](H:/AI1experiment/kgqa/metrics.py:131)。本次对同伴 v2 保存预测实际调用评分，返回值与保存指标一致。

**E. 实验范围与结论：WARN。**

- 5000 电影划分确实为 3500/750/750，电影 ID 无交集。但独立检查发现 train/valid/test **恰好共享同一 15 种模板，零未匹配题**。题目生成规则见 [prepare.py:12](H:/AI1experiment/kgqa/prepare.py:12)，电影划分见第 22 行。
- A0 只训练 TF-IDF＋逻辑回归关系分类器；推理的实体链接候选及事实库包含全部划分电影：[pipeline.py:57](H:/AI1experiment/kgqa/pipeline.py:57)、[pipeline.py:63](H:/AI1experiment/kgqa/pipeline.py:63)、[pipeline.py:120](H:/AI1experiment/kgqa/pipeline.py:120)。因此测试电影没参与分类器训练，并不意味着测试事实在推理时不可见。
- 扩展版本还主动把歧义片名改成年份/QID形式并加入别名：[expand.py:254](H:/AI1experiment/kgqa/expand.py:254)、[expand.py:282](H:/AI1experiment/kgqa/expand.py:282)，并检查所有标准题标题能正确链接（第 392–396 行）。实际例子：历史 [qa.jsonl:2404](H:/AI1experiment/data/processed/qa.jsonl:2404) 是 `Who directed City on Fire?`，5000 [qa.jsonl:2404](H:/AI1experiment/data/processed/movies_5000_20261029/qa.jsonl:2404) 变为 `Who directed City on Fire (Q166734)?`。这解释了扩展题库比历史版更容易全部准确链接，不能把增长解释成开放问法能力提升。
- A0 可答题置信度范围 **0.997337574–0.998783739**，所以 0.30/0.70/0.90/0.99 均不会改变任何可答题。0.998 和 0.999 才跨过实际置信度范围。不是四次独立训练，也不是校准后的“真实正确概率”。
- 200 版三次重复明示同一个已拟合模型重复推理：[config.json:29](H:/AI1experiment/verification/2026-10-29/runs/A0/config.json:29)。同伴追加高阈值明确是诊断后的补充，不能改称预注册泛化实验：[teammate_threshold_summary.json:3](H:/AI1experiment/verification/2026-10-31/a0_audit/teammate_threshold_summary.json:3)。
- 现有 README 已限制范围：`b8fadde:README.md:100`；同伴 `c71ff0e:README.md:74,102,108,141` 同样说明共享模板、完整图谱和推理敏感性。未发现这里把满分宣传为开放域泛化。V2 自然问题金标准仍是 TODO：[EXPERIMENT_TRACKER.md:48](H:/AI1experiment/refine-logs/EXPERIMENT_TRACKER.md:48)，不得借用 V1 满分填充。

**F. 评估类型：PASS，分类为 `dataset_derived_programmatic_template_gt`。**

这是 **Wikidata 冻结快照派生的程序化模板 GT**，评估“已知图谱上的受控问答流程”。不是模型生成的 `synthetic_proxy`，不是人工标注问答或已完成人工评审，也不是未知图谱事实预测。若机器格式只能使用技能既有枚举，可写 `real_gt`，但必须附 `gt_subtype=dataset_derived_programmatic_template_gt` 与完整知识库可见的限制。

A0/B0 的关键差异：A0 识别电影和关系后直接返回图谱实体、套模板；B0 接收问题、检索事实文本及 TransE 结构前缀，经 FLAN-T5-small 生成文字，再抽取实体评分。[neural.py:42](H:/AI1experiment/kgqa/neural.py:42)、[neural.py:57](H:/AI1experiment/kgqa/neural.py:57)、[neural.py:93](H:/AI1experiment/kgqa/neural.py:93)。B0 额外承担名称生成、文本解码和实体抽取错误；两者都可以访问答案所在的图谱，不是让 B0 凭空记忆电影知识。

建议及结论影响：

- **支持**：“A0 在这份冻结 5000 电影、共享模板、完整知识库可见的 8917 题测试上达到 100%。”
- **不支持**：“A0 所有评估都是 100%”“任意自然问句准确率 100%”“对未知知识泛化 100%”“重复三遍证明训练稳定性”。
- 保留当前真实分数与负结果；报告显式列出版本、分母、模板和知识库可见条件。
- 恢复匹配哈希的 200 版完整输入后再称其完全可复算。
- 后续评估应冻结独立自然问题及事实核验，并让改写模板与训练模板真正隔离；不要为了制造差异修改已有分数或用这些测试结果重选默认阈值。

