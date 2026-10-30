# 自由问法构建记录

| Rung | 功能 | 单条验收命令 | Tier | 状态 |
|---|---|---|---|---|
| F0 | 真实 CLI 问题到答案文件 | `python -m kgqa.freeform --question "Who directed The Avengers?" --output output/freeform/spine.json` | MUST | 通过 |
| F1 | 自然问法解析、实体消歧和拒答 | `python -m pytest -q tests/test_freeform.py` | MUST | 通过 |
| F2 | 显式网页模式与解析结果 | `python -m pytest -q tests/test_web.py tests/test_freeform.py` | MUST | 通过 |
| F3 | 独立问句导出与后续训练参数 | `python -m pytest -q tests/test_freeform_data.py tests/test_freeform.py` | MUST | 通过 |
| F4 | 完整回归与使用说明 | `python -m pytest -q` | MUST | 通过 |

## 运行记录

开始前已确认原训练使用一次性 `read_jsonl(...qa.jsonl)`；目前训练不具备问句热更新。

F0：命令退出 0，生成 `output/freeform/spine.json`，真实 A0 根据 P57 回答 Joss Whedon；此阶段直通已有基线，不含问法扩展。

F1：31 项检查通过（1.51 秒），F0 再次退出 0。初次检查发现中文相邻 QID 的词边界和 wrote 未覆盖，修复后通过；规则仍限定于声明的单跳意图，未训练新神经模型。

F2：46 项检查通过（9.16 秒）。首次运行遇到既有超长请求测试的 Windows 连接中止，未改动 HTTP 边界处理，重跑整组通过。网页仅在后端声明能力后展示新选择，API 默认仍为 standard。

F3：36 项数据与解析检查通过，连同网页共 51 项通过。完整神经依赖环境另有 4 项 CPU 检查通过，覆盖实际数据/映射目录和训练在模型加载前记录所选 QA/检查点；没有执行 GPU 训练。首次生成器测试缺少夹具三元组文件，补全夹具并在写入前检查源文件。

实际另存 `data/processed/movies_5000_freeform_en_v1`：86974 道问答，train/valid/test 为 73202/6868/6904。原 11 个冻结分发资产全部哈希校验通过。合成改写不能视为独立人工自由问法数据。

F4：完整 CPU 检查 290 项通过，1 个可选神经测试模块因轻量环境未安装 torch 而跳过（13.06 秒）；该模块在完整环境另有 4 项通过（5.50 秒）。Ruff、compileall、actionlint、JS 语法、CLI 和 F0 再运行通过。使用说明与训练交接已写入 `docs/自由问法与训练交接.md`。本阶段只能确认上述执行检查通过，不能据此声称自由问题泛化效果已被证明。

审查修正后：完整轻量检查 **297 项通过，1 个可选模块跳过（13.22 秒）**；可选模块的 4 项 CPU 检查已在完整环境通过。新增评测路径回归，确保配置指向新 QA；保留混合语言实体歧义，拒绝禁映、配音、字幕和故事地点限定词。

真实网页验证：在本轮新建的 7862 端口，`《盗梦空间》是谁执导的？` 被转换为 `Who directed Inception?`，A0 根据 P57 回答 Christopher Nolan。截图保存于 `output/freeform/freeform-ui.png`。原服务不停止；预览服务由本轮创建并在修正后重载。

## 语义约定与效果边界

- A-001：只处理单电影、已有四关系的单意图；多电影/多关系要求澄清，已识别的未支持事实拒答。
- A-002：采用显式规则和词典，有限中文转英文，答案保持英文；适配层不表示旧模型已学会原始自由问句。
- A-003：自由模式与实验标准模式分开，保留原句与标准句，不能把两种入口的分数混为一个基线。
- A-004：新增问句为受控合成改写，继承电影划分并分开句式族；不能替代独立人工自然提问评测。
- A-009：结构输入需要匹配数据；缺少图实体/关系映射时停止，而非静默删除事实。
- A-010：评测依据各运行声明的数据目录，旧配置仍走旧默认目录。
- A-011：对审查发现的混合语言歧义和事实限定词已增加拒答检查；规则不保证识别全部不支持表达。

## 审查记录

依技能完成一次新上下文 Codex 审查，请求模型为 `gpt-6-astra/xhigh`，返回身份为 GPT-6 family（确切 serving variant 不可得）。原始回复完整保存在 `SILENT_ASSUMPTION_SWEEP.json`。该审查发现 2 项 semantic 缺口，已增加 A-011 并修正上述复现用例；执行者另发现并修正自定义运行的评测路径，新增 A-010。

`review_independence = same-family`，`acceptance_status = provisional`。balanced 的一次审查轮已使用，修正后仅由执行者运行回归，没有二次审查，不声称独立审查通过。

## 交付状态

F0–F4 的执行检查均通过，未保留占位实现；语义效果仍受上面约定限制。CLI、网页和下一轮训练参数已可运行，当前 GPU 训练、新模型效果及人工独立评分均未执行。代码在本地 `codex/freeform-movie-questions` 分支分阶段提交，未自动推送或联系同伴。接口选择 A-005–A-008 的来源和默认值完整保存在 `ASSUMPTIONS.md`，ASK=never。

## 延后

开放域问答、多跳、会话记忆、通用中文生成、人工独立自由问法评分、GPU 微调均未纳入本次实现。后续训练可使用本次另存的数据。

## 阻塞

无。
