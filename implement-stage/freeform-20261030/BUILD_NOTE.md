# 自由问法构建记录

| Rung | 功能 | 单条验收命令 | Tier | 状态 |
|---|---|---|---|---|
| F0 | 真实 CLI 问题到答案文件 | `python -m kgqa.freeform --question "Who directed The Avengers?" --output output/freeform/spine.json` | MUST | 通过 |
| F1 | 自然问法解析、实体消歧和拒答 | `python -m pytest -q tests/test_freeform.py` | MUST | 待执行 |
| F2 | 显式网页模式与解析结果 | `python -m pytest -q tests/test_web.py tests/test_freeform.py` | MUST | 待执行 |
| F3 | 独立问句导出与后续训练参数 | `python -m pytest -q tests/test_freeform_data.py tests/test_freeform.py` | MUST | 待执行 |
| F4 | 完整回归与使用说明 | `python -m pytest -q` | MUST | 待执行 |

## 运行记录

开始前已确认原训练使用一次性 `read_jsonl(...qa.jsonl)`；目前训练不具备问句热更新。

F0：命令退出 0，生成 `output/freeform/spine.json`，真实 A0 根据 P57 回答 Joss Whedon；此阶段直通已有基线，不含问法扩展。

## 延后

开放域问答、多跳、会话记忆、通用中文生成、人工独立自由问法评分、GPU 微调均未纳入本次实现。后续训练可使用本次另存的数据。

## 阻塞

无。
