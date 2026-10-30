# 自由问法构建记录

| Rung | 功能 | 单条验收命令 | Tier | 状态 |
|---|---|---|---|---|
| F0 | 真实 CLI 问题到答案文件 | `python -m kgqa.freeform --question "Who directed The Avengers?" --output output/freeform/spine.json` | MUST | 通过 |
| F1 | 自然问法解析、实体消歧和拒答 | `python -m pytest -q tests/test_freeform.py` | MUST | 通过 |
| F2 | 显式网页模式与解析结果 | `python -m pytest -q tests/test_web.py tests/test_freeform.py` | MUST | 通过 |
| F3 | 独立问句导出与后续训练参数 | `python -m pytest -q tests/test_freeform_data.py tests/test_freeform.py` | MUST | 通过 |
| F4 | 完整回归与使用说明 | `python -m pytest -q` | MUST | 待执行 |

## 运行记录

开始前已确认原训练使用一次性 `read_jsonl(...qa.jsonl)`；目前训练不具备问句热更新。

F0：命令退出 0，生成 `output/freeform/spine.json`，真实 A0 根据 P57 回答 Joss Whedon；此阶段直通已有基线，不含问法扩展。

F1：31 项检查通过（1.51 秒），F0 再次退出 0。初次检查发现中文相邻 QID 的词边界和 wrote 未覆盖，修复后通过；规则仍限定于声明的单跳意图，未训练新神经模型。

F2：46 项检查通过（9.16 秒）。首次运行遇到既有超长请求测试的 Windows 连接中止，未改动 HTTP 边界处理，重跑整组通过。网页仅在后端声明能力后展示新选择，API 默认仍为 standard。

F3：36 项数据与解析检查通过，连同网页共 51 项通过。完整神经依赖环境另有 4 项 CPU 检查通过，覆盖实际数据/映射目录和训练在模型加载前记录所选 QA/检查点；没有执行 GPU 训练。首次生成器测试缺少夹具三元组文件，补全夹具并在写入前检查源文件。

实际另存 `data/processed/movies_5000_freeform_en_v1`：86974 道问答，train/valid/test 为 73202/6868/6904。原 11 个冻结分发资产全部哈希校验通过。合成改写不能视为独立人工自由问法数据。

## 延后

开放域问答、多跳、会话记忆、通用中文生成、人工独立自由问法评分、GPU 微调均未纳入本次实现。后续训练可使用本次另存的数据。

## 阻塞

无。
