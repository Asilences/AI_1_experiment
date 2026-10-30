# Build Note — 电影目录V2

| Rung | 功能 | 验收命令 | Tier | 状态 |
|---|---|---|---|---|
| F0 | V1目录→确定性pilot选择→独立输出 | `python -m kgqa.v2_data select --output output/catalog_v2/selection.json` | MUST | PASS |
| F1 | 真实statement导入、200部审计、版本化完整目录 | `python -m pytest -q tests/test_v2_data.py` | MUST | PASS |
| F2 | 中英标题/片段/错拼候选与明确确认 | `python -m pytest -q tests/test_candidates.py` | MUST | PASS |
| F3 | 类型事实、多属性、集合比较、两跳与证据 | `python -m pytest -q tests/test_explorer.py` | MUST | PASS |
| F4 | 网页入口、确认/证据/分页、回归与使用文档 | `python -m pytest -q tests/test_web.py tests/test_explorer.py tests/test_candidates.py tests/test_v2_data.py` | MUST | PASS |

## Run record

开工前检查：基准9e6d714；原追踪文件无修改。新增设计文件尚未提交。原始扩展API缓存可复用，覆盖数量待扫描。

F0：运行表中命令，exit=0，输出200个电影ID及V1校验值，`output/catalog_v2/selection.json`。没有占位事实或虚构分数。

F1：4个导入、来源/rank、限定金额与缺失值测试通过，ruff致命错误检查通过。真实200部导入已找到完整声明，正在补齐4234个新增实体名称；正式冻结/覆盖统计随采集结束补记。此PASS仅指加载器测试，不代表数据采集已经结束。

F2：候选检索/中英别名/同名消歧/未知数字续集/短词误匹配测试通过。累计7 passed，F0选择命令再次exit=0。近似匹配只产生候选，不进入事实回答。

F3：21个累计测试通过。查询器核对冻结manifest；输出类型化事实、原限定条件及来源；多属性缺项单列；共同演员与导演两跳有证据路径；金额范围不明不比较。剧情、预算、主演、版本、地域/日期过滤和其他未实现条件明确拒绝，不降级成四关系答案。

F4：首轮相关测试37 passed；全套回归321 passed/1 optional neural skip（29.45s），原V1发布11文件校验和A0真实烟雾PASS。真实API回放发现繁简中文标题及directors复数两个解析缺口，已加OpenCC与复数规则、相应回归测试。最终数量和UI验证另行补记。

数据：pilot 200部/7037声明完成；全量5000部/170516声明/322.65 MiB冻结。全量名称46017个待补齐。Entity API和WDQS多次返回429，遵守退避；停止本任务的长时间远程补名进程，转显式offline-labels保留完整事实与已有名称；没有停止其他训练进程。缓存保留，可用新目录后续补名。

真实API回放：`output/catalog_v2/http_replay.json`；修复前的两项解析问题如上，不把它们隐藏或写成全通过。页面本轮端口7863，旧7862不改。

审阅修复：第一轮反例涉及未知副标题、角色约束、遗漏第二电影、地区过滤、候选确认保留已有电影；已修复并增加回归。混合语言选择策略补记A-020。审阅原文保存在`SILENT_ASSUMPTION_SWEEP_ROUND1.json`，同GPT-6系列审查仅暂定。修复后全套`329 passed, 1 skipped`（30.80s）；skip为CI环境未安装的可选神经依赖。ruff E9/F63/F7/F82通过。网页和第二轮复核待记。

## Deferred

完整设计中的文本剧情、向量/拼音模型、Qwen规划器与训练、推荐真人相关性实验：依赖数据主线与基线，不冒充本轮已完成。

## Blockers

暂无。原`reviews/`是既有本地审计资料，不纳入提交。
