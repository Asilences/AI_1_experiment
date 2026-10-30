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

## 最终验收（2026-10-30）

代码`b973ac3`：全部MUST梯级的工程检查通过。最终全套330 passed、1 skipped（32.58s），skip是CI环境缺少可选神经模块；ruff致命规则、compileall、v2_data/explorer CLI help均exit=0。旧V1目录与模型未覆盖，既有5000发布11文件校验及A0烟雾问答PASS。

真实本地HTTP：`output/catalog_v2/http_replay_final.json`，17/17预期状态通过；另外核对导演实体、共同项双片证据、确认后保留两片、翻页无重叠及来源revision/提取时间7项检查通过。不是17题准确率实验。原`http_replay.json`保留修复前失败；`http_replay_expectation_errors.json`保留验收载荷错误：错误假定目录包含两个Dune版本，且误用Q172241。改用已核实的两个The Thing版本后完成双片确认测试，未为通过而改被测代码。

浏览器已验证：错拼Inceptoin候选→点击Inception→编剧Christopher Nolan；The Thing同名候选→确认1982版本→与Inception双片导演同时保留；The Avengers出演53项，首1—30、次31—53；中文导演+编剧输出与2条可点击Wikidata修订来源。截图在`output/catalog_v2/ui-candidate-final.png`、`ui-confirmed-final.png`、`ui-final.png`。预览`http://127.0.0.1:7863/`，勾选“扩展事实问答”。当前服务会话77918，原7862不变；无需当前训练完成即可使用。

第二轮假设审阅clean：未声明0、stale rows 0。第一轮有1个interface缺项已补A-020，5个语义反例在审阅期间修复并记A-015—A-019。实际身份仅能核实GPT-6系列，不能把请求gpt-6-astra当成观测身份；同系列验收状态provisional。第二轮之后另修复片名内部and误澄清（A-021），由执行者新增回归及实际Harry Potter查询验证，未进行第三轮审阅。审阅不证明方法有效。

本批无运行中的占位实现。未开放的文本剧情、角色/地区/版本过滤、推荐、Qwen/LoRA及图融合列为明确未支持，不输出伪结果。数据有5000电影/170516声明，标签缺46017；官方API/WDQS限流429导致全量以offline-labels冻结，没有后台补名任务。补名必须新建版本。未租GPU、未做真人标注、未完成R003/正式400题实验；执行表按IN_PROGRESS记录，不能标完成。

## 本批语义假设（完整列出）

以下取自实施前及修复时的ledger；ASK=never。Source=sweep表示审阅发现，default表示执行者决定，均未冒称用户已逐项指定。

| ID | 请求未确定的选择 | 采用方案 | Class | Source |
|---|---|---|---|---|
| A-001 | 从完整方案哪里开始执行 | 先完成数据审计、实体确认、类型工具和网页主线；训练留待错误分析与版本冻结之后 | semantic | default |
| A-002 | 新增资料的时间与来源 | 优先复用带API参数、提取时间和revision的原始完整声明缓存，缺失再请求官方API；每条事实注明实际来源时间，不声称全部于今日采集 | semantic | default |
| A-003 | statement rank取舍 | 保留全部非deprecated声明及rank，查询不全局丢弃normal日期/金额；缺失/novalue/somevalue独立保留 | semantic | default |
| A-004 | 200部采样如何分层 | 以V1可见的主要国家、语言、年代分层轮转，未知年代单列；固定42，仅审计覆盖，不声称代表全球电影分布 | semantic | default |
| A-006 | 新电影识别是否自动近似接受 | 完整明确名称/QID可解析；残缺/错拼/同名用候选确认。检索分数不是概率；未知续集不退化成前作直接回答 | semantic | default |
| A-007 | 生成/语言能力的当前实现 | 首批规则解析与中英证据模板，命名为扩展工具问答；不声称Qwen、向量召回或LoRA已实现 | semantic | default |
| A-008 | 金额、日期及列表的答案定义 | 保留值/精度/单位/限定条件/来源；仅同币种且明确同范围的金额可比较；长列表显式分页；缺失不表示0或否定事实 | semantic | default |
| A-009 | 多跳/推荐目录边界 | 仅当前冻结目录，最多两跳；列出实际路径/依据，未记录不证明没有。本轮推荐尚不开放，后续先作内容筛选并验证相关性 | semantic | default |
| A-011 | API补名称持续429限流 | 全量新导入改用官方WDQS按250实体查询中英标签，保留查询缓存/提取时间；标签未解析仍显示QID，不造名称 | semantic | default |
| A-012 | WDQS标签是否具有声明修订号 | 结构声明仍用Entity API逐项revision；WDQS仅标签按查询快照溯源，不声称标签有逐项revision | semantic | default |
| A-013 | 官方端点持续429时是否阻塞全部功能 | 用显式`--offline-labels`冻结已有完整声明与已缓存名称，未解析名字保留QID并统计缺失；后续补名必须另建目录，不覆盖冻结文件 | semantic | default |
| A-014 | 实际中文标题只有繁体标签时如何检索 | 使用OpenCC t2s统一索引与查询，保留原始标签；归一后重名继续消歧，不将转换名宣称为官方别名 | semantic | default |
| A-015 | 多属性问题包含另一个未识别片名时如何回答 | 对连词后未消费的实体/条件先澄清，不仅回答已识别电影；确认QID与已识别电影合并，不能丢掉原电影 | semantic | sweep |
| A-016 | 未知非数字续集或副标题被较短标题覆盖 | Part/Chapter/罗马序号及冒号副标题触发确认，不自动用旧作的事实回答 | semantic | sweep |
| A-017 | 角色问题是否能用演员列表代替 | played/portrayed/角色/饰演等约束目前明确不支持，不把全部演员当作特定角色的答案 | semantic | sweep |
| A-018 | 未枚举地区可否混入上映/票房问法 | 去除片名、属性与支持的问句语法后，日期/金额问题仍有未消费文字则拒绝筛选；不靠有限国家名单放行其他地区 | semantic | sweep |
| A-019 | 同名电影与明确电影共现的确认 | 保留已识别电影，确认候选只补充待定电影；多个同名槽位暂要求明确QID，不能一次点击默默确定全部 | semantic | sweep |
| A-021 | 片名自带and/和是否意味着第二电影 | 只检查屏蔽完整片名后剩余的连词与未消费条件，不用原问题连词否定完整单片查询；网页验收中发现后修复 | semantic | default |

## 接口假设

A-005：独立explore接口/模式，保留原标准与四关系自然问法。A-010：只有全部5000实际导入才称全量，失败ID/覆盖统计须保留。A-020：问题任意位置有汉字即用中文模板，否则英文。目前共3个interface项（2 default、1 sweep），详细调用与数据schema见SPEC和ledger。
