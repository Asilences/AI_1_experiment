# Assumption Ledger — 电影目录V2

ASK mode: never。

| ID | 请求未确定的选择 | 采用方案 | Class | Source |
|---|---|---|---|---|
| A-001 | 从完整方案哪里开始执行 | 先完成数据审计、实体确认、类型工具和网页主线；训练留待错误分析与版本冻结之后 | semantic | default |
| A-002 | 新增资料的时间与来源 | 优先复用带API参数、提取时间和revision的原始完整声明缓存，缺失再请求官方API；每条事实注明实际来源时间，不声称全部于今日采集 | semantic | default |
| A-003 | statement rank取舍 | 保留全部非deprecated声明及rank，查询不全局丢弃normal日期/金额；缺失/novalue/somevalue独立保留 | semantic | default |
| A-004 | 200部采样如何分层 | 以V1可见的主要国家、语言、年代分层轮转，未知年代单列；固定42，仅审计覆盖，不声称代表全球电影分布 | semantic | default |
| A-005 | 新入口与旧实验关系 | 单独explore工具接口/模式与独立V2数据；standard/freeform和旧模型文件不改语义 | interface | default |
| A-006 | 新电影识别是否自动近似接受 | 完整明确名称/QID可解析；残缺/错拼/同名用候选确认。检索分数不是概率；未知续集不退化成前作直接回答 | semantic | default |
| A-007 | 生成/语言能力的当前实现 | 首批规则解析与中英证据模板，命名为扩展工具问答；不声称Qwen、向量召回或LoRA已实现 | semantic | default |
| A-008 | 金额、日期及列表的答案定义 | 保留值/精度/单位/限定条件/来源；仅同币种且明确同范围的金额可比较；长列表显式分页；缺失不表示0或否定事实 | semantic | default |
| A-009 | 多跳/推荐目录边界 | 仅当前冻结目录，最多两跳；列出实际路径/依据，未记录不证明没有。推荐先作确定性内容筛选，不宣称个性化排序有效 | semantic | default |
| A-010 | 缺少完整资料时如何扩规模 | 保存采集进度与失败ID；V2只声明实际已处理数量，当前目录内未扩字段电影明确显示缺项；全部5000完成后才宣称全量 | interface | default |
| A-011 | API补名称持续429限流 | 全量新导入改用官方WDQS按250实体查询中英标签，保留查询缓存/提取时间；标签未解析仍显示QID，不造名称 | semantic | default |
| A-012 | WDQS标签是否具有声明修订号 | 结构声明仍用Entity API逐项revision；WDQS仅标签按查询快照溯源，不声称标签有逐项revision | semantic | default |
| A-013 | 官方端点持续429时是否阻塞全部功能 | 用显式`--offline-labels`冻结已有完整声明与已缓存名称，未解析名字保留QID并统计缺失；后续补名必须另建目录，不覆盖冻结文件 | semantic | default |
| A-014 | 实际中文标题只有繁体标签时如何检索 | 使用OpenCC t2s统一索引与查询，保留原始标签；归一后重名继续消歧，不将转换名宣称为官方别名 | semantic | default |

原V1/F1与模型、运行不覆盖。具体缓存/API、年份未知比例与实际覆盖率据执行结果填写。
