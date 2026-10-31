# 实现选择记录

ASK 模式：never。依据用户授权及既有方案自主实施；以下均在依赖实现前声明。

| ID | 未完全指定的选择 | 采用方式 | 类别 | 来源 |
|---|---|---|---|---|
| A001 | 具体冻结模型与部署 | 优先部署 Qwen2.5-3B-Instruct，本地独立 HTTP 服务；现有 Transformers 可读架构，缓存 H 盘；若质量不足再评估同系列更大模型 | semantic | default |
| A002 | 计划接口 | 第一版采用 flat `kgqa-plan/1`，显式 movie_ids/properties/operation/status/constraints，复用既有算子，不执行任意代码或开放DAG | interface | default |
| A003 | 新系统身份 | P0独立入口；不把其成绩归为B0，不覆盖旧四关系实验 | semantic | user / prior design |
| A004 | 模糊实体与未知事实 | 候选确认，未支持条件明确拒绝，缺边不推断事实为假 | semantic | user / prior design |
| A005 | 本轮评测性质 | 新程序编写问题为开发验收；真实模型输出与mock协议测试分别保存；独立人工测试未完成就如实标记 | semantic | default |
| A006 | 资源 | 本地可用RAM仅约2.45GiB，改用已授权服务器GPU1，冻结3B模型FP16推理、分配器25%、CPU线程2，SSH回环隧道访问；不动其他进程 | interface | default |
| A007 | 事实回答核验 | 类型化结果与引用检查，保留原文；不确定的生成转明确标注模板兜底；自动检查不声称通用事实验证 | semantic | default |
| A008 | 接口默认 | 本地127.0.0.1:7870/v1；env可配置模型服务；API密钥不进日志/前端 | interface | default |
| A009 | 规划与候选绑定 | 有精确片名时提供候选卡；缺片名时先由冻结模型抽取原问题中的片名片段，再用既有索引召回。非精确候选需用户确认；原问题及未解析条件始终交规划器 | semantic | default |
| A010 | 漏槽位与代词 | 用整句精确片名及标题引号检查模型抽取遗漏；短英文普通词在未加引号时不授权电影链接，防止it代词变电影It | semantic | default |
| A011 | 漏执行条件 | 除模型计划外独立保守检查显式日期/排除/地区等未实现条件，先遮蔽片名；可能增加不必要拒答，须纳入开发验收 | semantic | default |
| A012 | 生成可见结果预算 | 超过10条声明、日期/金额/未知值或尚需分页的结果直接用完整句模板，保留原限定条件；此类结果计入模板比例 | semantic | default |
