# 电影问答 V2：首批可运行能力

日期：2026-10-30。用户要求：开始执行《开放式电影问答扩展设计》。ASK=never，EFFORT=balanced。

基准提交：9e6d714d23d2464ec934ca4ad9564ef572b9f38f。

目标：先完成 M0—M2 的真实数据与工具主线，给出200部字段审计、独立V2目录、电影候选确认、双语扩展事实/多属性/比较/两跳查询，并接入已有网页。审计通过后扩大到原5000电影目录，不改V1。

输入：V1电影ID、已有Wikidata原始响应，必要时通过官方API补齐；本地只读问答。

输出：`data/processed/movies_v2_20261030/` 的电影/实体/statement/manifest；`output/catalog_v2/` 的审计与验证；新增CLI、工具、网页入口、测试和操作文档。

成功命令：`python -m pytest -q tests/test_v2_data.py tests/test_candidates.py tests/test_explorer.py tests/test_web.py`，加真实数据CLI/API回放。

本轮切分：先交付确定性工具路线；文本剧情采集、向量/拼音检索、Qwen任务规划、推荐相关性标注及训练消融，须在数据/基线完成后继续推进，不把规则系统称为已训练大模型。当前同伴训练不停止、不热更新数据。
