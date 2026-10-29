# 5000 部电影扩展复评

[结果、方法及分工](../../../reports/实验改进补充记录.md)。

- asset_verification.json：11 个上传资产与两道实际示例校验。
- a0_robustness/：旧链接器的8917题回归及测试扰动。
- unicode_valid_before_v2/、unicode_valid_after/：验证集修复前后。
- unicode_validation_selection.json：验证选择及最终链接源码哈希。
- unicode_test_after/：冻结实现后的完整测试与配对鲁棒性。
- neural_preflight_5000.json：58994题纯文本事实的训练前检查；其源码哈希对应运行当时版本。
- unicode_live_demo.json：实际5000 A0服务的中文别名和未知电影问答。
- improvement_comparison.png：两组实验分别计分的结果图。

本轮没有重新训练模型，数据与权重保持原始字节哈希。
