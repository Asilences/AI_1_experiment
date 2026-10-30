# 神经名称恢复实验

[结果及方法](../../../reports/实验改进补充记录.md)。valid/ 保存验证选择；test/ 与 robustness/ 保存冻结方案后的严格复评、原始输出和替换证据。live_demo.json 是真实B0检查点通过服务接口生成的问答。原始B0成绩继续保存在相邻 runs/B0/。

CLI：python -m kgqa.recovery_experiment --help。复核test/robustness必须提供同输入、分词器与源码哈希的验证报告；不覆盖既有输出目录。
