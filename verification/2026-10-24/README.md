# 2026年10月24日代码验收记录

本目录保存周新永个人工作提交前的实际自动检查摘要。执行环境为新建的独立 CPU Python 3.12 环境，安装 `requirements-test.txt` 中固定的依赖；代码副本仅包含版本控制中的文件和本次新增源文件，运行前没有处理后数据、模型及历史运行目录。

- `pytest.log`：181 项测试通过。
- `syntax.log`、`ruff.log`、`powershell.log`：Python 编译、致命代码错误及 PowerShell 语法检查通过。编译检查成功时日志为空。
- `acceptance.log`、`acceptance_summary.json`：真实离线验收成功，13 项行为检查通过；10 部虚构电影，30 道完整测试题，五组配置各三次推理。
- `acceptance_examples.json`：实际生成的四类关系、多答案、别名与拒答例子。
- `acceptance_environment.json`：本轮隔离环境的软件版本。它与10月19日的正式计算环境分别记录。
- `demo.log`：演示从验收目录加载指定 A0 模型和数据，返回电影、关系、证据及完整句回答。
- `commands.json`：实际执行命令及退出码。
- `formal_regression.json`：使用10月19日既有模型和真实快照复核新代码，未重新训练；保留当次数据与源代码哈希。

验收夹具只检查代码运行与预期行为。正式200部Wikidata电影的原始训练和参数实验记录仍在 `verification/2026-10-19/`。完整数据、模型及运行目录未进入提交包，操作入口见 `docs/提交验收说明.md`。
