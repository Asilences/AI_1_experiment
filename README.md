# Wikidata 电影知识图谱问答

本项目为《人工智能 I》课程实验，实现并比较两类基于 Wikidata 电影知识图谱的单跳问答系统：

- **A0 管道式系统**：实体链接、TF-IDF 关系分类、图谱事实查询和模板化答案生成。
- **B0–B3 神经生成系统**：使用图谱事实与冻结的 TransE 结构向量作为条件，基于 FLAN-T5-small 直接生成答案。
- **C0 消融系统**：保留图谱事实文本，但不使用 TransE 向量。

虚拟环境、缓存、模型和临时文件均保存在 H 盘项目目录中，避免占用 C 盘空间。

## 环境配置

```powershell
cd H:\AI1experiment
python -m venv .venv
. .\scripts\environment.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 交互式问答演示

已训练的模型和处理后的数据位于本地项目目录中。进入项目目录后，运行：

```powershell
cd H:\AI1experiment
.\scripts\run_demo.cmd
```

`.cmd` 启动器只会在本次运行中临时绕过 PowerShell 脚本执行策略，不会修改系统策略。

出现 `Question>` 后输入英文电影问题，输入 `exit` 退出。示例：

```text
Who directed Shutter Island?
What genre is Boss Level?
```

默认的 `both` 模式会并排展示 A0 与验证集选出的最佳神经模型 B2，并显示电影实体链接、关系预测、知识图谱证据、最终答案和回答延迟。

只运行一个系统：

```powershell
.\scripts\run_demo.cmd -System a0
.\scripts\run_demo.cmd -System best
```

直接运行一道题：

```powershell
.\scripts\run_demo.cmd -Question "Who directed Shutter Island?" -System both
```

系统当前支持四类英文单跳问题：导演、类型、制作国家和原始语言。问题中的电影必须存在于冻结的本地知识库；实体无法链接或缺少事实时，系统会明确拒答。

## 小规模验证实验

```powershell
.\scripts\run_experiment.ps1 -Mode smoke -Target 50
```

## 完整实验

```powershell
.\scripts\run_experiment.ps1 -Mode full -Target 1000
```

Wikidata 采集器会缓存成功请求并支持断点续跑。完整实验需要下载模型并执行多次微调，耗时可能较长。运行日志保存在 `runs/`，图表和评估表保存在 `output/`。

自动生成的 `human_evaluation.csv` 人工评分列保持为空，必须由实验者填写；自动指标不作为人工事实核查的替代。
