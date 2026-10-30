# A0补充阈值运行记录

本目录保存同一冻结A0与完整8917道测试题的实际补充推理记录。原四个预设阈值未观察到成绩变化后，根据已保存的置信度诊断增加0.995、0.998、0.999，并在同一进程重复0.30对照。这是诊断后的敏感性补充，不是预先注册的泛化实验；没有重新拟合模型，也不根据测试成绩重选默认阈值，默认值保持0.30。

| 阈值 | 测试题数 | 实体Macro-F1（%） | 可回答题误拒答数 |
|---|---:|---:|---:|
| 0.300 | 8917 | 100.000000 | 0 |
| 0.995 | 8917 | 100.000000 | 0 |
| 0.998 | 8917 | 84.120220 | 1416 |
| 0.999 | 8917 | 9.094987 | 8106 |

四份`threshold_*.jsonl`为实际逐题预测，对应`*_metrics.json`保留实际评分、误拒答和耗时；`summary.json`保留环境、输入/模型/源码哈希及前后不变检查。这九个文件与原始运行输出逐字节一致，未重新生成或修改指标。数据和A0权重保持不变。

`runner_original.py`是实际执行脚本的字节副本，SHA-256为`eb83044184dcafa8e036914526756921bbd44921d82dbd6c94ebff4e6950da4d`。其原始相对目录按当时运行位置设置，归档后不应直接从本目录执行。

公开复现入口为[run_report_threshold_sensitivity.py](../../benchmarks/run_report_threshold_sensitivity.py)，SHA-256为`59c6eb74c93b56f4eeb5432f5386636092405f1b3ef472534c28fa142ea9ce11`。该文件仅把运行根目录、输入/输出目录改为公开仓库路径与CLI参数，并使外部输入的路径记录使用`data/`或`model/`相对键；四个阈值、8917题断言、预测循环、评分、两线程限制和输入/源码不变检查均沿用原脚本。公开入口的源码哈希与实际原脚本不同，不能将其称为原始运行脚本。逐文件哈希见[publication_provenance.json](publication_provenance.json)。

在源码仓库根目录、Python 3.12及`requirements-test.txt`环境中复现，输出目录必须不存在：

```powershell
python benchmarks/run_report_threshold_sensitivity.py --output-dir output/report_threshold_sensitivity_01
```

默认读取`data/processed/movies_5000_20261029`与`models/pipeline_5000_20261029/model.pkl`。使用独立完整运行包时，需将该公开脚本放入包内`benchmarks/`，并明确指定配对默认资产：

```powershell
python benchmarks/run_report_threshold_sensitivity.py --data-dir data/processed --model-path models/pipeline/model.pkl --output-dir output/report_threshold_sensitivity_01
```

原四阈值记录与主实验仍保留各自上下文。本补充属于报告中的推理敏感性分析，不改写原始选模、主实验指标或已发布运行包快照。
