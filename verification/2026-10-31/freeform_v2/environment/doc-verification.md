# 独立文档执行复核

按run-experiment环境契约由独立代理原样执行文档命令。两次退出码均为0；先384旧版本，后512最终版本。实际设备V100、GPU1、线程2、分配器0.25。512是输入上限，本次真实长事实小批实际372token、24条图前缀；不声称这是所有训练batch的最坏显存证明。固定分配器仍将限制正式训练，超限会失败。

512实测allocated2995.284MiB、reserved3464MiB，进程显存采样4468MiB。实际cap8519974912字节=8125.28125MiB。复核者未改旧数据、未保存微型更新到发布检查点、未启动正式训练。原始响应本地保存于`.aris/traces/v100-doc-verification/20261031/response.md`；证据JSON随本目录提交。后续将评测batch独立设为32以提升吞吐；训练microbatch及witness命令均保持8。

第三次最终复核：启动检查发现训练自动精度检测包含BF16软件模拟后，作者中断新尝试并修正检测。独立代理按更新文档原样执行`v100_freeform_v2_fp32_doccheck_20261031`，退出0，新增`TRAINING_PRECISION_WITNESS selected=float32 capability=(7, 0)`与`launch.training_precision=float32`匹配；线程2、输入512、分配器25%，显存实测与上述相同。最终证据为本目录`fp32-witness-*`文件。前两次只证明手动float32小批可跑，不能替代最终训练精度验证；三次原始响应分别保留。
