# Results

保存实验配置、运行元数据、决策日志摘要、图表和表格。原始模型输出、完整证据快照和大型索引默认只保存在本地或对象存储中，并通过 manifest 和哈希关联。

建议每次运行至少记录：运行 ID、Git commit、配置文件、模型与检索器版本、数据版本、随机种子、开始时间、结束时间和失败原因（如有）。

门控实验还应记录 coverage、selective risk、unsupported-answer rate、retrieval calls、context/output tokens 和 p50/p95 latency。汇总时保留所有阈值点，再筛选非支配点绘制 Pareto frontier；不要只保存单个最佳 accuracy。
