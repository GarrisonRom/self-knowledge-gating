# Scripts

计划中的命令行入口：

- `prepare_data.py`：校验来源清单、清洗数据并生成 manifest。
- `run_experiment.py`：读取配置，执行门控、检索和冲突裁决流程。
- `evaluate.py`：计算回答质量、校准、选择性风险和检索成本。
- `summarize_runs.py`：合并运行元数据，生成表格和图表输入。

脚本应从项目根目录运行，所有路径通过配置或命令行参数传入，不在代码中写入本机绝对路径。
