"""
翻译字典 - 中文到英文

格式: {中文原文: 英文翻译}
支持格式化占位符: {name}, {count} 等
"""

TRANSLATIONS = {
    # ===== 配置显示 =====
    "BoolQ评估配置": "BoolQ Evaluation Config",
    "模型": "Model",
    "评估模式": "Eval Mode",
    "验证答案": "Validate Answer",
    "直接回答": "Direct Answer",
    "风格": "Style",
    "推理": "Reasoning",
    "数据集": "Dataset",
    "并发数": "Concurrency",
    "反转比例": "Reversal Ratio",
    
    # ===== 日志消息 =====
    "实验日志初始化完成": "Experiment logger initialized",
    "日志文件": "Log file",
    "数据文件": "Data file",
    "已加载Prompt配置": "Loaded prompt config",
    "正在加载 BoolQ 数据集 (split: {split})...": "Loading BoolQ dataset (split: {split})...",
    "数据集加载完成, 共 {count} 条数据": "Dataset loaded, {count} records total",
    "Train: {train_count} 条, Validation: {val_count} 条": "Train: {train_count}, Validation: {val_count}",
    "限制数据条数为 {limit}": "Limited to {limit} records",
    "请求限制 {limit} 超过数据集大小 {total}, 使用全部数据": 
        "Requested limit {limit} exceeds dataset size {total}, using all data",
    
    # ===== 脏数据相关 =====
    "发现 {count} 个脏数据文件": "Found {count} dirty data file(s)",
    "脏数据路径不存在: {path}, 跳过过滤": "Dirty data path not found: {path}, skipping filter",
    "脏数据文件夹为空: {path}": "Dirty data folder is empty: {path}",
    "脏数据加载完成: {valid}/{total} 文件, {hashes} 条唯一哈希 ({valid_records} 有效/{invalid} 无效)":
        "Dirty data loaded: {valid}/{total} files, {hashes} unique hashes ({valid_records} valid/{invalid} invalid)",
    "跳过 {count} 条脏数据, 索引: {indices}": "Skipped {count} dirty records, indices: {indices}",
    "脏数据: 已加载 {count} 条哈希, 当前数据集将跳过 {skip} 条":
        "Dirty data: loaded {count} hashes, will skip {skip} in current dataset",
    "未加载到任何有效的脏数据": "No valid dirty data loaded",
    
    # ===== 评估相关 =====
    "开始评估: 模型={model}, 风格={style}": "Starting evaluation: model={model}, style={style}",
    "数据集: {count} 条, 并发数: {concurrency}": "Dataset: {count} records, concurrency: {concurrency}",
    "推理模式: {mode}, 顺序: {order}": "Reasoning mode: {mode}, order: {order}",
    "启用": "Enabled",
    "禁用": "Disabled",
    "评估完成": "Evaluation Complete",
    "评估被用户中断": "Evaluation interrupted by user",
    
    # ===== 进度条 =====
    "评估": "Eval",
    
    # ===== 结果摘要 =====
    "评估结果总结": "Evaluation Summary",
    "总样本数": "Total Samples",
    "有效解析": "Valid Parsed",
    "正确": "Correct",
    "错误": "Wrong",
    "解析/API错误": "Parse/API Errors",
    "准确率": "Accuracy",
    "Token统计": "Token Statistics",
    "输入Token": "Input Tokens",
    "输出Token": "Output Tokens",
    "总Token": "Total Tokens",
    "推理Token": "Reasoning Tokens",
    "成本统计 (USD)": "Cost Statistics (USD)",
    "输入成本": "Input Cost",
    "输出成本": "Output Cost",
    "总成本": "Total Cost",
    "LogProbs统计": "LogProbs Statistics",
    "平均LogProbs (全部)": "Avg LogProbs (All)",
    "平均LogProbs (正确)": "Avg LogProbs (Correct)",
    "平均LogProbs (错误)": "Avg LogProbs (Wrong)",
    
    # ===== 写入器 =====
    "异步结果写入器已启动": "Async result writer started",
    "异步结果写入器已停止, 共写入 {count} 条记录": "Async result writer stopped, {count} records written",
    
    # ===== PromptManager =====
    "PromptManager初始化: mode={mode}, style={style}, reasoning={reasoning}, order={order}":
        "PromptManager initialized: mode={mode}, style={style}, reasoning={reasoning}, order={order}",
    
    # ===== 错误消息 =====
    "API key not found for provider": "API key not found for provider",
    "未知的风格": "Unknown style",
    "未知的评估模式": "Unknown eval mode",
    "找不到约束配置": "Constraint config not found",
    "Prompt配置文件不存在": "Prompt config file not found",
    
    # ===== 绘图工具 =====
    "BoolQ评估结果绘图工具": "BoolQ Evaluation Results Plotter",
    "输入目录路径 (默认: outputs)": "Input directory path (default: outputs)",
    "输出图片文件名 (默认: accuracy_vs_logprob.png)": "Output image filename (default: accuracy_vs_logprob.png)",
    "绘制完成后显示图片": "Show image after plotting",
    "显示所有实验，不去重": "Show all experiments without deduplication",
    "图表标题": "Chart title",
    "输出统计结果到JSON文件": "Output statistics to JSON file",
    "界面语言 (zh/en)": "UI language (zh/en)",
    "找到 {count} 个实验结果文件": "Found {count} experiment result file(s)",
    "去重后保留 {kept}/{total} 个实验（使用 --all 显示全部）":
        "After dedup: {kept}/{total} experiments (use --all to show all)",
    "处理: {filepath}": "Processing: {filepath}",
    "  [跳过] 无有效数据": "  [Skip] No valid data",
    "\n✓ 图表已保存: {output_file}": "\n✓ Chart saved: {output_file}",
    "✓ 统计结果已保存: {path}": "✓ Statistics saved: {path}",
    "[错误] 输入目录不存在: {path}": "[Error] Input directory not found: {path}",
    "[错误] 未找到jsonl文件: {pattern}": "[Error] No jsonl files found: {pattern}",
    "[错误] 无有效实验数据": "[Error] No valid experiment data",
    "[警告] 无法解析文件名: {filepath}": "[Warning] Cannot parse filename: {filepath}",
    "[警告] 加载文件失败 {filepath}: {error}": "[Warning] Failed to load file {filepath}: {error}",
    "使用IQR方法过滤异常值 (Q1-1.5*IQR, Q3+1.5*IQR)": "Filter outliers using IQR method (Q1-1.5*IQR, Q3+1.5*IQR)",
    "IQR倍数 (默认: 1.5)": "IQR multiplier (default: 1.5)",
    "启用IQR异常值过滤 (k={k})": "IQR outlier filtering enabled (k={k})",
    "  过滤异常值: {count}/{total} ({pct:.1f}%)": "  Filtered outliers: {count}/{total} ({pct:.1f}%)",
    
    # ===== 命令行帮助 =====
    "BoolQ LLM评估实验": "BoolQ LLM Evaluation Experiment",
    "模型名称": "Model name",
    "提示词和解析风格": "Prompt and parsing style",
    "评估模式: validate=验证答案正确性, answer=直接回答问题":
        "Eval mode: validate=verify answer correctness, answer=directly answer question",
    "限制数据条数, 0表示全部": "Limit records, 0 means all",
    "启用详细推理": "Enable detailed reasoning",
    "数据集分割": "Dataset split",
    "答案反转比例": "Answer reversal ratio",
    "推理顺序": "Reasoning order",
    "最大并发数": "Max concurrency",
    "语言 (zh/en)": "Language (zh/en)",
    
    # ===== 批量任务 =====
    "批量任务列表": "Batch Task List",
    "共 {count} 个任务": "Total {count} task(s)",
    "任务名称": "Task Name",
    "数据限制": "Data Limit",
    "全部": "All",
    "任务": "Task",
    "批量评估开始": "Batch Evaluation Started",
    "总任务数": "Total Tasks",
    "输出目录": "Output Dir",
    "任务完成": "Task Completed",
    "任务失败": "Task Failed",
    "批量评估完成": "Batch Evaluation Complete",
    "成功": "Success",
    "失败": "Failed",
    "失败任务": "Failed Tasks",
    "没有可执行的任务": "No tasks to execute",
    "配置文件不存在": "Config file not found",
    "请创建配置文件或指定正确的路径": "Please create config file or specify correct path",
    "示例配置文件": "Example config",
    "配置文件解析失败": "Failed to parse config file",
    "无效的任务索引": "Invalid task index",
    
    # ===== 其他 =====
    "已加载Prompt配置": "Loaded prompt config",
}

