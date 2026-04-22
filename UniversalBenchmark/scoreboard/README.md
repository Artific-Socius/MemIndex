# Scoreboard

`UniversalBenchmark` 结果汇总可视化：读取 `scripts/summarize_locomo_results.py` 导出的 JSON，以 **配置总榜**（`by_group`）展示 **数据集 / Memory / Agent 模型** 与分数；下方 **全宽题型榜**（`by_question_type`）展示各题型排行。

## 生成汇总 JSON

在 `UniversalBenchmark` 目录执行：

```bash
uv run python scripts/summarize_locomo_results.py outputs/locomo --recursive ^
  --json-out scoreboard/public/summary.json
```

Windows PowerShell 可将 `^` 换为 `` ` `` 或写成一行。

## 汇总字段（与打榜相关）

| 字段 | 说明 |
|------|------|
| `schema_version` | 当前为 `2` |
| `weighted` | 全局加权通过 / 总题 / 准确率 |
| `by_question_type` | 各题型 passed / total / accuracy，前端 **题型榜**（全宽大表）使用 |
| `dimensions` | `{ datasets, memories, agents }` 去重列表，供筛选 |
| `by_group` | 按 `dataset_name + memory_type + agent_model` 聚合 |
| `per_file[]` | 每文件一行（汇总脚本生成用）；**前端不再展示逐文件表** |

### Replay 时的 Agent 模型

当 `per_file.source === "replay"` 时，`agent_model` **优先为重放评测所用模型**，读取顺序大致为：

1. `metadata.replay.replay_model`（以及同对象内可能的 `model`）
2. `aggregate.extra.replay_model`，或 `aggregate.extra.replay` 内的 `replay_model` / `model`
3. 回退：`run_config.model`
4. 再回退：解析 `agent_identifier`

非 replay 口径时仍以 `run_config.model` 为主（与原先一致）。

## 启动前端

```bash
cd scoreboard
npm install
npm run dev
```

浏览器打开终端提示的本地地址（默认端口 `5174`）。

- 若存在 `public/summary.json`，页面会自动 `fetch('/summary.json')`。
- 可使用顶部 **导入 JSON** 选择任意路径的汇总文件。

## UI 说明

- **浅色扁平**（OpenAI / ChatGPT 风格）：白与浅灰背景、细边框、绿色少量强调、无渐变装饰与侧栏挤压。
- **布局**：顶部 KPI → 筛选条 → **配置总榜**（可排序，# 名次）→ **全宽题型榜**（更大字号与行高）。

## 生产构建

```bash
npm run build
npm run preview
```

构建产物在 `scoreboard/dist/`，可挂到任意静态文件服务器。
