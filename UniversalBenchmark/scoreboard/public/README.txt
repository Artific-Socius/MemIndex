将 summarize_locomo_results.py 生成的汇总 JSON 复制为本目录下的 summary.json，
即可在运行 npm run dev 时由前端自动加载（GET /summary.json）。

示例（在 UniversalBenchmark 目录）::

  uv run python scripts/summarize_locomo_results.py outputs/locomo --recursive --json-out scoreboard/public/summary.json
