import { useCallback, useEffect, useMemo, useState } from "react";
import { CategoryLeaderboard } from "./components/CategoryLeaderboard";
import { GroupScoreTable, toggleSortDir, type GroupSortKey } from "./components/ScoreTable";
import { buildCategoryLeaderboard } from "./data/categoryRanking";
import {
  emptyDimensions,
  fetchDefaultSummary,
  filterGroupRows,
  loadSummaryFromFile,
} from "./data/loadSummary";
import type { Dimensions, SummaryPayload } from "./types";

export default function App() {
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [loadNote, setLoadNote] = useState<string>("正在加载 /summary.json …");
  const [dataset, setDataset] = useState("");
  const [memory, setMemory] = useState("");
  const [agent, setAgent] = useState("");
  const [query, setQuery] = useState("");

  const [gSortKey, setGSortKey] = useState<GroupSortKey>("accuracy_percent");
  const [gSortDir, setGSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const s = await fetchDefaultSummary();
      if (cancelled) return;
      if (s) {
        setSummary(s);
        setLoadNote("已加载 summary（/summary.json）。");
      } else {
        setLoadNote("未找到 /summary.json：请将汇总 JSON 放到 public/summary.json，或使用「导入」。");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const dimensions: Dimensions = useMemo(
    () => summary?.dimensions ?? emptyDimensions(),
    [summary],
  );

  const categoryRows = useMemo(
    () => buildCategoryLeaderboard(summary?.by_question_type),
    [summary],
  );

  const groupRows = useMemo(() => {
    const raw = summary?.by_group ?? [];
    return filterGroupRows(raw, { dataset, memory, agent, query });
  }, [summary, dataset, memory, agent, query]);

  const weighted = summary?.weighted;
  const hasGroup = (summary?.by_group?.length ?? 0) > 0;

  const onPickFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    try {
      const s = await loadSummaryFromFile(f);
      setSummary(s);
      setLoadNote(`已导入：${f.name}`);
    } catch (err) {
      setLoadNote(`解析失败：${err instanceof Error ? err.message : String(err)}`);
    }
  }, []);

  const onGroupSort = useCallback(
    (k: GroupSortKey) => {
      if (k === gSortKey) {
        setGSortDir((d) => toggleSortDir(d));
      } else {
        setGSortDir(k === "accuracy_percent" ? "desc" : "asc");
      }
      setGSortKey(k);
    },
    [gSortKey],
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div>
            <div className="brand-title">Scoreboard</div>
            <div className="brand-sub">Benchmark 汇总</div>
          </div>
        </div>
        <label className="btn-import">
          <input type="file" accept="application/json,.json" onChange={onPickFile} />
          导入 JSON
        </label>
      </header>

      <p className="load-line">{loadNote}</p>

      {summary ? (
        <>
          <section className="kpi-strip">
            <div className="kpi-card">
              <div className="kpi-label">加权准确率</div>
              <div className="kpi-value kpi-em">
                {weighted ? `${weighted.accuracy_percent.toFixed(2)}%` : "—"}
              </div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">总题数</div>
              <div className="kpi-value tabular">{weighted?.total ?? "—"}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">通过</div>
              <div className="kpi-value tabular">{weighted?.passed ?? "—"}</div>
            </div>
            <div className="kpi-card">
              <div className="kpi-label">样本文件</div>
              <div className="kpi-value tabular">{summary.file_count}</div>
            </div>
          </section>

          <section className="filters-bar panel">
            <div className="filters-inner">
              <label className="fld">
                <span>数据集</span>
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                  className="fld-control"
                >
                  <option value="">全部</option>
                  {dimensions.datasets.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </label>
              <label className="fld">
                <span>Memory</span>
                <select
                  value={memory}
                  onChange={(e) => setMemory(e.target.value)}
                  className="fld-control"
                >
                  <option value="">全部</option>
                  {dimensions.memories.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>
              <label className="fld">
                <span>Agent</span>
                <select
                  value={agent}
                  onChange={(e) => setAgent(e.target.value)}
                  className="fld-control"
                >
                  <option value="">全部</option>
                  {dimensions.agents.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </label>
              <label className="fld fld-grow">
                <span>搜索</span>
                <input
                  className="fld-control"
                  placeholder="筛选关键词…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </label>
            </div>
          </section>

          <section className="stack">
            <div className="panel section-head">
              <h2 className="section-title">配置总榜</h2>
              <p className="section-meta mono">{summary.input_dir}</p>
            </div>

            {hasGroup ? (
              <GroupScoreTable
                rows={groupRows}
                sortKey={gSortKey}
                sortDir={gSortDir}
                onSort={onGroupSort}
              />
            ) : (
              <div className="panel empty-inline">
                <p>当前汇总无 by_group 数据（可能未生成 schema v2 或筛选后为空）。</p>
              </div>
            )}

            <p className="foot-meta">
              schema {summary.schema_version ?? 1} · 配置行 {groupRows.length}
              {summary.prefer_replay ? " · replay 优先" : ""}
            </p>

            <div className="categories-block">
              <CategoryLeaderboard rows={categoryRows} />
            </div>
          </section>
        </>
      ) : (
        <section className="panel empty-state">
          <h2>暂无数据</h2>
          <p>{loadNote}</p>
        </section>
      )}
    </div>
  );
}
