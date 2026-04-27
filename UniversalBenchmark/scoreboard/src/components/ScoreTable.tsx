import type { GroupRow } from "../types";

export type GroupSortKey =
  | "dataset_name"
  | "memory_type"
  | "agent_model"
  | "passed"
  | "total"
  | "accuracy_percent";

type SortDir = "asc" | "desc";

export function toggleSortDir(prev: SortDir): SortDir {
  return prev === "asc" ? "desc" : "asc";
}

function ThSort<K extends string>({
  label,
  colKey,
  active,
  dir,
  onSort,
}: {
  label: string;
  colKey: K;
  active: K;
  dir: SortDir;
  onSort: (k: K) => void;
}) {
  const isActive = active === colKey;
  return (
    <th>
      <button
        type="button"
        className={`th-btn ${isActive ? "active" : ""}`}
        onClick={() => onSort(colKey)}
      >
        {label}
        {isActive ? <span className="th-dir">{dir === "asc" ? "↑" : "↓"}</span> : null}
      </button>
    </th>
  );
}

function sortGroupRows(
  rows: GroupRow[],
  key: GroupSortKey,
  dir: SortDir,
): GroupRow[] {
  const mul = dir === "asc" ? 1 : -1;
  const copy = [...rows];
  copy.sort((a, b) => {
    const va = a[key];
    const vb = b[key];
    if (typeof va === "number" && typeof vb === "number") {
      return (va - vb) * mul;
    }
    return String(va).localeCompare(String(vb)) * mul;
  });
  return copy;
}

interface GroupTableProps {
  rows: GroupRow[];
  sortKey: GroupSortKey;
  sortDir: SortDir;
  onSort: (key: GroupSortKey) => void;
}

export function GroupScoreTable({ rows, sortKey, sortDir, onSort }: GroupTableProps) {
  const sorted = sortGroupRows(rows, sortKey, sortDir);
  return (
    <div className="table-wrap panel table-panel">
      <table className="score-table">
        <thead>
          <tr>
            <th className="col-rank th-rank">#</th>
            <ThSort<GroupSortKey>
              label="数据集"
              colKey="dataset_name"
              active={sortKey}
              dir={sortDir}
              onSort={onSort}
            />
            <ThSort<GroupSortKey>
              label="Memory"
              colKey="memory_type"
              active={sortKey}
              dir={sortDir}
              onSort={onSort}
            />
            <ThSort<GroupSortKey>
              label="Agent 模型"
              colKey="agent_model"
              active={sortKey}
              dir={sortDir}
              onSort={onSort}
            />
            <ThSort<GroupSortKey>
              label="通过"
              colKey="passed"
              active={sortKey}
              dir={sortDir}
              onSort={onSort}
            />
            <ThSort<GroupSortKey>
              label="总题"
              colKey="total"
              active={sortKey}
              dir={sortDir}
              onSort={onSort}
            />
            <ThSort<GroupSortKey>
              label="准确率"
              colKey="accuracy_percent"
              active={sortKey}
              dir={sortDir}
              onSort={onSort}
            />
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr
              key={`${r.dataset_name}|${r.memory_type}|${r.agent_model}`}
              className={i < 3 ? `row-top row-top-${i + 1}` : ""}
            >
              <td className="col-rank mono muted">{i + 1}</td>
              <td className="mono">{r.dataset_name}</td>
              <td>
                <span className="tag">{r.memory_type}</span>
              </td>
              <td className="mono cell-model">{r.agent_model}</td>
              <td className="num">{r.passed}</td>
              <td className="num">{r.total}</td>
              <td className="num strong">{r.accuracy_percent.toFixed(2)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
