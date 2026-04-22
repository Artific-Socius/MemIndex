import type {
  Dimensions,
  GroupRow,
  PerFileRow,
  SummaryPayload,
} from "../types";

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort((a, b) => a.localeCompare(b));
}

/** 当 JSON 为旧版（无 dimensions / by_group）时，从 per_file 推导。 */
export function enrichFromPerFile(summary: SummaryPayload): SummaryPayload {
  const rows = summary.per_file ?? [];
  if (!rows.length) {
    return summary;
  }

  let dimensions = summary.dimensions;
  let by_group = summary.by_group;

  const hasMeta =
    rows.some((r) => r.dataset_name != null) ||
    rows.some((r) => r.memory_type != null) ||
    rows.some((r) => r.agent_model != null);

  if (!hasMeta) {
    return summary;
  }

  if (!dimensions) {
    dimensions = {
      datasets: uniqueSorted(
        rows.map((r) => (r.dataset_name?.trim() ? r.dataset_name : "unknown") as string),
      ),
      memories: uniqueSorted(
        rows.map((r) => (r.memory_type?.trim() ? r.memory_type : "unknown") as string),
      ),
      agents: uniqueSorted(
        rows.map((r) => (r.agent_model?.trim() ? r.agent_model : "unknown") as string),
      ),
    };
  }

  if (!by_group) {
    const acc = new Map<string, { p: number; t: number }>();
    for (const r of rows) {
      const ds = r.dataset_name?.trim() || "unknown";
      const mem = r.memory_type?.trim() || "unknown";
      const ag = r.agent_model?.trim() || "unknown";
      const key = `${ds}\x00${mem}\x00${ag}`;
      const cur = acc.get(key) ?? { p: 0, t: 0 };
      cur.p += r.passed;
      cur.t += r.total;
      acc.set(key, cur);
    }
    const out: GroupRow[] = [];
    for (const [key, v] of acc) {
      const [dataset_name, memory_type, agent_model] = key.split("\x00");
      out.push({
        dataset_name,
        memory_type,
        agent_model,
        passed: v.p,
        total: v.t,
        accuracy_percent: v.t > 0 ? (100 * v.p) / v.t : 0,
      });
    }
    out.sort((a, b) => {
      const d = a.dataset_name.localeCompare(b.dataset_name);
      if (d !== 0) return d;
      const m = a.memory_type.localeCompare(b.memory_type);
      if (m !== 0) return m;
      return a.agent_model.localeCompare(b.agent_model);
    });
    by_group = out;
  }

  return { ...summary, dimensions, by_group };
}

export function parseSummaryJson(text: string): SummaryPayload {
  const raw = JSON.parse(text) as unknown;
  if (!raw || typeof raw !== "object") {
    throw new Error("Summary 不是 JSON 对象");
  }
  const s = raw as SummaryPayload;
  if (!Array.isArray(s.per_file)) {
    throw new Error("缺少 per_file 数组");
  }
  if (!s.weighted || typeof s.weighted !== "object") {
    throw new Error("缺少 weighted 对象");
  }
  return enrichFromPerFile(s);
}

export async function fetchDefaultSummary(): Promise<SummaryPayload | null> {
  try {
    const res = await fetch("/summary.json", { cache: "no-store" });
    if (!res.ok) return null;
    const text = await res.text();
    return parseSummaryJson(text);
  } catch {
    return null;
  }
}

export async function loadSummaryFromFile(file: File): Promise<SummaryPayload> {
  const text = await file.text();
  return parseSummaryJson(text);
}

export function filterGroupRows(
  rows: GroupRow[],
  filters: { dataset: string; memory: string; agent: string; query: string },
): GroupRow[] {
  const q = filters.query.trim().toLowerCase();
  return rows.filter((r) => {
    if (filters.dataset && r.dataset_name !== filters.dataset) return false;
    if (filters.memory && r.memory_type !== filters.memory) return false;
    if (filters.agent && r.agent_model !== filters.agent) return false;
    if (!q) return true;
    const hay = `${r.dataset_name} ${r.memory_type} ${r.agent_model}`.toLowerCase();
    return hay.includes(q);
  });
}

export function filterPerFileRows(
  rows: PerFileRow[],
  filters: { dataset: string; memory: string; agent: string; query: string },
): PerFileRow[] {
  const q = filters.query.trim().toLowerCase();
  return rows.filter((r) => {
    const ds = r.dataset_name ?? "unknown";
    const mem = r.memory_type ?? "unknown";
    const ag = r.agent_model ?? "unknown";
    if (filters.dataset && ds !== filters.dataset) return false;
    if (filters.memory && mem !== filters.memory) return false;
    if (filters.agent && ag !== filters.agent) return false;
    if (!q) return true;
    const hay = `${r.basename} ${ds} ${mem} ${ag}`.toLowerCase();
    return hay.includes(q);
  });
}

export function emptyDimensions(): Dimensions {
  return { datasets: [], memories: [], agents: [] };
}
