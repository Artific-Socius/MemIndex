import type { QuestionTypeBlock } from "../types";

export interface CategoryRankRow {
  rank: number;
  name: string;
  passed: number;
  total: number;
  accuracy_percent: number;
}

/** 将 by_question_type 转为按准确率、题数排序的榜单行。 */
export function buildCategoryLeaderboard(
  byQuestionType: Record<string, QuestionTypeBlock> | undefined,
): CategoryRankRow[] {
  if (!byQuestionType || typeof byQuestionType !== "object") {
    return [];
  }
  const entries = Object.entries(byQuestionType).map(([name, v]) => ({
    name,
    passed: v.passed,
    total: v.total,
    accuracy_percent: v.accuracy_percent,
  }));
  entries.sort((a, b) => {
    const d = b.accuracy_percent - a.accuracy_percent;
    if (Math.abs(d) > 1e-9) return d;
    return b.total - a.total;
  });
  return entries.map((e, i) => ({ rank: i + 1, ...e }));
}
