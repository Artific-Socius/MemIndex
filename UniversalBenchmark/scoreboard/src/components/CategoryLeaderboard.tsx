import type { CategoryRankRow } from "../data/categoryRanking";

interface Props {
  rows: CategoryRankRow[];
}

export function CategoryLeaderboard({ rows }: Props) {
  if (!rows.length) {
    return (
      <div className="panel categories-panel categories-panel--full">
        <h2 className="categories-heading">题型榜</h2>
        <p className="panel-empty">暂无 categories 数据（缺少 by_question_type）</p>
      </div>
    );
  }

  return (
    <div className="panel categories-panel categories-panel--full">
      <h2 className="categories-heading">题型榜</h2>
      <p className="categories-lead">按准确率降序，总题数相同时题多者优先</p>
      <div className="cat-table-wrap">
        <table className="cat-table cat-table--lg">
          <thead>
            <tr>
              <th className="col-rank">#</th>
              <th>题型</th>
              <th className="num">通过</th>
              <th className="num">题数</th>
              <th className="num">准确率</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.name} className={r.rank <= 3 ? `cat-row-top cat-row-${r.rank}` : ""}>
                <td className="col-rank mono">{r.rank}</td>
                <td className="cat-name">{r.name}</td>
                <td className="num">{r.passed}</td>
                <td className="num">{r.total}</td>
                <td className="num strong">{r.accuracy_percent.toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
