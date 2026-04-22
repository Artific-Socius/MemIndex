import json
import codecs
import argparse
import os

def analyze(json_file_path, output_dir):
    with codecs.open(json_file_path, 'r', 'utf-8') as f:
        data = json.load(f)

    stats = {}
    wrong_questions = []

    scenarios = data.get("scenarios", [])
    for scenario in scenarios:
        for turn in scenario.get("turns", []):
            if turn.get("type") != "evaluation":
                continue
                
            metadata = turn.get("metadata", {})
            evidence = metadata.get("evidence", {})
            payload = evidence.get("payload", {})
            q_type = payload.get("question_type", "unknown")
            
            score_info = turn.get("score", {})
            passed = score_info.get("passed", False)
            
            if q_type not in stats:
                stats[q_type] = {"total": 0, "correct": 0}
            
            stats[q_type]["total"] += 1
            if passed:
                stats[q_type]["correct"] += 1
            else:
                message_trace = turn.get("message_trace", {})
                extra = message_trace.get("extra", {})
                messages = extra.get("query_ready_messages_snapshot", [])
                
                wrong_questions.append({
                    "turn_index": turn.get("index"),
                    "question": metadata.get("question_text") or turn.get("user_input"),
                    "agent_answer": turn.get("response"),
                    "ground_truth": metadata.get("ground_truth"),
                    "score": score_info.get("score"),
                    "question_type": q_type,
                    "messages": messages
                })

    # Save summary data
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'wrong_questions_summary.json')
    with codecs.open(summary_path, 'w', 'utf-8') as f:
        json.dump(wrong_questions, f, ensure_ascii=False, indent=2)

    # Generate Markdown Report
    report_path = os.path.join(output_dir, 'wrong_questions_analysis.md')
    with codecs.open(report_path, 'w', 'utf-8') as f:
        f.write("# LoCoMo Benchmark 错题分析报告\n\n")
        f.write("## 1. 各题型准确率统计\n\n")
        f.write("| Question Type | Total | Correct | Wrong | Accuracy |\n")
        f.write("|---|---|---|---|---|\n")
        for q_type, s in stats.items():
            acc = (s["correct"] / s["total"]) * 100 if s["total"] > 0 else 0
            f.write(f"| {q_type} | {s['total']} | {s['correct']} | {s['total'] - s['correct']} | {acc:.2f}% |\n")
        
        f.write(f"\n**总计错题数量**: {len(wrong_questions)}\n\n")
        
        f.write("## 2. 详细错题清单与启示\n\n")
        for i, q in enumerate(wrong_questions):
            agent_answer = str(q['agent_answer'])
            q_type = q['question_type']
            
            is_dont_know = False
            lower_ans = agent_answer.lower()
            if '没有提到' in lower_ans or '不知道' in lower_ans or 'not mentioned' in lower_ans or 'i need more' in lower_ans:
                is_dont_know = True
                
            if is_dont_know and q_type != 'adversarial':
                reason = "模型说不知道"
            elif is_dont_know and q_type == 'adversarial':
                reason = "回答错了（召唤给了真实情况，但模型被误导或输出格式验证失败）"
            else:
                reason = "回答错了（通常在此 Benchmark 下，召回栈已提供全量信息，模型被问题结构/干扰词骗过或时间跨度推导算错）"
                
            f.write(f"### Q{i+1}: {q_type}\n")
            f.write(f"- **问题**: {q['question'].replace(chr(10), ' ')}\n")
            f.write(f"- **Agent 回答**: {agent_answer}\n")
            f.write(f"- **标准答案 (GT)**: {q['ground_truth']}\n")
            f.write(f"- **错误分类分析**: **{reason}**\n\n")

    print(f"Analysis saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze LoCoMo JSON Benchmark Results")
    parser.add_argument("json_file", help="Path to the locomo benchmark json output file")
    parser.add_argument("--out_dir", default="./locomo_analysis_output", help="Directory to save the analysis reports")
    args = parser.parse_args()
    analyze(args.json_file, args.out_dir)
