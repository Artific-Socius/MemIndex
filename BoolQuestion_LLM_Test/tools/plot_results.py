#!/usr/bin/env python3
"""
BoolQ评估结果绘图工具

用于绘制准确率与LogProb阈值的关系曲线

输出目录: outputs/images/

使用方法:
    python -m tools.plot_results                         # 输出到 outputs/images/
    python -m tools.plot_results --input outputs/        # 指定输入目录
    python -m tools.plot_results --show                  # 绘制后显示图片
    python -m tools.plot_results --output custom.png     # 指定输出文件路径
    python -m tools.plot_results --all                   # 不去重，显示所有实验
    python -m tools.plot_results --filter-outliers       # 使用IQR过滤异常值
"""
from __future__ import annotations

import argparse
import json
import os
import glob
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# 添加项目路径以导入i18n
sys.path.insert(0, str(Path(__file__).parent.parent))
from i18n import t, set_language, get_language

# 延迟导入绑定matplotlib后端（避免GUI弹窗）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ExperimentInfo:
    """实验信息"""
    model: str
    style: str
    reason_order: str
    timestamp: datetime
    uuid: str
    filepath: str
    
    @property
    def key(self) -> str:
        """实验唯一键（用于去重）"""
        return f"{self.model}_{self.style}_{self.reason_order}"
    
    @property
    def label(self) -> str:
        """图表标签"""
        return f"{self.model} ({self.style})"


def parse_filename(filepath: str) -> Optional[ExperimentInfo]:
    """
    解析文件名提取实验信息
    
    文件名格式: {model}_{style}_{reason_order}_{timestamp}_{uuid}.jsonl
    例如: gpt-4o-mini_direct_direct_20251201_010805_b46c3263.jsonl
    """
    base = os.path.basename(filepath)
    name = os.path.splitext(base)[0]  # 去掉扩展名
    
    # 匹配时间戳和UUID
    # 格式: YYYYMMDD_HHMMSS_XXXXXXXX
    match = re.search(r'_(\d{8}_\d{6})_([a-f0-9]{8})$', name)
    if not match:
        return None
    
    timestamp_str = match.group(1)
    uuid = match.group(2)
    
    # 提取前面的部分
    prefix = name[:match.start()]
    
    # 解析 model_style_reason_order
    # 从后往前解析，因为model名可能包含下划线
    parts = prefix.rsplit('_', 2)
    if len(parts) < 3:
        # 尝试兼容旧格式
        parts = prefix.rsplit('_', 1)
        if len(parts) >= 2:
            model = parts[0]
            style = parts[1]
            reason_order = "unknown"
        else:
            return None
    else:
        model = parts[0]
        style = parts[1]
        reason_order = parts[2]
    
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        timestamp = datetime.min
    
    return ExperimentInfo(
        model=model,
        style=style,
        reason_order=reason_order,
        timestamp=timestamp,
        uuid=uuid,
        filepath=filepath,
    )


def load_experiment_data(filepath: str) -> list[dict]:
    """
    加载实验数据
    
    仅保留成功的记录并提取必要字段
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if (record.get('status') == 'success' 
                        and 'avg_logprobs' in record 
                        and 'is_correct' in record
                        and record['avg_logprobs'] is not None):
                        data.append(record)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(t("[警告] 加载文件失败 {filepath}: {error}", filepath=filepath, error=e), file=sys.stderr)
    return data


def filter_outliers_iqr(
    data: list[dict], 
    field: str = 'avg_logprobs',
    k: float = 1.5
) -> tuple[list[dict], int]:
    """
    使用IQR方法过滤异常值
    
    异常值定义: x < Q1 - k*IQR 或 x > Q3 + k*IQR
    
    Args:
        data: 数据列表
        field: 用于检测异常值的字段名
        k: IQR倍数 (默认1.5，标准统计学阈值)
        
    Returns:
        (filtered_data, outlier_count): 过滤后的数据和被剔除的数量
    """
    if len(data) < 4:
        return data, 0
    
    values = np.array([d[field] for d in data])
    
    # 计算四分位数
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    # 计算边界
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # 过滤
    filtered_data = [
        d for d in data 
        if lower_bound <= d[field] <= upper_bound
    ]
    
    outlier_count = len(data) - len(filtered_data)
    
    return filtered_data, outlier_count


def compute_accuracy_curve(
    data: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算准确率曲线
    
    Args:
        data: 实验数据列表
        
    Returns:
        (thresholds, accuracies, ratios)
        - thresholds: LogProb阈值数组
        - accuracies: 对应阈值下的准确率
        - ratios: 对应阈值下的保留样本比例
    """
    if not data:
        return np.array([]), np.array([]), np.array([])
    
    scores = np.array([d['avg_logprobs'] for d in data])
    corrects = np.array([1.0 if d['is_correct'] else 0.0 for d in data])
    total_samples = len(scores)
    
    # 获取唯一阈值
    thresholds = np.unique(scores)
    thresholds.sort()
    
    final_thresholds = []
    final_accuracies = []
    final_ratios = []
    
    for t in thresholds:
        mask = scores >= t
        subset_correct = corrects[mask]
        count_kept = len(subset_correct)
        
        if count_kept == 0:
            continue
        
        acc = np.mean(subset_correct)
        ratio = count_kept / total_samples
        
        final_thresholds.append(t)
        final_accuracies.append(acc)
        final_ratios.append(ratio)
    
    return np.array(final_thresholds), np.array(final_accuracies), np.array(final_ratios)


def filter_latest_experiments(experiments: list[ExperimentInfo]) -> list[ExperimentInfo]:
    """
    按实验配置去重，保留最新的实验
    """
    groups: dict[str, ExperimentInfo] = {}
    
    for exp in experiments:
        key = exp.key
        if key not in groups or exp.timestamp > groups[key].timestamp:
            groups[key] = exp
    
    return list(groups.values())


def plot_accuracy_curves(
    experiments: list[ExperimentInfo],
    output_file: str,
    title: str = "BoolQ Accuracy & Retention vs. Confidence Threshold",
    figsize: tuple[int, int] = (20, 10),
    filter_outliers: bool = False,
    iqr_k: float = 1.5,
) -> dict:
    """
    绘制准确率曲线
    
    Args:
        experiments: 实验信息列表
        output_file: 输出文件路径
        title: 图表标题
        figsize: 图表大小
        filter_outliers: 是否使用IQR方法过滤异常值
        iqr_k: IQR倍数 (默认1.5)
        
    Returns:
        统计摘要字典
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    # 颜色和标记
    cmap_name = 'tab20' if len(experiments) > 10 else 'tab10'
    cmap = plt.get_cmap(cmap_name)
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '<', '>']
    
    stats_summary = []
    results = {}
    total_outliers = 0
    all_accuracies = []  # 收集所有准确率用于计算Y轴范围
    
    for i, exp in enumerate(experiments):
        print(t("处理: {filepath}", filepath=exp.filepath))
        data = load_experiment_data(exp.filepath)
        
        # 过滤异常值
        if filter_outliers and data:
            original_count = len(data)
            data, outlier_count = filter_outliers_iqr(data, 'avg_logprobs', iqr_k)
            total_outliers += outlier_count
            if outlier_count > 0:
                print(t("  过滤异常值: {count}/{total} ({pct:.1f}%)", 
                       count=outlier_count, total=original_count, 
                       pct=100*outlier_count/original_count))
        
        thresholds, accuracies, ratios = compute_accuracy_curve(data)
        
        if thresholds.size == 0:
            print(t("  [跳过] 无有效数据"))
            continue
        
        # 计算最大准确率及对应保留比例
        max_acc = np.max(accuracies)
        max_acc_indices = np.where(accuracies == max_acc)[0]
        best_idx = max_acc_indices[np.argmax(ratios[max_acc_indices])]
        best_ratio = ratios[best_idx]
        best_threshold = thresholds[best_idx]
        
        # 基础准确率（无过滤）
        base_acc = accuracies[0] if len(accuracies) > 0 else 0
        
        stats_summary.append(
            f"{exp.label}:\n"
            f"    Base: {base_acc:.2%} | Max: {max_acc:.2%} @ {best_ratio:.1%}"
        )
        
        results[exp.key] = {
            "model": exp.model,
            "style": exp.style,
            "base_accuracy": float(base_acc),
            "max_accuracy": float(max_acc),
            "best_ratio": float(best_ratio),
            "best_threshold": float(best_threshold),
            "sample_count": len(data),
        }
        
        # 绘图
        color = cmap(i % cmap.N)
        marker = markers[i % len(markers)]
        x_values = -thresholds
        mask = x_values > 0
        
        ax1.plot(x_values[mask], accuracies[mask], 
                 label=f"{exp.label} (Acc)", 
                 marker=marker, markersize=5, linewidth=2, 
                 linestyle='-', color=color)
        
        ax2.plot(x_values[mask], ratios[mask], 
                 label=f"{exp.label} (Ratio)", 
                 marker='', linewidth=1.5, linestyle='--', 
                 alpha=0.6, color=color)
        
        # 收集准确率用于计算Y轴范围
        all_accuracies.extend(accuracies[mask].tolist())
    
    # 设置坐标轴
    ax1.set_xlabel('Negative Avg LogProb (-T) [Log Scale]', fontsize=12)
    ax1.set_ylabel('Accuracy', color='black', fontsize=12)
    ax2.set_ylabel('Retention Ratio', color='gray', fontsize=12)
    
    ax1.set_xscale('log')
    ax1.invert_xaxis()
    
    # 动态计算Y轴下限: 1.0 - (1.0 - min) * 1.1
    if all_accuracies:
        min_acc = min(all_accuracies)
        y_lower = 1.0 - (1.0 - min_acc) * 1.1
        y_lower = max(0, y_lower)  # 确保下限不小于0
    else:
        y_lower = 0
    
    ax1.set_ylim(y_lower, 1.05)
    ax2.set_ylim(0, 1.05)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
               bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    
    # 统计摘要
    if stats_summary:
        stats_text = "Statistics:\n" + "\n".join(stats_summary)
        ax1.text(1.02, 0.0, stats_text, transform=ax1.transAxes,
                 fontsize=8, verticalalignment='bottom', 
                 fontfamily='monospace',
                 bbox=dict(boxstyle="round", alpha=0.1, facecolor='white'))
    
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(t("\n✓ 图表已保存: {output_file}", output_file=output_file))
    
    plt.close(fig)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description=t("BoolQ评估结果绘图工具"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.plot_results
  python -m tools.plot_results --input outputs/
  python -m tools.plot_results --show --output my_plot.png
  python -m tools.plot_results --all --title "My Experiment"
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="outputs",
        help=t("输入目录路径 (默认: outputs)")
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=t("输出图片文件路径 (默认: outputs/images/accuracy_vs_logprob.png)")
    )
    
    parser.add_argument(
        "--show",
        action="store_true",
        help=t("绘制完成后显示图片")
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help=t("显示所有实验，不去重")
    )
    
    parser.add_argument(
        "--title",
        type=str,
        default="BoolQ Accuracy & Retention vs. Confidence Threshold",
        help=t("图表标题")
    )
    
    parser.add_argument(
        "--json",
        type=str,
        help=t("输出统计结果到JSON文件")
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        choices=["zh", "en"],
        help=t("界面语言 (zh/en)")
    )
    
    parser.add_argument(
        "--filter-outliers",
        action="store_true",
        help=t("使用IQR方法过滤异常值 (Q1-1.5*IQR, Q3+1.5*IQR)")
    )
    
    parser.add_argument(
        "--iqr-k",
        type=float,
        default=1.5,
        help=t("IQR倍数 (默认: 1.5)")
    )
    
    args = parser.parse_args()
    
    # 设置语言
    if args.lang:
        set_language(args.lang)
    
    # 确定输入目录（相对于脚本所在的BoolQuestion_LLM_Test目录）
    script_dir = Path(__file__).parent.parent
    input_dir = script_dir / args.input
    
    if not input_dir.exists():
        # 也尝试从当前目录查找
        input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(t("[错误] 输入目录不存在: {path}", path=input_dir), file=sys.stderr)
        sys.exit(1)
    
    # 查找所有jsonl文件
    pattern = str(input_dir / "*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        print(t("[错误] 未找到jsonl文件: {pattern}", pattern=pattern), file=sys.stderr)
        sys.exit(1)
    
    print(t("找到 {count} 个实验结果文件", count=len(files)))
    
    # 解析实验信息
    experiments = []
    for f in files:
        info = parse_filename(f)
        if info:
            experiments.append(info)
        else:
            print(t("[警告] 无法解析文件名: {filepath}", filepath=f), file=sys.stderr)
    
    if not experiments:
        print(t("[错误] 无有效实验数据"), file=sys.stderr)
        sys.exit(1)
    
    # 去重（可选）
    if not args.all:
        original_count = len(experiments)
        experiments = filter_latest_experiments(experiments)
        if len(experiments) < original_count:
            print(t("去重后保留 {kept}/{total} 个实验（使用 --all 显示全部）", 
                    kept=len(experiments), total=original_count))
    
    # 排序
    experiments.sort(key=lambda x: (x.model, x.style, x.timestamp))
    
    # 确定输出路径
    if args.output:
        # 用户指定了输出路径
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / args.output
    else:
        # 默认输出到 outputs/images/
        images_dir = script_dir / "outputs" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        output_path = images_dir / "accuracy_vs_logprob.png"
    
    # 打印过滤信息
    if args.filter_outliers:
        print(t("启用IQR异常值过滤 (k={k})", k=args.iqr_k))
    
    # 绘图
    results = plot_accuracy_curves(
        experiments=experiments,
        output_file=str(output_path),
        title=args.title,
        filter_outliers=args.filter_outliers,
        iqr_k=args.iqr_k,
    )
    
    # 输出JSON统计（可选）
    if args.json:
        json_path = Path(args.json)
        if not json_path.is_absolute():
            # 默认保存到图片同目录
            json_path = output_path.parent / args.json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(t("✓ 统计结果已保存: {path}", path=json_path))
    
    # 显示图片（可选）
    if args.show:
        # 需要重新导入使用交互式后端
        import subprocess
        import platform
        
        if platform.system() == 'Windows':
            os.startfile(str(output_path))
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(output_path)])
        else:  # Linux
            subprocess.run(['xdg-open', str(output_path)])


if __name__ == "__main__":
    main()

