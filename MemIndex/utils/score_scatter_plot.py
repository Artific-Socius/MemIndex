"""
ScoreScatterPlot - 子测试分数绘图工具

为每个配置文件生成图表，X轴为子测试名称，Y轴为得分比率。
图例按 eval_prompt 类型（default, strict, lenient）分类。

支持散点图和箱型图两种绘制模式。

使用方法:
    python -m utils.score_scatter_plot                          # 为每个配置生成散点图
    python -m utils.score_scatter_plot --plot-type box          # 绘制箱型图
    python -m utils.score_scatter_plot --report-dir ./data/reports
    python -m utils.score_scatter_plot --output-dir ./output
    python -m utils.score_scatter_plot --eval-mode binary       # 只处理 binary 模式配置
    python -m utils.score_scatter_plot --show                   # 绘制后显示图片
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath


# 支持的 eval_prompt 类型
EVAL_PROMPT_TYPES = ["default", "strict", "lenient"]

# 颜色配置
COLORS = {
    "default": "#4285F4",   # Google Blue
    "strict": "#EA4335",    # Google Red
    "lenient": "#34A853",   # Google Green
}

# 标记配置
MARKERS = {
    "default": "o",
    "strict": "s",
    "lenient": "^",
}

# X轴偏移量（避免重叠）
X_OFFSETS = {
    "default": -0.25,
    "strict": 0,
    "lenient": 0.25,
}


def simple_dbscan(data: list[float], eps: float = 0.05, min_samples: int = 2) -> np.ndarray:
    """
    简易 DBSCAN 聚类实现（使用 scipy）

    Args:
        data: 一维数据列表
        eps: 邻域半径
        min_samples: 最小样本数

    Returns:
        labels: 聚类标签，-1 表示噪声点
    """
    from scipy.spatial.distance import pdist, squareform

    n = len(data)
    if n < min_samples:
        return np.array([-1] * n)

    data_arr = np.array(data).reshape(-1, 1)
    if n == 1:
        return np.array([-1])

    dist_matrix = squareform(pdist(data_arr))
    labels = np.array([-1] * n)
    visited = [False] * n
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dist_matrix[i] <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            continue
        labels[i] = cluster_id
        seed_set = [nb for nb in neighbors if nb != i]
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = np.where(dist_matrix[q] <= eps)[0].tolist()
                if len(q_neighbors) >= min_samples:
                    seed_set.extend([nb for nb in q_neighbors if nb not in seed_set])
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1
        cluster_id += 1

    return labels


def compute_advanced_stats(points: list[tuple[float, float]], dbscan_eps: float = 0.05) -> dict:
    """
    计算统计指标（使用标准统计学方法）

    Args:
        points: [(x, y), ...] 点列表，x 是位置索引，y 是分数比率
        dbscan_eps: DBSCAN 邻域半径参数

    Returns:
        dict: {
            'n': 总点数,
            'mean': 总体均值,
            'icc': ICC（组内相关系数，衡量重复测量一致性）,
            'within_std': 位置内标准差（衡量重复性）,
            'between_std': 位置间标准差（衡量题目难度分布）,
            'avg_clusters': 平均聚类数（DBSCAN，衡量数据分散程度）,
        }
    """
    from collections import defaultdict

    if not points:
        return {'n': 0, 'mean': 0, 'icc': 0, 'within_std': 0, 'between_std': 0, 'avg_clusters': 0}

    # 按位置（X 坐标）分组
    position_groups: dict[int, list[float]] = defaultdict(list)
    for x, y in points:
        position_groups[round(x)].append(y)

    n_positions = len(position_groups)
    if n_positions == 0:
        return {'n': len(points), 'mean': np.mean([p[1] for p in points]),
                'icc': 0, 'within_std': 0, 'between_std': 0, 'avg_clusters': 0}

    # 计算每个位置的统计量
    position_means = []
    position_vars = []
    cluster_counts = []

    for pos, ratios in position_groups.items():
        pos_mean = np.mean(ratios)
        pos_var = np.var(ratios)
        position_means.append(pos_mean)
        position_vars.append(pos_var)

        # DBSCAN 聚类
        labels = simple_dbscan(ratios, eps=dbscan_eps, min_samples=2)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        # 如果没有形成聚类（全是噪声），至少算 1 个"区域"
        cluster_counts.append(max(1, n_clusters))

    # 计算指标
    overall_mean = np.mean([p[1] for p in points])
    within_var = np.mean(position_vars)  # 位置内平均方差 (σ_w²)
    between_var = np.var(position_means)  # 位置均值的方差 (σ_b²)

    within_std = np.sqrt(within_var)
    between_std = np.sqrt(between_var)

    # ICC (Intraclass Correlation Coefficient) - 组内相关系数
    # ICC = σ_b² / (σ_b² + σ_w²)
    # 衡量"差异主要来自位置间还是位置内"
    total_var = between_var + within_var
    icc = between_var / total_var if total_var > 0 else 0

    # 平均聚类数（DBSCAN）
    avg_clusters = np.mean(cluster_counts) if cluster_counts else 0

    return {
        'n': len(points),
        'mean': overall_mean,
        'icc': icc,
        'within_std': within_std,
        'between_std': between_std,
        'n_positions': n_positions,
        'avg_clusters': avg_clusters,
    }


def count_overlapping_points(x_values: list[float], y_values: list[float], tolerance: float = 0.02) -> dict[tuple, int]:
    """
    统计重叠点的数量

    Args:
        x_values: X坐标列表
        y_values: Y坐标列表
        tolerance: 判断重叠的容差（Y轴方向）

    Returns:
        {(x, y): count} 字典，只包含 count > 1 的点
    """
    from collections import Counter

    # 将坐标四舍五入到一定精度避免浮点数问题
    points = [(round(x, 2), round(y, 2)) for x, y in zip(x_values, y_values)]
    counts = Counter(points)

    # 只返回重叠的点（count > 1）
    return {k: v for k, v in counts.items() if v > 1}


def plot_scatter_with_overlap(
    ax,
    x_values: list[float],
    y_values: list[float],
    color: str,
    marker: str,
    label: str,
    base_size: int = 150,
):
    """
    绘制散点图，对重叠点使用深色并标注数量（split 模式使用）

    Args:
        ax: matplotlib axes
        x_values: X坐标
        y_values: Y坐标
        color: 基础颜色
        marker: 标记形状
        label: 图例标签
        base_size: 基础点大小
    """
    from collections import Counter

    # 统计每个位置的点数
    points = [(round(x, 2), round(y, 2)) for x, y in zip(x_values, y_values)]
    point_counts = Counter(points)

    # 分离单点和重叠点
    single_x, single_y = [], []
    overlap_points = {}  # {(x, y): count}

    seen = set()
    for (x, y), count in point_counts.items():
        if count == 1:
            single_x.append(x)
            single_y.append(y)
        else:
            overlap_points[(x, y)] = count

    # 绘制单点（正常颜色）
    if single_x:
        ax.scatter(
            single_x, single_y,
            label=label,
            color=color,
            marker=marker,
            alpha=0.7,
            s=base_size,
        )
    else:
        # 如果没有单点，也需要添加图例
        ax.scatter([], [], label=label, color=color, marker=marker, s=base_size)

    # 绘制重叠点（深色，接近黑色）
    if overlap_points:
        overlap_x = [p[0] for p in overlap_points.keys()]
        overlap_y = [p[1] for p in overlap_points.keys()]

        ax.scatter(
            overlap_x, overlap_y,
            color='#1a1a1a',  # 接近黑色
            marker=marker,
            alpha=0.9,
            s=base_size * 1.2,  # 稍大一点
            edgecolors=color,
            linewidths=2,
        )

        # 在点的右上角标注数量
        for (x, y), count in overlap_points.items():
            ax.annotate(
                str(count),
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                color='#1a1a1a',
            )


def get_wedge_marker(start_angle, end_angle, num_steps=20):
    """
    创建一个扇形的 Marker Path。

    参数:
    start_angle: 起始角度（度）
    end_angle: 结束角度（度）
    num_steps: 弧线的平滑程度
    """
    # 将角度转换为弧度
    theta = np.linspace(np.radians(start_angle), np.radians(end_angle), num_steps)

    # 顶点列表：从圆心(0,0)开始
    verts = [(0, 0)]

    # 添加圆弧上的点
    # x = cos(t), y = sin(t)
    for t in theta:
        verts.append((np.cos(t), np.sin(t)))

    # 闭合路径回到圆心
    verts.append((0, 0))

    # 定义路径指令
    codes = [mpath.Path.MOVETO] + \
            [mpath.Path.LINETO] * num_steps + \
            [mpath.Path.CLOSEPOLY]

    return mpath.Path(verts, codes)

def scatter_pie_markers(x, y, counts, colors_list, size=100, ax=None, rotation=90):
    """
    参数增加了 rotation：
    rotation: 整体旋转的角度（度）。
              默认为 90，意味着第一个扇形从 12 点钟方向开始逆时针绘制。
              如果不加 rotation，默认是从 3 点钟方向开始。
    """
    if ax is None:
        ax = plt.gca()

    # --- 1. 维度与格式安全检查 (之前的修复) ---
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    counts = np.atleast_1d(counts)

    if len(counts) == 1 and len(x) > 1:
        counts = np.full(len(x), counts[0])

    if len(x) == 1 and len(colors_list) == counts[0]:
        colors_list = [colors_list]
    # ---------------------------------------

    unique_counts = np.unique(counts)

    for n in unique_counts:
        indices = np.where(counts == n)[0]
        sub_x = x[indices]
        sub_y = y[indices]
        sub_colors = [colors_list[i] for i in indices]

        wedge_angle = 360.0 / n

        for i in range(n):
            # --- 2. 关键修改：加入 rotation 偏移 ---
            # 原始逻辑：0度从3点钟开始
            # 新逻辑：加上 rotation。例如 rotation=90，则 0 -> 90 (12点钟)
            start_ang = i * wedge_angle + rotation
            end_ang = (i + 1) * wedge_angle + rotation

            marker_shape = get_wedge_marker(start_ang, end_ang)

            layer_colors = [c[i] for c in sub_colors]

            ax.scatter(sub_x, sub_y,
                       s=size,
                       marker=marker_shape,
                       c=layer_colors,
                       edgecolors='none')

def plot_scatter_merged_with_concentric(
    ax,
    all_points: dict[str, list[tuple[float, float]]],
    base_size: int = 250,
    dbscan_eps: float = 0.05,

):
    """
    绘制 merge 模式的散点图，使用同心圆环表示多类型重叠

    Args:
        ax: matplotlib axes
        all_points: {eval_prompt: [(x, y), ...]} 所有点按类型分组
        base_size: 基础点大小
        dbscan_eps: DBSCAN 聚类邻域半径参数
    """
    from collections import defaultdict

    # 统计每个位置有哪些类型和数量
    # {(x, y): {eval_prompt: count}}
    position_data: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for eval_prompt, points in all_points.items():
        for x, y in points:
            pos = (round(x, 2), round(y, 2))
            position_data[pos][eval_prompt] += 1

    # 先添加图例（绘制空点）
    for eval_prompt in EVAL_PROMPT_TYPES:
        if eval_prompt in all_points and all_points[eval_prompt]:
            stats = compute_advanced_stats(all_points[eval_prompt], dbscan_eps=dbscan_eps)
            ax.scatter(
                [], [],
                label=f"{eval_prompt} (n={stats['n']}, μ={stats['mean']:.0%}, ICC={stats['icc']:.0%}, K={stats['avg_clusters']:.2f})",
                color=COLORS.get(eval_prompt, 'gray'),
                marker='o',
                s=base_size,
            )

    # 遍历每个位置
    for (x, y), type_counts in position_data.items():
        total_count = sum(type_counts.values())
        types_present = list(type_counts.keys())
        num_types = len(types_present)

        # 构建易读的分类计数字典
        # key: eval_prompt 类型 (default/strict/lenient)
        # value: 该类型在此位置 (x, y) 出现的次数
        category_counts: dict[str, int] = {
            eval_prompt: count
            for eval_prompt, count in type_counts.items()
        }
        # 示例: {"default": 3, "strict": 2} 表示该位置有 3 个 default 类型点和 2 个 strict 类型点

        scatter_pie_markers(x,y, len(type_counts), [COLORS.get(k, 'gray') for k in type_counts.keys()], size=base_size, ax=ax)

        # step_size = base_size / len(type_counts)
        # for i, (k, v) in enumerate(type_counts.items()):
        #     color = COLORS.get(k, 'gray')
        #     ax.scatter(
        #         [x], [y],
        #         color=color,
        #         marker='o',
        #         alpha=1.0,
        #         s=max(1, base_size - (step_size * i) * 1.5),
        #     )

        if num_types == 1:
            # 只有一种类型
            eval_prompt = types_present[0]
            count = category_counts[eval_prompt]
            color = COLORS.get(eval_prompt, 'gray')

            if count == 1:
                # 单点，正常颜色
                # ax.scatter(
                #     [x], [y],
                #     color=color,
                #     marker='o',
                #     alpha=0.8,
                #     s=base_size,
                # )
                pass
            else:
                # 同类型重叠，深色
                # ax.scatter(
                #     [x], [y],
                #     color='#1a1a1a',
                #     marker='o',
                #     alpha=1.0,
                #     s=base_size * 1.3,
                #     edgecolors=color,
                #     linewidths=3,
                # )
                # 标注数量
                ax.annotate(
                    str(count),
                    (x, y),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    color='#1a1a1a',
                )
        else:
            # 多种类型重叠，绘制同心圆环（空心圆，只有边框）
            # 按照 EVAL_PROMPT_TYPES 的顺序排序（外层到内层）
            sorted_types = [t for t in EVAL_PROMPT_TYPES if t in types_present]
            num_layers = len(sorted_types)

            # 圆环厚度配置
            min_ring_width = 15  # 最小厚度
            extra_width_budget = 5  # 额外厚度预算（按比例分配）

            # 计算每种类型的比重和对应厚度
            # 每层都有 min_ring_width，然后按比例分配额外厚度
            type_widths = {}
            for eval_prompt in sorted_types:
                count = type_counts[eval_prompt]
                ratio = count / total_count
                # 厚度 = 最小厚度 + 比例 * 额外厚度
                width = min_ring_width + ratio * extra_width_budget
                type_widths[eval_prompt] = width

            # 计算圆环尺寸（从外到内，每层根据厚度递减）
            outer_size = base_size * 1.0  # 最外层尺寸

            # 先绘制白色背景底圆
            # ax.scatter(
            #     [x], [y],
            #     facecolors='white',
            #     edgecolors='none',
            #     marker='o',
            #     s=outer_size * 1.15,
            #     zorder=9,
            # )

            # 从外到内绘制圆环
            current_size = outer_size
            for layer_idx, eval_prompt in enumerate(sorted_types):
                color = COLORS.get(eval_prompt, 'gray')
                ring_width = type_widths[eval_prompt]

                # 绘制圆环
                # ax.scatter(
                #     [x], [y],
                #     facecolors='none',
                #     edgecolors=color,
                #     marker='o',
                #     s=current_size,
                #     linewidths=ring_width,
                #     zorder=10 + layer_idx,
                # )

                # 下一层尺寸递减
                size_reduction = ring_width * 10
                current_size = max(current_size - size_reduction, base_size * 0.15)

            # 中心白点（缩小）
            # ax.scatter(
            #     [x], [y],
            #     facecolors='white',
            #     edgecolors='none',
            #     marker='o',
            #     s=base_size * 0.12,
            #     zorder=10 + num_layers,
            # )

            # 分类计数标注（一个框，用 "/" 分割，每个数字保持从属颜色）
            # 先绘制白色背景框
            # 构建完整文本计算宽度（加空格增加右边距）
            counts_text = '/'.join([str(type_counts[t]) for t in sorted_types]) + '  '
            ax.annotate(
                counts_text,
                (x, y),
                xytext=(12, 18),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color='white',  # 占位，不可见
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1),
                zorder=20,
            )

            # 逐个绘制带颜色的数字和分隔符
            x_text_offset = 12
            for type_idx, eval_prompt in enumerate(sorted_types):
                count = type_counts[eval_prompt]
                color = COLORS.get(eval_prompt, 'gray')

                # 绘制数字
                ax.annotate(
                    str(count),
                    (x, y),
                    xytext=(x_text_offset, 18),
                    textcoords='offset points',
                    fontsize=12,
                    fontweight='bold',
                    color=color,
                    zorder=21,
                )

                # 计算下一个位置（数字宽度约 8-10 像素）
                digit_width = len(str(count)) * 8
                x_text_offset += digit_width

                # 添加分隔符（如果不是最后一个）
                if type_idx < len(sorted_types) - 1:
                    ax.annotate(
                        '/',
                        (x, y),
                        xytext=(x_text_offset, 18),
                        textcoords='offset points',
                        fontsize=12,
                        fontweight='bold',
                        color='#1a1a1a',
                        zorder=21,
                    )
                    x_text_offset += 8  # 分隔符宽度


@dataclass
class ReportInfo:
    """单个报告信息"""
    filepath: str
    benchmark_name: str
    eval_mode: str
    eval_prompt: str
    chat_prompt: str
    timestamp: datetime
    score: float
    total_score: float
    items: list[dict] = field(default_factory=list)


@dataclass
class ConfigInfo:
    """配置文件信息"""
    config_path: str
    config_name: str
    folder: str
    eval_mode: str
    benchmarks: list[str]
    reports: list[ReportInfo] = field(default_factory=list)


def parse_config_name(filename: str) -> tuple[str, str] | None:
    """
    解析配置文件名，提取配置名和 eval_mode

    文件名格式: [report]-xxx-binary.json 或 [report]-xxx-score.json
    """
    if not filename.startswith("[report]-") or not filename.endswith(".json"):
        return None

    name = filename[len("[report]-"):-len(".json")]

    if name.endswith("-binary"):
        return name[:-len("-binary")], "binary"
    elif name.endswith("-score"):
        return name[:-len("-score")], "score"
    else:
        return name, "unknown"


def parse_report_filename(filename: str) -> datetime | None:
    """
    解析报告文件名，提取时间戳
    """
    match = re.search(r'\[(\d+)\]-(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\.\d+)\.json$', filename)
    if match:
        timestamp_str = match.group(2).replace('_', ':')
        try:
            return datetime.fromisoformat(timestamp_str)
        except ValueError:
            pass
    return None


def load_report(filepath: str, result_key: str = "result", detailed_x: bool = False) -> ReportInfo | None:
    """
    加载单个报告文件

    Args:
        filepath: 报告文件路径
        result_key: 结果字段名（默认为 "result"）
        detailed_x: 是否启用详细 X 轴模式（每个子项目单独一列）
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 从 split_sequence 中提取子测试分数
        items = []
        split_sequence = data.get('split_sequence', {})

        for test_name, test_items in split_sequence.items():
            if not test_items:
                continue

            valid_items = [x for x in test_items if x and x.get('score')]
            if not valid_items:
                continue

            if detailed_x:
                # 详细模式：每个子项目单独一列
                for idx, item in enumerate(valid_items):
                    item_score = item['score'].get('score', 0)
                    item_result = item['score'].get(result_key, item['score'].get('result', 0))

                    # 如果只有一个项目，不加索引
                    if len(valid_items) == 1:
                        item_name = test_name
                    else:
                        item_name = f"{test_name}[{idx}]"

                    items.append({
                        'name': item_name,
                        'score': item_score,
                        'result': item_result,
                        'ratio': item_result / item_score if item_score > 0 else 0,
                        'item_count': 1,
                    })
            else:
                # 默认模式：汇总每个子测试
                total_score = sum(x['score'].get('score', 0) for x in valid_items)
                # 使用指定的 result_key 获取结果
                total_result = sum(x['score'].get(result_key, x['score'].get('result', 0)) for x in valid_items)

                items.append({
                    'name': test_name,
                    'score': total_score,
                    'result': total_result,
                    'ratio': total_result / total_score if total_score > 0 else 0,
                    'item_count': len(valid_items),
                })

        filename = os.path.basename(filepath)
        timestamp = parse_report_filename(filename)
        if timestamp is None:
            timestamp = datetime.min

        return ReportInfo(
            filepath=filepath,
            benchmark_name=data.get('benchmark_name', ''),
            eval_mode=data.get('eval_mode', 'binary'),
            eval_prompt=data.get('eval_prompt', 'default'),
            chat_prompt=data.get('chat_prompt', 'default'),
            timestamp=timestamp,
            score=data.get('score', 0),
            total_score=data.get('total_score', 0),
            items=items,
        )
    except Exception as e:
        print(f"[警告] 加载报告失败 {filepath}: {e}", file=sys.stderr)
        return None


def load_config(config_path: str) -> ConfigInfo | None:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filename = os.path.basename(config_path)
        parsed = parse_config_name(filename)
        if parsed is None:
            return None

        config_name, eval_mode = parsed

        return ConfigInfo(
            config_path=config_path,
            config_name=config_name,
            folder=data.get('folder', ''),
            eval_mode=eval_mode,
            benchmarks=data.get('current_benchmarks', []),
        )
    except Exception as e:
        print(f"[警告] 加载配置失败 {config_path}: {e}", file=sys.stderr)
        return None


def scan_configs(report_dir: str) -> list[ConfigInfo]:
    """扫描所有配置文件"""
    configs = []
    report_path = Path(report_dir)

    if not report_path.exists():
        print(f"[错误] 报告目录不存在: {report_dir}", file=sys.stderr)
        return configs

    # 查找所有 [report]-*.json 配置文件
    # 注意: glob 中方括号会被当作字符类，所以使用 iterdir + 过滤
    for config_file in report_path.iterdir():
        if not config_file.is_file():
            continue
        if not config_file.name.startswith("[report]-"):
            continue
        if not config_file.name.endswith(".json"):
            continue
        if config_file.name.endswith(".lock"):
            continue

        config = load_config(str(config_file))
        if config:
            configs.append(config)

    return configs


def load_config_reports(config: ConfigInfo, result_key: str = "result", detailed_x: bool = False) -> None:
    """
    加载配置下的所有报告

    Args:
        config: 配置信息
        result_key: 结果字段名
        detailed_x: 是否启用详细 X 轴模式
    """
    folder = Path(config.folder)
    if not folder.exists():
        print(f"  [警告] 报告文件夹不存在: {folder}")
        return

    for benchmark_name in config.benchmarks:
        report_path = folder / benchmark_name
        if report_path.exists():
            report = load_report(str(report_path), result_key=result_key, detailed_x=detailed_x)
            if report:
                config.reports.append(report)


def filter_latest_by_eval_prompt(reports: list[ReportInfo]) -> dict[str, ReportInfo]:
    """
    按 eval_prompt 分组，保留每组最新的报告

    Returns:
        {eval_prompt: ReportInfo}
    """
    groups: dict[str, ReportInfo] = {}

    for report in reports:
        prompt = report.eval_prompt
        if prompt not in groups or report.timestamp > groups[prompt].timestamp:
            groups[prompt] = report

    return groups


def group_all_by_eval_prompt(reports: list[ReportInfo]) -> dict[str, list[ReportInfo]]:
    """
    按 eval_prompt 分组，保留每组所有的报告

    Returns:
        {eval_prompt: [ReportInfo, ...]}
    """
    groups: dict[str, list[ReportInfo]] = {}

    for report in reports:
        prompt = report.eval_prompt
        if prompt not in groups:
            groups[prompt] = []
        groups[prompt].append(report)

    # 按时间排序
    for prompt in groups:
        groups[prompt].sort(key=lambda x: x.timestamp)

    return groups


def plot_config_scatter(
    config: ConfigInfo,
    reports_by_prompt: dict[str, ReportInfo],
    output_file: str,
    figsize: tuple[int, int] = (32, 18),
    dpi: int = 300,
    title_suffix: str = "",
    layout: str = "merge",
) -> dict:
    """
    为单个配置绘制散点图（只使用最新报告）

    Args:
        layout: "split" 分离模式（X轴偏移）或 "merge" 合并模式（同心圆）
    """
    fig, ax = plt.subplots(figsize=figsize)

    stats_summary = {}
    all_y = []

    # 收集所有子测试名称
    all_test_names: set[str] = set()
    for report in reports_by_prompt.values():
        for item in report.items:
            all_test_names.add(item['name'])

    sorted_test_names = sorted(all_test_names)
    name_to_idx = {name: idx for idx, name in enumerate(sorted_test_names)}

    # 添加垂直分隔线区分不同子测试
    for i in range(len(sorted_test_names) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    if layout == "merge":
        # Merge 模式：收集所有点，使用同心圆处理重叠
        all_points: dict[str, list[tuple[float, float]]] = {}

        for eval_prompt in EVAL_PROMPT_TYPES:
            if eval_prompt not in reports_by_prompt:
                continue

            report = reports_by_prompt[eval_prompt]
            points = []

            for item in report.items:
                x = name_to_idx[item['name']]  # 不加偏移
                y = item['ratio']
                points.append((x, y))
                all_y.append(y)

            if points:
                all_points[eval_prompt] = points

                stats_summary[eval_prompt] = {
                    'count': len(points),
                    'mean': np.mean([p[1] for p in points]),
                    'std': np.std([p[1] for p in points]),
                    'total_score': report.score,
                    'max_score': report.total_score,
                }

        # 使用同心圆绘制
        plot_scatter_merged_with_concentric(ax, all_points)
    else:
        # Split 模式：使用 X 轴偏移
        for eval_prompt in EVAL_PROMPT_TYPES:
            if eval_prompt not in reports_by_prompt:
                continue

            report = reports_by_prompt[eval_prompt]
            x_values = []
            y_values = []

            # 获取该类型的 X 轴偏移量
            x_offset = X_OFFSETS.get(eval_prompt, 0)

            for item in report.items:
                x_values.append(name_to_idx[item['name']] + x_offset)
                y_values.append(item['ratio'])

            if not x_values:
                continue

            # 使用带重叠检测的散点绑制
            plot_scatter_with_overlap(
                ax=ax,
                x_values=x_values,
                y_values=y_values,
                color=COLORS.get(eval_prompt, 'gray'),
                marker=MARKERS.get(eval_prompt, 'o'),
                label=f"{eval_prompt} (n={len(x_values)}, avg={np.mean(y_values):.2%})",
            )

            all_y.extend(y_values)

            stats_summary[eval_prompt] = {
                'count': len(x_values),
                'mean': np.mean(y_values),
                'std': np.std(y_values),
                'total_score': report.score,
                'max_score': report.total_score,
            }

    # 设置 X 轴
    if sorted_test_names:
        ax.set_xticks(range(len(sorted_test_names)))
        ax.set_xticklabels(sorted_test_names, rotation=45, ha='right', fontsize=14)

    # 设置 X 轴范围（留出边距）
    ax.set_xlim(-0.5, len(sorted_test_names) - 0.5)

    # 设置坐标轴
    ax.set_xlabel('Sub-test Name', fontsize=18)
    ax.set_ylabel('Score Ratio (result / total)', fontsize=18)
    title = f"{config.config_name} ({config.eval_mode.upper()})"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)

    # Y轴范围
    ax.set_ylim(-0.05, 1.15)

    # 图例
    ax.legend(loc='upper right', fontsize=14)

    # 网格（只显示水平线）
    ax.grid(True, alpha=0.3, axis='y')

    # 统计信息
    if stats_summary:
        stats_text = "Statistics (Latest Run):\n"
        for prompt_type, stats in stats_summary.items():
            stats_text += f"  {prompt_type}: μ={stats['mean']:.2%}, σ={stats['std']:.2%}, total={stats['total_score']:.2f}/{stats['max_score']:.2f}\n"

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle="round", alpha=0.1, facecolor='white')
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {output_file}")

    plt.close(fig)

    return stats_summary


def plot_config_box(
    config: ConfigInfo,
    reports_by_prompt: dict[str, ReportInfo],
    output_file: str,
    figsize: tuple[int, int] = (32, 18),
    dpi: int = 300,
    title_suffix: str = "",
) -> dict:
    """
    为单个配置绘制箱型图（只使用最新报告）
    """
    fig, ax = plt.subplots(figsize=figsize)

    stats_summary = {}

    # 收集所有子测试名称
    all_test_names: set[str] = set()
    for report in reports_by_prompt.values():
        for item in report.items:
            all_test_names.add(item['name'])

    sorted_test_names = sorted(all_test_names)
    name_to_idx = {name: idx for idx, name in enumerate(sorted_test_names)}

    # 为每种 eval_prompt 类型准备数据
    box_data = []
    box_positions = []
    box_colors = []
    box_labels = []

    width = 0.25

    for i, eval_prompt in enumerate(EVAL_PROMPT_TYPES):
        if eval_prompt not in reports_by_prompt:
            continue

        report = reports_by_prompt[eval_prompt]

        # 收集所有分数
        ratios = [item['ratio'] for item in report.items]

        if not ratios:
            continue

        # 添加箱型图数据
        box_data.append(ratios)
        box_positions.append(i)
        box_colors.append(COLORS.get(eval_prompt, 'gray'))
        box_labels.append(f"{eval_prompt} (n={len(ratios)})")

        stats_summary[eval_prompt] = {
            'count': len(ratios),
            'mean': np.mean(ratios),
            'std': np.std(ratios),
            'median': np.median(ratios),
            'total_score': report.score,
            'max_score': report.total_score,
        }

    if box_data:
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8),
        )

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    # 设置 X 轴
    ax.set_xticks(range(len(box_labels)))
    ax.set_xticklabels(box_labels, fontsize=14)

    # 设置坐标轴
    ax.set_xlabel('Eval Prompt Type', fontsize=18)
    ax.set_ylabel('Score Ratio (result / total)', fontsize=18)
    title = f"{config.config_name} ({config.eval_mode.upper()})"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)

    # Y轴范围
    ax.set_ylim(-0.05, 1.15)

    # 网格
    ax.grid(True, alpha=0.3, axis='y')

    # 统计信息
    if stats_summary:
        stats_text = "Statistics (Latest Run):\n"
        for prompt_type, stats in stats_summary.items():
            stats_text += f"  {prompt_type}: μ={stats['mean']:.2%}, med={stats['median']:.2%}, σ={stats['std']:.2%}\n"

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle="round", alpha=0.1, facecolor='white')
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {output_file}")

    plt.close(fig)

    return stats_summary


def plot_config_box_all_runs(
    config: ConfigInfo,
    reports_by_prompt: dict[str, list[ReportInfo]],
    output_file: str,
    figsize: tuple[int, int] = (32, 18),
    dpi: int = 300,
    title_suffix: str = "",
) -> dict:
    """
    为单个配置绘制箱型图（使用所有报告）
    """
    fig, ax = plt.subplots(figsize=figsize)

    stats_summary = {}

    # 为每种 eval_prompt 类型准备数据
    box_data = []
    box_positions = []
    box_colors = []
    box_labels = []

    for i, eval_prompt in enumerate(EVAL_PROMPT_TYPES):
        if eval_prompt not in reports_by_prompt:
            continue

        reports = reports_by_prompt[eval_prompt]

        # 收集所有分数
        all_ratios = []
        for report in reports:
            all_ratios.extend([item['ratio'] for item in report.items])

        if not all_ratios:
            continue

        # 添加箱型图数据
        box_data.append(all_ratios)
        box_positions.append(i)
        box_colors.append(COLORS.get(eval_prompt, 'gray'))
        box_labels.append(f"{eval_prompt} (runs={len(reports)}, n={len(all_ratios)})")

        stats_summary[eval_prompt] = {
            'run_count': len(reports),
            'data_count': len(all_ratios),
            'mean': np.mean(all_ratios),
            'std': np.std(all_ratios),
            'median': np.median(all_ratios),
        }

    if box_data:
        bp = ax.boxplot(
            box_data,
            positions=box_positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8),
        )

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    # 设置 X 轴
    ax.set_xticks(range(len(box_labels)))
    ax.set_xticklabels(box_labels, fontsize=14)

    # 设置坐标轴
    ax.set_xlabel('Eval Prompt Type', fontsize=18)
    ax.set_ylabel('Score Ratio (result / total)', fontsize=18)
    title = f"{config.config_name} ({config.eval_mode.upper()}) - All Runs"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)

    # Y轴范围
    ax.set_ylim(-0.05, 1.15)

    # 网格
    ax.grid(True, alpha=0.3, axis='y')

    # 统计信息
    if stats_summary:
        stats_text = "Statistics (All Runs):\n"
        for prompt_type, stats in stats_summary.items():
            stats_text += f"  {prompt_type}: runs={stats['run_count']}, μ={stats['mean']:.2%}, med={stats['median']:.2%}, σ={stats['std']:.2%}\n"

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle="round", alpha=0.1, facecolor='white')
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {output_file}")

    plt.close(fig)

    return stats_summary


def plot_config_scatter_all_runs(
    config: ConfigInfo,
    reports_by_prompt: dict[str, list[ReportInfo]],
    output_file: str,
    figsize: tuple[int, int] = (32, 18),
    dpi: int = 300,
    title_suffix: str = "",
    layout: str = "merge",
    dbscan_eps: float = 0.05,
) -> dict:
    """
    为单个配置绘制散点图（使用所有报告的所有数据点）

    Args:
        layout: "split" 分离模式（X轴偏移）或 "merge" 合并模式（同心圆）
        dbscan_eps: DBSCAN 聚类邻域半径参数
    """
    fig, ax = plt.subplots(figsize=figsize)

    stats_summary = {}

    # 收集所有子测试名称
    all_test_names: set[str] = set()
    for reports in reports_by_prompt.values():
        for report in reports:
            for item in report.items:
                all_test_names.add(item['name'])

    sorted_test_names = sorted(all_test_names)
    name_to_idx = {name: idx for idx, name in enumerate(sorted_test_names)}

    # 添加垂直分隔线区分不同子测试
    for i in range(len(sorted_test_names) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    if layout == "merge":
        # Merge 模式：收集所有点，使用同心圆处理重叠
        all_points: dict[str, list[tuple[float, float]]] = {}

        for eval_prompt in EVAL_PROMPT_TYPES:
            if eval_prompt not in reports_by_prompt:
                continue

            reports = reports_by_prompt[eval_prompt]
            points = []

            for report in reports:
                for item in report.items:
                    x = name_to_idx[item['name']]  # 不加偏移
                    y = item['ratio']
                    points.append((x, y))

            if points:
                all_points[eval_prompt] = points

                # 使用新的统计指标
                adv_stats = compute_advanced_stats(points, dbscan_eps=dbscan_eps)
                stats_summary[eval_prompt] = {
                    'run_count': len(reports),
                    'point_count': adv_stats['n'],
                    'overall_mean': adv_stats['mean'],
                    'icc': adv_stats['icc'],
                    'within_std': adv_stats['within_std'],
                    'between_std': adv_stats['between_std'],
                    'avg_clusters': adv_stats['avg_clusters'],
                }

        # 使用同心圆绘制
        plot_scatter_merged_with_concentric(ax, all_points, dbscan_eps=dbscan_eps)
    else:
        # Split 模式：使用 X 轴偏移
        for eval_prompt in EVAL_PROMPT_TYPES:
            if eval_prompt not in reports_by_prompt:
                continue

            reports = reports_by_prompt[eval_prompt]

            # 获取该类型的 X 轴偏移量
            x_offset = X_OFFSETS.get(eval_prompt, 0)

            # 收集所有数据点
            x_values = []
            y_values = []

            for report in reports:
                for item in report.items:
                    x_values.append(name_to_idx[item['name']] + x_offset)
                    y_values.append(item['ratio'])

            if not x_values:
                continue

            color = COLORS.get(eval_prompt, 'gray')

            # 使用新的统计指标
            points = list(zip(x_values, y_values))
            adv_stats = compute_advanced_stats(points, dbscan_eps=dbscan_eps)

            # 使用带重叠检测的散点绑制
            plot_scatter_with_overlap(
                ax=ax,
                x_values=x_values,
                y_values=y_values,
                color=color,
                marker=MARKERS.get(eval_prompt, 'o'),
                label=f"{eval_prompt} (n={adv_stats['n']}, μ={adv_stats['mean']:.0%}, ICC={adv_stats['icc']:.0%}, K={adv_stats['avg_clusters']:.2f})",
            )

            stats_summary[eval_prompt] = {
                'run_count': len(reports),
                'point_count': adv_stats['n'],
                'overall_mean': adv_stats['mean'],
                'icc': adv_stats['icc'],
                'within_std': adv_stats['within_std'],
                'between_std': adv_stats['between_std'],
                'avg_clusters': adv_stats['avg_clusters'],
            }

    # 设置 X 轴
    if sorted_test_names:
        ax.set_xticks(range(len(sorted_test_names)))
        ax.set_xticklabels(sorted_test_names, rotation=45, ha='right', fontsize=14)

    # 设置 X 轴范围（留出边距）
    ax.set_xlim(-0.5, len(sorted_test_names) - 0.5)

    # 设置坐标轴
    ax.set_xlabel('Sub-test Name', fontsize=18)
    ax.set_ylabel('Score Ratio', fontsize=18)
    title = f"{config.config_name} ({config.eval_mode.upper()}) - All Runs"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(title, fontsize=22, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)

    # Y轴范围
    ax.set_ylim(-0.05, 1.15)

    # 图例
    ax.legend(loc='upper right', fontsize=14)

    # 网格（只显示水平线）
    ax.grid(True, alpha=0.3, axis='y')

    # 统计信息（使用标准统计学指标）
    if stats_summary:
        stats_text = "Statistics (All Runs):\n"
        stats_text += "  ICC=intraclass corr, K=avg DBSCAN clusters\n"
        for prompt_type, stats in stats_summary.items():
            stats_text += f"  {prompt_type}: n={stats['point_count']}, u={stats['overall_mean']:.0%}, ICC={stats['icc']:.0%}, K={stats['avg_clusters']:.2f}\n"

        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle="round", alpha=0.1, facecolor='white')
        )

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {output_file}")

    plt.close(fig)

    return stats_summary


def main():
    parser = argparse.ArgumentParser(
        description="子测试分数散点图绘制工具（按配置文件分别生成）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--report-dir", "-r",
        type=str,
        default="./data/reports",
        help="报告目录路径 (默认: ./data/reports)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录 (默认: 与报告目录相同)"
    )

    parser.add_argument(
        "--eval-mode", "-m",
        type=str,
        choices=["binary", "score", "all"],
        default="all",
        help="评估模式过滤 (默认: all，处理所有有标记的配置)"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="绘制完成后显示图片"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="图片分辨率 (默认: 300)"
    )

    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="使用所有测试数据（而非只用最新一次），显示均值和标准差"
    )

    parser.add_argument(
        "--plot-type", "-t",
        type=str,
        choices=["scatter", "box"],
        default="scatter",
        help="图表类型: scatter (散点图) 或 box (箱型图)，默认: scatter"
    )

    parser.add_argument(
        "--layout", "-l",
        type=str,
        choices=["split", "merge"],
        default="merge",
        help="散点布局: split (不同eval类型X轴偏移) 或 merge (同位置用同心圆)，默认: merge"
    )

    parser.add_argument(
        "--detailed-x",
        action="store_true",
        help="启用详细 X 轴模式：每个子测试的每个项目单独一列（如 location[0], location[1]）"
    )

    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.05,
        help="DBSCAN 聚类邻域半径参数（用于计算平均聚类数 K），默认: 0.05"
    )

    args = parser.parse_args()

    # 确定报告目录
    script_dir = Path(__file__).parent.parent
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = script_dir / args.report_dir

    if not report_dir.exists():
        print(f"[错误] 报告目录不存在: {report_dir}", file=sys.stderr)
        sys.exit(1)

    # 确定输出目录
    output_dir = Path(args.output_dir) if args.output_dir else report_dir
    if not output_dir.is_absolute():
        output_dir = script_dir / args.output_dir if args.output_dir else report_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 扫描配置文件
    print(f"扫描配置文件: {report_dir}")
    configs = scan_configs(str(report_dir))
    print(f"找到 {len(configs)} 个配置文件")

    if not configs:
        print("[错误] 未找到有效配置", file=sys.stderr)
        sys.exit(1)

    output_files = []

    # 有效的 eval_mode（排除 unknown）
    valid_eval_modes = ["binary", "score"]

    # 根据 plot_type 选择绘图函数
    plot_type_prefix = "scatter" if args.plot_type == "scatter" else "box"

    def do_plot(config: ConfigInfo, result_key: str, mode_suffix: str):
        """执行绘图的内部函数"""
        # 清空之前加载的报告
        config.reports = []

        # 加载报告（使用指定的 result_key 和 detailed_x）
        load_config_reports(config, result_key=result_key, detailed_x=args.detailed_x)
        print(f"  加载了 {len(config.reports)} 个报告 ({mode_suffix})" + (" [详细模式]" if args.detailed_x else ""))

        if not config.reports:
            print(f"  [跳过] 没有有效报告 ({mode_suffix})")
            return None

        if args.all_runs:
            # 使用所有数据
            reports_by_prompt = group_all_by_eval_prompt(config.reports)
            run_counts = {k: len(v) for k, v in reports_by_prompt.items()}
            print(f"  按 eval_prompt 分组 (所有数据): {run_counts}")

            if not reports_by_prompt:
                print(f"  [跳过] 没有有效分组 ({mode_suffix})")
                return None

            # 生成输出文件名
            detailed_suffix = "_detailed" if args.detailed_x else ""
            output_filename = f"{plot_type_prefix}_{config.config_name}_{mode_suffix}_all_runs{detailed_suffix}.png"
            output_path = output_dir / output_filename

            # 根据 plot_type 选择绘图函数
            if args.plot_type == "box":
                plot_config_box_all_runs(
                    config=config,
                    reports_by_prompt=reports_by_prompt,
                    output_file=str(output_path),
                    dpi=args.dpi,
                )
            else:
                plot_config_scatter_all_runs(
                    config=config,
                    reports_by_prompt=reports_by_prompt,
                    output_file=str(output_path),
                    dpi=args.dpi,
                    layout=args.layout,
                    dbscan_eps=args.dbscan_eps,
                )
        else:
            # 只使用最新数据
            reports_by_prompt = filter_latest_by_eval_prompt(config.reports)
            print(f"  按 eval_prompt 分组 (最新): {list(reports_by_prompt.keys())}")

            if not reports_by_prompt:
                print(f"  [跳过] 没有有效分组 ({mode_suffix})")
                return None

            # 生成输出文件名
            detailed_suffix = "_detailed" if args.detailed_x else ""
            output_filename = f"{plot_type_prefix}_{config.config_name}_{mode_suffix}{detailed_suffix}.png"
            output_path = output_dir / output_filename

            # 根据 plot_type 选择绘图函数
            if args.plot_type == "box":
                plot_config_box(
                    config=config,
                    reports_by_prompt=reports_by_prompt,
                    output_file=str(output_path),
                    dpi=args.dpi,
                )
            else:
                plot_config_scatter(
                    config=config,
                    reports_by_prompt=reports_by_prompt,
                    output_file=str(output_path),
                    dpi=args.dpi,
                    layout=args.layout,
                )

        return str(output_path)

    for config in configs:
        # 跳过 unknown 类型
        if config.eval_mode not in valid_eval_modes:
            continue

        # 过滤评估模式
        if args.eval_mode != "all":
            if config.eval_mode != args.eval_mode:
                continue

        print(f"\n处理配置: {config.config_name} ({config.eval_mode})")

        # 使用默认的 result 字段
        output_path = do_plot(config, "result", config.eval_mode)
        if output_path:
            output_files.append(output_path)

    print(f"\n完成！共生成 {len(output_files)} 张图")

    # 显示图片
    if args.show and output_files:
        import subprocess
        import platform

        for output_file in output_files:
            if platform.system() == 'Windows':
                os.startfile(output_file)
            elif platform.system() == 'Darwin':
                subprocess.run(['open', output_file])
            else:
                subprocess.run(['xdg-open', output_file])


if __name__ == "__main__":
    main()
