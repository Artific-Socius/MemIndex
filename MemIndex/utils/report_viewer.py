"""
ReportViewer - 基准测试报告查看器

使用 Gradio 构建的报告可视化界面。
"""

from __future__ import annotations

import loguru
import argparse
import datetime
import json
import os
import sys

import gradio as gr
import pandas as pd

# 获取模块路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

from .data_loader import BenchmarkDataset, BenchmarkItemExtra, format_text
from core.report import ReportStructure, ReportMainFile
loguru.logger.info("Import finished")

loguru.logger.info("ReportViewer - 基准测试报告查看器")

reports_folder = os.path.join(_parent_dir, "data/reports")

def update_configs(config_switch):
    """更新配置列表"""
    files = [
        os.path.join("data/config", x) 
        for x in os.listdir("data/config") 
        if os.path.isfile(os.path.join("data/config", x))
    ]
    files = list(map(lambda x: x.replace(os.path.sep, "/"), files))
    return gr.update(config_switch, choices=files)


def update_report(report_switch):
    """更新报告列表"""
    files = list(
        map(
            lambda x: x.replace(os.path.sep, "/"),
            [
                os.path.join(reports_folder, x) 
                for x in os.listdir(reports_folder) 
                if os.path.isfile(os.path.join(reports_folder, x)) 
                and x.endswith(".json") 
                and x.startswith("[report]")
            ],
        )
    )
    files = list(map(lambda x: x.replace(os.path.sep, "/"), files))
    return gr.update(report_switch, choices=files)


def transfer_highlight(text: str, highlight_position: list[tuple[int, int, str]]):
    """转换高亮位置"""
    result = []
    last_index = 0
    for start, end, label in highlight_position:
        result.append((text[last_index:start], None))
        result.append((text[start:end], label))
        last_index = end
    result.append((text[last_index:], None))
    return result


def draw_item_report(
    data: list[BenchmarkItemExtra],
    report_type: bool = True,
    extra_index: list[int] = None,
    tokens: list[int] = None,
):
    """绘制项目报告"""
    color_map = {
        "Question Reference": "green",
        "Answer Reference": "cyan",
        "Time Delta Reference": "blue",
    }
    
    index_set: list[BenchmarkItemExtra | None] = [None] * (
        max([x.index for x in data if x]) + 1
    )
    for item in data:
        if item:
            index_set[item.index] = item
    
    with gr.Column():
        for item in index_set:
            if item:
                if report_type:
                    title = f"#### Index: {item.index}"
                    if extra_index and item.index - 1 < len(extra_index):
                        title += f"(Global: {extra_index[item.index - 1] + 1})"
                    if tokens and item.index - 1 < len(tokens):
                        title += f"[Tokens: {tokens[item.index - 1] + 1}]"
                    if not item.response:
                        title += '<sup style="background-color:red; color:white; padding:2px 5px; margin: 2px; border-radius:6px;">Skip</sup>'
                    gr.Markdown(title)
                else:
                    gr.Markdown(f"#### Index: {item.index}")
                
                with gr.Row():
                    content, highlight_position = format_text(
                        item.ask.replace("\\", ""),
                        index_set,
                        return_highlight_position=True,
                    )
                    gr.HighlightedText(
                        value=transfer_highlight(content, highlight_position),
                        label="Ask",
                        combine_adjacent=len(highlight_position) > 0,
                        show_legend=len(highlight_position) > 0,
                        color_map={
                            k: v 
                            for k, v in color_map.items() 
                            if k in [x[2] for x in highlight_position]
                        },
                    )
                    if item.response:
                        gr.HighlightedText(
                            value=[(item.response, None)],
                            label="Agent Response",
                        )
                
                if item.score:
                    with gr.Row():
                        content, highlight_position = format_text(
                            item.score.answer.replace("\\", ""),
                            index_set,
                            return_highlight_position=True,
                        )
                        label = "得分标准" if not item.score.is_multiple else "得分标准[细粒度化判定]"
                        gr.HighlightedText(
                            value=transfer_highlight(content, highlight_position),
                            label=label,
                            combine_adjacent=len(highlight_position) > 0,
                            show_legend=len(highlight_position) > 0,
                            color_map={
                                k: v 
                                for k, v in color_map.items() 
                                if k in [x[2] for x in highlight_position]
                            },
                        )
                        
                        if item.score.is_lazy:
                            gr.Textbox(
                                value=str(item.score.lazy_count),
                                label="目标轮次跳跃数",
                            )
                            gr.HighlightedText(
                                value=[(item.score.lazy_eval_response_snapshot, None)],
                                label="Lazy目标回复",
                            )
                        
                        if report_type:
                            gr.Textbox(
                                value=f"{item.score.result}/{item.score.score}",
                                label="分数",
                            )
                        else:
                            gr.Textbox(value=f"{item.score.score}", label="分数")
                        
                        if report_type:
                            content, highlight_position = format_text(
                                item.score.reason.replace("\\", ""),
                                index_set,
                                return_highlight_position=True,
                            )
                            gr.HighlightedText(
                                transfer_highlight(content, highlight_position),
                                label="评分原因",
                                combine_adjacent=len(highlight_position) > 0,
                                show_legend=len(highlight_position) > 0,
                                color_map={
                                    k: v 
                                    for k, v in color_map.items() 
                                    if k in [x[1] for x in highlight_position]
                                },
                            )
                    
                    # 新增: 显示加权二元评分项的详细结果
                    if report_type and item.score.binary_items and len(item.score.binary_items) > 0:
                        with gr.Accordion("📊 加权二元评分详情", open=False):
                            # 计算统计信息
                            total_items = len(item.score.binary_items)
                            passed_items = sum(1 for bi in item.score.binary_items if bi.result)
                            total_weight = sum(bi.weight for bi in item.score.binary_items)
                            passed_weight = sum(bi.weight for bi in item.score.binary_items if bi.result)
                            
                            gr.Markdown(
                                f"**统计**: {passed_items}/{total_items} 项通过 | "
                                f"**权重得分**: {passed_weight:.2f}/{total_weight:.2f}"
                            )
                            
                            # 显示每个评分项
                            for bi in item.score.binary_items:
                                status_icon = "✅" if bi.result else "❌"
                                status_color = "green" if bi.result else "red"
                                
                                with gr.Row():
                                    gr.Markdown(
                                        f"<span style='color:{status_color};font-size:1.2em;'>{status_icon}</span> "
                                        f"**{bi.key}** (权重: {bi.weight})"
                                    )
                                
                                with gr.Row():
                                    gr.Textbox(
                                        value=bi.answer,
                                        label="评判标准",
                                        lines=1,
                                    )
                                    gr.Textbox(
                                        value=bi.reason if bi.reason else "(无理由)",
                                        label="评分理由",
                                        lines=1,
                                    )
                
                with gr.Row():
                    if item.depend:
                        with gr.Column():
                            gr.Markdown("##### Depend")
                            with gr.Row():
                                for dep in item.depend:
                                    gr.Textbox(value=dep, label="依赖项")
                    
                    if item.refs:
                        with gr.Column():
                            gr.Markdown("##### References")
                            with gr.Row():
                                for ref in item.refs:
                                    gr.Textbox(
                                        value=f"{ref.target}({ref.type})",
                                        label=f"{ref.type} Reference",
                                    )
                
                if item.post_process:
                    gr.Markdown("##### Post Process")
                    gr.Textbox(
                        value=item.post_process.replace("\\", ""),
                        label="后处理",
                    )


def on_report_config_change(report_config, report_switch):
    """报告配置变更处理"""
    with open(report_config, "r") as f:
        report_config_data = ReportMainFile(**json.load(f))
    
    return gr.update(
        report_switch,
        choices=[
            os.path.join(
                report_config_data.folder.replace("\\", "/"),
                x.replace("\\", "/"),
            )
            for x in report_config_data.current_benchmarks
        ],
    )


def export_table(df: pd.DataFrame):
    """导出表格"""
    file_path = "data.csv"
    df.to_csv(file_path, index=False)
    return file_path


def main(args):
    """主函数"""
    loguru.logger.info("Initializing report viewer")
    with gr.Blocks() as base:
        with gr.Tab("报告"):
            gr.Markdown("# 切换报告")
            
            files = list(
                map(
                    lambda x: x.replace(os.path.sep, "/"),
                    [
                        os.path.join(reports_folder, x)
                        for x in os.listdir(reports_folder)
                        if os.path.isfile(os.path.join(reports_folder, x))
                        and x.endswith(".json")
                        and x.startswith("[report]")
                    ],
                )
            )
            
            report_config_switch = gr.Dropdown(
                label="测试配置报告",
                choices=files,
                value=files[0] if len(files) > 1 else None,
            )
            
            report_switch = gr.Dropdown(label="报告", choices=[])
            refresh_report_option = gr.Button("刷新报告文件")
            refresh_report_option.click(
                fn=update_report,
                inputs=[report_config_switch],
                outputs=[report_config_switch],
            )
            report_config_switch.change(
                on_report_config_change,
                inputs=[report_config_switch, report_switch],
                outputs=[report_switch],
            )
            
            @gr.render(inputs=[report_switch, report_config_switch])
            def on_report_change(report_path, report_config):
                if not report_path or not report_config:
                    return
                
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                
                report = ReportStructure(**report_data)
                if report is None:
                    return
                
                with gr.Column():
                    gr.Markdown(f"# 报告 | 执行配置: {report.config_path}")
                    gr.Markdown(f"## 一共{len(report.total_sequence)}条消息")
                    gr.Markdown(f"## 含有{len(report.split_sequence)}个子测试")
                    
                    score_pct = (
                        (report.score / report.total_score * 100) 
                        if report.total_score > 0 
                        else 0
                    )
                    gr.Markdown(
                        f"## 测试得分:{report.score :.2f}/{report.total_score :.2f}({score_pct :.2f}%)"
                    )
                    
                    gr.Markdown(
                        f"## 开始/结束时间: "
                        f"{datetime.datetime.fromtimestamp(report.time_start).isoformat()}/"
                        f"{datetime.datetime.fromtimestamp(report.time_end).isoformat()}"
                    )
                    gr.Markdown(f"## 耗时: {report.time_usage / 60 :.4f}min")
                    gr.Markdown(f"## Agent: {report.agent}")
                    gr.Markdown(f"## Benchmark Name: {report.benchmark_name}")
                    gr.Markdown(f"## Tokens: {report.full_tokens}")
                    
                    # 显示评估配置
                    eval_mode = getattr(report, 'eval_mode', 'binary')
                    eval_mode_display = {
                        'binary': '🎯 二元评估 (Binary)',
                        'score': '📊 分数评估 (Score 0-1)',
                    }.get(eval_mode, eval_mode)
                    gr.Markdown(f"## 评估方式: {eval_mode_display}")
                    
                    # 显示 Prompt 配置
                    chat_prompt = getattr(report, 'chat_prompt', 'default') or 'default'
                    eval_prompt = getattr(report, 'eval_prompt', 'default') or 'default'
                    gr.Markdown(f"## Prompt 配置:")
                    gr.Markdown(f"- **Chat Prompt**: `{chat_prompt}`")
                    gr.Markdown(f"- **Eval Prompt**: `{eval_prompt}`")
                    
                    gr.Markdown(f"---")
                    gr.Markdown(f"## Agent额外信息:")
                    
                    for key, data in report.extra_metadata.items():
                        data_type = data.type
                        description = data.description
                        value = data.value
                        gr.Textbox(value=description, label=key)
                        
                        if data_type == "text":
                            gr.Markdown(value.replace("\n", "\n\n\t"))
                        elif data_type in ["number", "boolean", "array"]:
                            gr.Textbox(value=str(value))
                        elif data_type == "table":
                            df = pd.DataFrame(value["data"], columns=value["columns"])
                            download_btn = gr.Button("导出并下载 CSV")
                            file_output = gr.File()
                            download_btn.click(lambda: export_table(df), outputs=file_output)
                            gr.Dataframe(value=df)
                        else:
                            gr.Textbox(value=str(value))
                    
                    with gr.Tab("Total"):
                        draw_item_report(report.total_sequence)
                    
                    for name, sequence in report.split_sequence.items():
                        score = 0
                        score_max = 0
                        for item in sequence:
                            if item and item.score:
                                score += item.score.result
                                score_max += item.score.score
                        
                        with gr.Tab(name):
                            gr.Markdown(f"## 一共{len(sequence)}条消息")
                            
                            sub_score_pct = (score / score_max * 100) if score_max > 0 else 0
                            gr.Markdown(
                                f"## 子测试得分:{score :.2f}/{score_max :.2f}({sub_score_pct :.2f}%)"
                            )
                            
                            draw_item_report(
                                sequence,
                                extra_index=(
                                    report.split_global_index[name] 
                                    if name in report.split_global_index 
                                    else None
                                ),
                                tokens=(
                                    report.split_global_tokens[name] 
                                    if name in report.split_global_tokens 
                                    else None
                                ),
                            )
    loguru.logger.info("UI Built")
    loguru.logger.info("Starting server")
    base.launch(server_name=args.host, server_port=args.port, share=args.share)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Report Viewer")
    parser.add_argument("--host", "-p", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", "-P", type=int, default=7860, help="Port number")
    parser.add_argument("--share", "-S", action="store_true", help="Share the Gradio app publicly")
    return parser.parse_args()


def main_cli():
    """CLI 入口函数"""
    main(parse_args())


if __name__ == "__main__":
    main_cli()



