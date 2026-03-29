"""演示在自定义 Benchmark / 脚本中配合全局 Rich 进度条。

``run_benchmark_lite.py`` 已在 stderr TTY 下自动 ``progress_context``。
编程调用时，请自行在最外层：

    from agent.progress import progress_context
    from benchmark_lite import Runner, get_progress

    with progress_context():
        runner = Runner()
        # 在 get_scenarios / next_turn 等处可用 get_progress() 增加子任务
        runner.run(agent, MyBenchmark())

在 ``InteractiveScenario.next_turn`` 内示例::

    from benchmark_lite import get_progress

    def next_turn(self, history):
        pg = get_progress()
        h = pg.add_task(\"自定义步骤\", total=None)
        try:
            # ... 耗时准备 ...
            pg.update(h, description=\"准备完成\")
            return Turn(\"...\")
        finally:
            pg.remove_task(h)
"""

from __future__ import annotations

__all__: list[str] = []
