# 原始数据目录（`raw/`）

本目录用于存放 **各 benchmark 对应的数据集原文**：Hugging Face 子模块、官方发布的语料与标注文件等。数据体积通常较大，**默认不纳入本仓库版本控制**（见仓库根目录 `.gitignore`），仅在本地或 CI 中通过克隆/下载获取。

## 获取数据

- **Git 子模块**：若项目在 `.gitmodules` 中声明了数据集仓库，可在仓库根目录执行：
  - `git submodule update --init --recursive`
  - 或使用 `UniversalBenchmark/benchmark/data/TEMP_init_single_benchmark.py` 等脚本按说明初始化单个 benchmark 数据。
- **LFS**：部分文件由 Git LFS 管理，拉取后若仍为指针文件，需执行 `git lfs pull`。

## 当前布局示例

| 路径 | 说明 |
|------|------|
| `EverMind-AI/EverMemBench-Static/` | EverMemBench-Static（QAR jsonl、`data/` 各规模语料等），多为子模块 |

具体字段与目录结构以各数据集上游 README 为准。

## 开发约定

- **代码与元数据**：在 `benchmark/data/providers/`、`benchmark/interfaces/` 等路径维护加载与标准化逻辑。
- **本目录**：只放「原文」；除本 `README.md` 外，路径下的内容应由 `.gitignore` 忽略，避免误提交大文件。
