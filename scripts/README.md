# 脚本索引

从实验仓库根目录运行。`simulate.py` 和 `check_fixture.py` 是保留原路径的常用入口；子目录下的脚本用模块名运行，例如 `python -m scripts.cpu.replay.study_dynamic_cpu --help`。实验的完整参数和输入见[实验索引](../docs/experiment-index.md)及各研究报告。

| 位置 | 用途 |
| --- | --- |
| [`data/`](data/) | 下载／检查公开轨迹，导入硬件成本数据 |
| [`cpu/replay/`](cpu/replay/) | CPU 轨迹回放、基线对照与离线最优审计 |
| [`cpu/dynamic/`](cpu/dynamic/) | R01 动态状态干预、留出和入口回放复算 |
| [`cpu/analysis/`](cpu/analysis/) | 汇总 CPU 回放结果 |
| [`cpu/lookahead/screening/`](cpu/lookahead/screening/) | 前瞻触发、复杂度削减与共同后续抵消 |
| [`cpu/lookahead/robustness/`](cpu/lookahead/robustness/) | 不确定域、候选差值、预算和多步组合 |
| [`cpu/lookahead/staging/`](cpu/lookahead/staging/) | 准入、释放、共享、搬运顺序和临界场景 |
| [`gpu/experiments/`](gpu/experiments/) | GPU 实验的配对验证、机制复算和规则留出 |
| [`gpu/analysis/`](gpu/analysis/) | GPU 日志对齐、成本拟合、结果汇总和外推 |
| [`validation/`](validation/) | 已归档证据复算、归档校验与提取 |

新增研究或复现脚本按主要用途入目录，不放到 `scripts/` 根目录。机器专用的测量脚本仍留在 `.local/`，运行产物写到 `outputs/`。脚本移动时同步更新模块导入、报告链接和复现命令。
