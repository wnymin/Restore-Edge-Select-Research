# 实验环境与运行说明

使用 Conda。仓库根目录的 [environment.yml](../environment.yml) 固定 Python 3.12.13，使用 conda-forge；模拟器仅依赖 Python 标准库。Conda 会按操作系统解析底层运行库。无需 CUDA、框架、模型权重或真实显存。环境文件更新后，可执行：

```sh
conda env update -n restore-edge-select -f environment.yml --prune
```

也可在仓库内创建本地环境：`conda env create --prefix ./.conda-env -f environment.yml`。之后执行 `conda activate ./.conda-env`，或使用 `conda run --prefix ./.conda-env python scripts/check_fixture.py`。本地环境目录已加入 Git 忽略清单。

```sh
python scripts/check_fixture.py
python scripts/simulate.py --prefetch-mib 0 128 160 192 --policy recompute longest threshold local-cost capacity offline --threshold-tokens 576
```

`--trace` 和 `--model` 接收 JSON 文件；默认读取仓库内的手算轨迹和 Qwen 配置。轨迹显式给出到达、前缀、缓存对象、候选边界及依赖顺序。模拟器按模型规格检查依赖区间、前缀、检查点位置和字节数，不读取轨迹中的预期结果。省略 `--prefetch-mib` 时扫描轨迹给出的预算档位。输出含请求顺序、所选边界、预留时刻、完成时间、排队与队首恢复等待、分路径传输量、重算 token 数、预算峰值和逐对象传输时间线。设备峰值是模型中的记账值，包含已预留容量和执行空间估算。

策略在首次进入窗口时选择一次。`longest` 取最大合法边界；`threshold` 取不低于指定 token 数的最大合法边界，否则从头算。两者允许队首按需恢复。`local-cost` 比较单请求的传输与后缀重算估算。`capacity` 在已有请求执行、且候选对象可整批预留时，假设传输可被执行覆盖，按重算时间打分；否则使用单请求成本。这个假设是启发式，真实重叠由事件时间线计算。`offline` 按响应时间总和找最优解，保留并列结果；默认最多遍历 10,000 个组合，可用 `--max-combinations` 调整。

设备初始常驻对象计入固定预算。多个请求持有同一预取对象时按 ID 记一份；共享循环检查点在有后缀续算时生成私有副本，需提供 `cost.device_copy` 的 `startup_ms` 和 `bytes_per_ms`。副本在交接时完成，计入当前请求恢复时间；外部 RAM/SSD 通道仍可工作。对象的额外对齐字节只计入显存，不计入传输载荷。执行预算按完整输入的 KV、私有循环状态和工作区预检。
