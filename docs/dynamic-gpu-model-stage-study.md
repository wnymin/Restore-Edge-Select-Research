# 真实模型状态的 GPU 暂存与移交

Qwen3.5-4B 的完整前缀状态可以经过 RAM 或 SSD 路径进入 GPU 中的临时序列，再移交给实际执行的序列。研究夹具使用 llama.cpp b10923 的公开接口和预编译 Vulkan 库，无需修改服务端。它把模拟中「临时对象可恢复」这一前提与真实混合状态对齐，但保持固定、同步的执行顺序。

第一项实验验证恢复语义。固定 1,152 或 1,536 token 检查点，每个深度 4 组；以从同样深度直接续算为参照，随机顺序比较主机内存、文件及设备快照。比较后续 8 个贪心位置的完整词表分数，强制各条路径用同一历史 token，以免第一个差异放大。24 个恢复的 8 个贪心位置均一致，归档精度内分数差为零。恢复中位数（设备 / RAM / 文件）分别为 4.57 / 11.41 / 43.87 ms 和 5.32 / 12.40 / 58.63 ms。文件未做冷读控制，计时不含提前保存，也无并发同行。设备快照的主机元数据很小，张量保存在同一模型上下文的设备缓冲区。

第二项实验让两个真实前缀竞争暂存。同行先通过 Windows 无缓冲读取进入主机，再从序列状态装入临时槽 2；目标从 RAM 装入临时槽 1。执行序列为槽 0，使用统一缓存，以序列移交和删除暂存引用消费对象。校验轮在实际续算前重新序列化槽 0，逐字节对照原检查点（包括参与恢复的注意力与循环状态）；计时轮不执行该检查。运行同行 512 个续算 token 后再消费目标。目标在紧额度下的深路径延后到同行计算之后才装入；宽额度或独立额度允许两对象同时暂存。显存池预先分配，逻辑额度以序列化大小计费，因此表中的占用不是对实际显存用量的测量。

1,152→1,536 工况中，180 MiB 共享额度让深候选的装入在 6/6 组等到同行计算之后，装入时刻相对浅路径的配对中位差约 +416 ms；210 MiB 共享或两条路径各 180 MiB 额度时均无准入等待。在三种额度下深边界均 6/6 更快。串行执行把一个装入操作从同行计算前移到其后，不能把两个从零点计时的装入时刻之差视为整条路径的额外代价。移交前的序列化检查逐字节通过，计时轮目标首 token 36/36 一致；布局不同的续算仍可产生最多约 0.064642 的词表分数绝对偏差，不能从精确检查点推出续算过程的数值逐位一致。

把深边界缩到 1,280 token 后，174 MiB 共享池正好仍容纳两个浅对象，但不能并存一个深对象和同行。8 个新提示词配对中，深路径在共享 174 MiB、共享 210 MiB、各自独立 174 MiB 下分别胜出 6、7、8 次；目标耗时的深减浅中位差为 −51.21、−126.26、−155.41 ms。紧额度下深路径 8/8 延后装入，但没有稳定翻转；同行计算与目标续算的绝对耗时存在块间波动，因此不能把三组中位数的差直接当作额度的固定惩罚。该轮首 token 48/48 一致，最大分数绝对差 0.073072。这里仍是固定串行模型；**目前只有 CUDA 设备事件夹具复现了可靠的容量翻转。**

这与 CUDA 夹具取互补证据：真实模型实验验证检查点可以进显存后被消费，并检验一种固定串行规则；CUDA 夹具用硬件事件确定并行窗口与传输就绪时刻。真实模型实验不声称 HTTP 服务支持异步分级预取。

归档：[实际模型三路恢复](evidence/dynamic-gpu-exact-state-7345.csv) · [真实状态竞争与校验](evidence/dynamic-gpu-model-stage-7346.csv) · [较窄边界对照](evidence/dynamic-gpu-model-stage-7347.csv)。

~~~powershell
git clone --depth 1 --branch b10923 https://github.com/ggml-org/llama.cpp.git outputs/llama-b10923-source
g++ -std=c++17 -O2 -Wall -Wextra -I outputs/llama-b10923-source/include -I outputs/llama-b10923-source/ggml/include .local/hardware/bench_gpu_exact_state.cpp data/raw/prefill/llama-vulkan/llama.dll data/raw/prefill/llama-vulkan/ggml.dll -o outputs/bench_gpu_exact_state.exe
$env:PATH = (Resolve-Path data/raw/prefill/llama-vulkan).Path + ';' + $env:PATH
Start-Process -WindowStyle Hidden -Wait -FilePath (Resolve-Path outputs/bench_gpu_exact_state.exe) -ArgumentList 'data/raw/prefill/qwen35-4b-q4km.gguf data/raw/prefill/llama-vulkan outputs/model-stage-repro.bin 6 --stage' -RedirectStandardOutput outputs/model-stage-repro.csv -RedirectStandardError outputs/model-stage-repro.log
python -m scripts.gpu.experiments.study_gpu_exact_state --exact docs/evidence/dynamic-gpu-exact-state-7345.csv --stage outputs/model-stage-repro.csv
~~~
