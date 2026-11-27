# R10-DATA-01 | 输入构建、审计与留出冻结

## OpenAssistant 独立文本输入

独立文本来自 OpenAssistant/oasst1 官方数据快照中的真实英文对话树。固定提交 `fdf72ae0827c1cda404aff25b6603abec9e3399b`，使用 `2023-04-12_oasst_ready.trees.jsonl.gz`。随源下载的 LICENSE 和 README 标明 Apache-2.0；原始对话和分词序列留在本地忽略目录，提交的清单只含哈希、长度、候选和分组信息。源文件 SHA-256 见[输入清单](evidence/r10-oasst-inputs.json)，许可文件 SHA-256 为 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`。

逐树读取 10,364 棵已发布树，其中 3,670 棵的根是经审核的英文提示，3,099 棵能构成通过审核的多轮路径。每棵选两个不同的真实用户提示，各沿所属分支提供固定历史。跳过删除、未通过审核、合成、控制标记冲突或超上下文的消息，不截断提示。用固定 Qwen GGUF 的纯文本模板分支，无工具和系统消息，关闭思考，保留语料里的助手历史。模型词表以 CPU 方式加载；未加载权重或启动服务。长度在 128–1,792 token 之内；后续服务试验预留 128 个输出 token。

分成 4 个校准块和 6 个留出块，每块 32 棵会话树、64 次请求。源码按固定哈希排序选择，按根文本规范化哈希、会话树、消息和完整 token 指纹检查重复；未发现跨块重叠。每块至少含 8 对共享 512 token 的提示与 4 棵用编码相关关键词或代码围栏标记的树。编码标签用来分层，不能当作人工审定的编程任务分类，也不代表 agent 工具执行。

| 分组 | 请求数 | 编码标签会话树 | 无整块共享的配对 | 最深共享边界 512 | 最深共享边界 1,024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 校准 | 256 | 16 | 96 | 30 | 2 |
| 留出 | 384 | 24 | 144 | 45 | 3 |

表中的边界由真实词元序列的最长公共前缀验证，同时能在边界之后继续计算。前一请求不使用后到请求产生的状态，第二请求候选只来自同棵树中已排在它之前的配对。各分支历史是固定文本，不依赖前一臂输出。合成的开环相对时刻将八个会话穿插提交，树内顺序按原消息时间；相对时刻不是服务端实测到达。样本按共享和编码标签富集，不能估计两类会话在生产中的总体出现率。留出中只发现 3 对同时有 512 与 1,024 的非零候选；多边界竞争仍须由更丰富的百炼重建或控制组交叉检查。

只发布结构摘要，输入可由[构建器](../scripts/data/build_r10_oasst.py)从固定源重建，再用[独立复核器](../scripts/data/audit_r10_oasst.py)按实际词元查重并核计边界。[逐块复核摘要](evidence/r10-oasst-audit.json)

复算示例（从仓库根目录执行，输出用新路径）：

~~~powershell
$src = (Resolve-Path outputs/r10/upstream/llama.cpp-8a56aedd6).Path
$runtime = (Resolve-Path outputs/r10/llama-native-vulkan-build/bin).Path
g++ -std=c++17 -O0 -Wall -Wextra -I "$src/include" -I "$src/ggml/include" scripts/data/r10_tokenize_stream.cpp "$runtime/libllama.dll" "$runtime/ggml.dll" "$runtime/ggml-base.dll" -o outputs/r10/r10_tokenize_stream.exe
python -m scripts.data.build_r10_oasst --source data/raw/oasst1/2023-04-12_oasst_ready.trees.jsonl.gz --model data/raw/prefill/qwen35-4b-q4km.gguf --worker outputs/r10/r10_tokenize_stream.exe --runtime $runtime --manifest outputs/r10/oasst-rerun.json --tokens outputs/r10/oasst-rerun.jsonl.gz --log outputs/r10/oasst-rerun.log
python -m scripts.data.audit_r10_oasst outputs/r10/oasst-rerun.json outputs/r10/oasst-rerun.jsonl.gz
~~~

## 百炼重建输入

校验四份公开轨迹的字节数与 SHA-256，对照 218 个既往源窗身份。每源取一个先导窗、两个预留窗，按原始行数切出 256 行缓存历史与 96 个在线到达。按完整哈希前缀生成稳定的 token ID。候选检查点仅来自前 256 行的固定历史。提示上限 2,560 tokens，保留实际到达时间差，按窗口登记被截断请求与多候选请求数量。12 窗共 1,152 个请求。这一批使用重建 token ID，保留轨迹中的到达与完整块前缀关系。[输入清单](evidence/r10-bailian-inputs.json) · [构建入口](../scripts/data/build_r10_bailian.py)

独立核对覆盖所有 1,152 个请求、942 个候选引用和 66,013 个窗内完整哈希前缀节点。按源哈希和重建 token 的对应关系逐节点验证后，窗内任意两请求在 16-token 完整块上的最长共享前缀保持不变；末尾不足一块不在上游可核对范围内。403 个请求被 2,560-token 上限截断。审计还核对原请求时间差、候选均出自固定历史，以及先导窗与预留窗未混用。[审计结果](evidence/r10-bailian-audit.json)

~~~powershell
python -m scripts.data.build_r10_bailian --out-dir outputs/r10-bailian-recheck --summary outputs/r10-bailian-recheck/inputs.json
python -m scripts.data.audit_r10_bailian outputs/r10-bailian-recheck/inputs.json outputs/r10-bailian-recheck --out outputs/r10-bailian-recheck/audit.json
~~~

## 正式留出

预填充留出选入全部六个 OASST 预留块。百炼四源先各取一个按固定哈希排序的保留块，再从余下四块取两条。12 块各做 5 对，两臂顺序预先固定。[成员与顺序](evidence/r10-local-holdout-layout.json) · [执行锁](evidence/r10-local-holdout-execution-lock.json)

服务留出与预填充主留出独立，含 2 个新增 OASST 块（128 请求、64 棵会话树）和 2 个百炼保留窗（192 请求）。OASST 按会话树、规范化根文本、消息和完整提示 token 指纹排重，百炼按源和原始窗口区间核对。各块覆盖短／混合输出及三个校准锚点。[成员与排重](evidence/r10-local-service-holdout-layout.json) · [执行锁](evidence/r10-local-service-holdout-execution-lock.json)
