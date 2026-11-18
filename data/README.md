# 百炼公开生产请求轨迹

来源：[Qwen-Bailian Anonymous Dataset](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)，Apache-2.0。版本、字节数与上游 Git LFS SHA-256 见 [manifest](bailian-manifest.json)。

四份原始轨迹合计约 312 MB，存放在本地仓库的 `data/raw/bailian/`，该目录由 Git 忽略。克隆本项目后，从仓库根目录执行：

```text
python -m scripts.data.fetch_bailian
python -m scripts.data.fetch_bailian --verify-only
```

回放保留原到达间隔，在相同的检查点与放置规则下比较策略。上游哈希对应应用模板后的输入；不能从哈希反推文本，也不应再次套用聊天模板。窗口和运行命令见[实验口径与复现](../docs/experiment-index.md)。
