# R08 协作者提交索引

本次提交汇集公开归档复核、队列压力、预取事件、尾部归因与后继联动的代码和证据。

| 内容 | 入口 |
| --- | --- |
| B481、B482 与 R07 服务归档复核 | [归档复核](r08-collaborator-archive-review.md) |
| 同平均到达率的突发与排队对照 | [配对复算](r08-collaborator-arrival-review.md)、[队列压力](r08-collaborator-queue-pressure.md) |
| 容量与预取过程 | [预留、搬运与使用](r08-collaborator-reservation-review.md) |
| 尾部与 SSD 搬运时机 | [尾部归因](r08-collaborator-tail-review.md)、[发出时机](r08-collaborator-ssd-dispatch-review.md)、[跨窗负对照](r08-collaborator-dispatch-controls.md) |
| 后继联动及完整公开矩阵计时 | [算法消融与计时](r08-collaborator-follower-ablation.md)、[含 1.5 倍成本门槛的计时证据](evidence/r08-visible-follower-io-ratio-timed.jsonl.gz) |

## 暂存容量与后继联动补充复核

B481（4B／12×）的冻结恢复点与后继最深边界合计需要 210.25 MiB。176／180 MiB 下，当前改选让后继约早 56.9 ms 拿到预留，全窗平均响应改善 13.251 ms。256 MiB 下，冻结选点已经更早为后继预留，试算维持原选点。B482（27B 代理／16×）在 512 MiB 下，两者依赖可同时装入，扩展试算仍改善 252.566 ms。因此，只判断两份依赖能否同时装入，会漏掉 B482 一类改选机会。预留时刻对照只强制当前请求，所列全窗改善来自完整候选策略回放。

逐配置归档包含 8 个冻结公开配置。[容量复核记录](evidence/r08-joint-staging.jsonl.gz)

```powershell
python -m scripts.validation.audit_r08_joint_staging
python -m scripts.validation.summarize_r08_visible_timing docs/evidence/r08-visible-follower-io-ratio-timed.jsonl.gz
```

目前结论来自公开归档和模拟回放。候选策略仍有超出替换门槛的配置，独立留出与真实服务闭环验收尚未完成。
