# OC-A2注意力改进

执行源码封印为d20b76d；199次拟合完成后，旧H3回归因评分后端设置差异被拦下。恢复补丁529fe8c只调用H3原predict并恢复A2设置，未重新训练。原evaluate.py及source-lock.json完整保留；resume.py生成的评估文件SHA256记录在evaluation-resume-seal.json，失败日志和状态单独留存。

学校机隔离worktree使用共享env中的Python；官方rtdl0.0.13及LibAUC1.4.0只在cache/vendor解压，按source-lock.json校验全部Python源。全新运行使用`python configs/oc-attention/run.py`；它故意拒绝覆盖已有训练及结果。resume.py只适用于已有attempt1证据、未写evaluation.json的本次恢复，不能拿来重新搜索。

本地只保存聚合结果。重新执行需要校内原始缓存、模型、逐样本分数及归档依赖，不能仅凭本地聚合文件重训。

[执行协议](../../docs/oc-attention-protocol-2026-11-04.md) · [结果解释](../../docs/oc-attention-findings-2026-11-04.md) · [执行台账](../../docs/oc-attention-execution-ledger-2026-11-04.md) · [完整报告](../../results/oc-attention/report-2026-11-04.md)
