# OC-O1：原OC-4划分验证

复用H1原train1600/dev600/cal400、RV1082和C1000，SVM与注意力两组重新训练。主测试RV原积分AUROC0.7565473145780051；所有测试均历史，非独立新确认。

学校机隔离worktree运行`../../env/bin/python configs/oc4-original/run.py`。需要H1原始缓存和模型、A2已哈希官方依赖；本地聚合归档不能独立重训。入口拒绝覆盖既有结果。首次SVC未收敛在测试前终止，取消迭代上限后重新封印；原SVM阶段代码保持独立冻结，OC-O1修订版另归档在结果source-snapshots/oc-svm-core.py，其hash对应OC-O1 source-lock。

144成功拟合、38新选定模型、126指标组、84万独立ROC与两批54差区间复算。27项主比较仅RV修正，C全部95%区间为描述性；H1旧四头全部种子分数复现。旧头与本轮神经训练程序不同，结构收益用本轮匹配基线判断。

[完整报告](../../results/oc4-original/report-2026-11-04.md) · [结果解释](../../docs/oc4-original-findings-2026-11-04.md) · [协议](../../docs/oc4-original-protocol-2026-11-04.md) · [执行台账](../../docs/oc4-original-execution-ledger-2026-11-04.md)
