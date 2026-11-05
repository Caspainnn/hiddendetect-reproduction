# OC-H2复核入口

学校工作树：`HiddenDetect-reproduction/cache/oc4c-worktree`。预先冻结源码4437cea；完整源锁见results/oc-confirmation/source-lock.json。

先固定OR-Bench官方公开commit和原文件SHA，再全历史去重、样本封印、8旧训练探针F复核、1900条新来源F提取、固定模型确认、15次训练重采样、1000校准重采样、独立统计复核。原H1的模型、参数和结果不覆盖；H2阶段一不被后续稳定性诊断替换。

完整原文、F、逐条分数和模型保留学校。学校工作树根目录使用既有env：

```text
env/bin/python configs/oc-confirmation/verify.py
env/bin/python configs/oc-confirmation/report.py
```

verify重载9个旧种子模型、15个稳健性模型，独立分档ROC复算10000次区间，并使用partition/searchsorted复核1000校准范围和机制门槛。report可重建数值报告；解释性结论另见docs/oc-heads-confirmation-findings-2026-11-04.md。

学校机器无外网，因此官方公开数据先在本地校验后上传。没有读取需申请的WildGuardMix或从镜像绕过访问限制。重做实验需另建输出目录和新封印，不能把已评分的OR-Bench记录再次称为新确认。
