# OC-H1复核说明

本轮在学校独立工作树 `HiddenDetect-reproduction/cache/oc4-worktree` 运行；训练1600、选择600、400正常校准，8头×3候选×3种子。冻结源码提交1f30ea1，实际封印以results/oc-heads/source-lock.json为准。

完整数据、F、逐条dev/test分数及24个选定模型保留学校。仅汇总结果交付本地，因此以下模型重放应在学校工作树根目录、使用既有env执行：

```text
env/bin/python configs/oc-heads/verify_replay.py
env/bin/python configs/oc-heads/verify_additional.py
env/bin/python configs/oc-heads/verify_statistics_independent.py --root .
```

原verify.py是冻结版本，要求逐位相等，保留其随机森林并行求和的末位差异失败证据；verify_replay.py是后续追加的容差复核，生产模型、分数及选型均未修改。浮点容差1e-12，点指标复现且混淆条数精确一致；dev重放恢复训练时的math-SDPA上下文。独立统计脚本不依赖原bootstrap实现，仅在学校内读取逐条分数并输出汇总。

重新运行实验需新建工作树与输出目录并保留旧封印。执行顺序：契约测试→prepare→probe→train→evaluate→上述三项复核→report。prepare/train/evaluate均拒绝覆盖已有完成文件，不能把历史数据重复称为新独立确认。
