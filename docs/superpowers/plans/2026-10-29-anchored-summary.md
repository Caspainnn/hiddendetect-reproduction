# R28—R29 Anchored summary implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Inline execution; one fresh-context read-only whole-branch review.

**Goal:** 在强末位不变的基础上，验证少量融合和报警边界局部融合，能否稳定利用用户摘要而不整体压低报警。

**Architecture:** 固定R27第一个种子draw0的岭回归B／均值／动态／双随机四个头，不按已看测试选种子。原dev600作为本轮明确的元训练集，完全不在这四头的训练输入中。训练折内以正常样本对齐摘要分数的均值和标准差，比较全局凸融合和紧支撑三角门控融合。冻结后同时测试新Aegis与新ToxicChat毒性辅助样本，分开解释。

**Tech Stack:** 复用NumPy、scikit-learn、PyTorch、现有完整表示与缓存，不安装新依赖、不训练底座。

**Spec:** 用户要求根据R27结论提出并执行进一步优化，自主推进；新增监督预算和所有对照明确记录。

## Global constraints

- R28全局融合：s=B+alpha*(E_aligned−B)。每摘要alpha为[.0625,.125,.1875,.25,.375,.5,.625,.75,1]，9候选。
- R29局部门控：s=B+alpha*g(B)*(E_aligned−B)，g=max(0,1−abs(B−tau)/width)。alpha=[.25,.5,1]×width_factor=[.5,1,2]，9候选。tau为元训练折正常B分数95%分位，width=正常B标准差×factor。远离门控区分数严格等于B。这里是报警边界，不解释为模型认知不确定性。
- E_aligned=(E−mu_E)*sd_B/sd_E+mu_B；均值、标准差、tau仅来自当前元训练折正常输入，sd下限1e−12。均值、动态、同维双随机各参加两策略，54候选＋B。随机仍是实际用户状态的两份随机加权摘要。
- 元训练600，固定相同重复五折两次（55×10=550评分条件），按平均pAUC(.1)每类选参，列表平局优先较弱alpha／较窄门。随机仅控制不参与最终选择。全体可选原始最佳与B的一SE保守选择均记录，一SE为启发式。
- 最终固定11头：B、三份原联合头、六份融合头、B_refit。B_refit用相同基础1600＋元训练600共2200标签重新拟合完整末位岭回归，9个lambda×相同10折调参，避免把额外600标签的价值误归为融合。B_refit不参与元训练CV的总体选择（它使用了该600的标签）。
- 元训练600有历史观察，不冒充未见验证。本轮不改变R27五组训练，不增加集成，摘要固定gamma1和四锚点，所有既有冻结代码不改。
- 新Aegis目标每类500；新ToxicChat仅human_annotation=true，目标每类200，可用不足时取两类最小值向下25取整，至少每类50，否则记录不可用，不拿旧输入冒充新确认。两任务不混算平均。历史全部ID、规范化精确／char3—5gram TF-IDF余弦≥.92近重复排除，仍限输入≤512。
- 固定C1000、历史Toxic400、Aegis正常校准400和Toxic正常校准400保持ID与标签。新确认不调参、不校准。历史集仅描述性。Aegis新确认仍为同源；Toxic新样本是跨来源毒性辅助验证，不等同有害请求检测的新领域证明。
- 七个预设对照：gated_dynamic−B、gated_dynamic−global_dynamic、gated_dynamic−gated_mean、gated_dynamic−gated_random、global_dynamic−B、gated_mean−B、gated_dynamic−B_refit。主终点新Aegis pAUC(.1)，AUROC、阈值召回／误报分开报告。5000次分层配对bootstrap，七比较Bonferroni每任务每指标分别校正；条件于固定训练和校准。
- 校准均为同400正常分数95%分位，严格大于报警，无5%真实误报保证。召回与误报必须并列，不用不同误报点宣称等误报收益。
- 原四头已在R27冻结，其分数可作为固定特征提取；不观察新确认指标。新B_refit与元参数必须提交冻结后才统一评分。

## Tasks and review focus

- [ ] 1. 为正常样本尺度拟合、门外完全保持B、alpha0／alpha1、隔离与七对照写失败测试，再实现`anchored_core.py`。
- [ ] 2. `build_manifest.py`锁元训练／固定校准测试／新确认与去重；`extract.py`复用旧2200条特征，新提取dev600及新确认，32旧训练探针核对全五摘要。保存来源commit、shape/dtype、前缀、模型、缓存哈希。
- [ ] 3. `prepare_scores.py`用R27四冻结头提取固定分数，并与历史逐样本分数对齐。`train.py`只用dev600选元参数，另拟合等标签B_refit。保存OOF、90折基线调参记录与全部来源哈希。
- [ ] 4. 一次只读全分支审查，修复需证据。提交冻结后`evaluate.py`统一评分；`verify.py`双端重放640折指标、全部测试条件和bootstrap。新数据不可用于挑最高头。
- [ ] 5. 报告解释全局／局部／随机／额外标签对照，更新R28／R29、Markmap与网页，验证后推送独立分支并同步学校目录。

## Rulings / ledger

用户已授权自主选择方案并验证，不重复询问设计批准。沿用独立实验目录和新分支，不复制旧大型缓存。高维正交残差、学习型token权重作为备选，不与本轮同时引入，避免无法解释效果来源。
