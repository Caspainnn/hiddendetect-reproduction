# OC-A2执行台账

2026-11-04。用户授权继续注意力；执行commit `d20b76d6f0bfece198c0b4c9ff1cba9f7acf5917`，branch `experiment/oc-attention-improvements`。所有执行源码、协议、输入身份及官方依赖源码在训练前封印。

- 合成检查通过：PLE边界和外推、线性编码信息等价、批内独立、有限梯度、CVaR代理及索引状态、模型重载和阈值。
- 199次拟合：165训练侧折内、33最终、1原程序回归；训练墙钟654.108秒。
- 原程序重训逐参数复现H3；33模型在读取目标数据前封印，再执行固定400正常校准和1500历史测试评分。
- 33模型重载，39组点指标与计数；130000次独立直方图ROC，全部bootstrap分布和15项区间复算。最大点指标差2.22e-16。
- 第一次评分在H3旧分数回归处失败：参数一致，因评分后端设置不同，校准分数最大差1.43051147e-6。恢复H3原predict入口并恢复A2后端后重跑评估；未重训、未修改选择或统计规则。失败前已读取校准与测试，未生成evaluation.json；保留attempt1状态与日志及evaluation-resume-seal.json。上轮SVM失败尝试另见原SVM报告。
- 本地仅归档聚合JSON、日志、报告和代码；原文、逐条F、模型、逐条分数及bootstrap分布留校，未下载。
- 最终源码及报告在隔离分支提交；主工作区仅同步指定文件，保留其他工作。页面由原维护源重建；收放、全屏、拖动、鼠标中心缩放及条件加粗用脚本逻辑检查，未将其称为浏览器实测。

[完整报告](../results/oc-attention/report-2026-11-04.md) · [封印](../results/oc-attention/source-lock.json) · [复核](../results/oc-attention/verification.json)
