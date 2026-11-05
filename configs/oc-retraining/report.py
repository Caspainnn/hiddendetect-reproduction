import json
from core import OUT,HEADS,SEEDS

def main():
    r=json.loads((OUT/'retraining.json').read_text());d=json.loads((OUT/'data-summary.json').read_text());v=json.loads((OUT/'verification.json').read_text());assert v['status']=='passed'
    lines=['# OC-H3：重训检测头的差别','',f"旧完整训练1600条；匹配旧源、新OR源各{d['new_training_count']}条。测试固定1000正常/500危险，校准固定400正常。主种子20261104；参数不重新搜索。",'','这是已观察H2测试集上的历史重训诊断，不是独立新确认。只训练小检测头，原大模型不更新。新旧小样本组使用相同训练规模和优化步数。','', '| 头/组 | AUROC | pAUC≤0.1 | 检出率（500危险） | 误报率（1000正常） |','|---|---:|---:|---:|---:|']
    for m in r['metrics']:lines.append(f"| {m['id']} | {m['auroc']:.6f} | {m['pauc']:.6f} | {m['tpr']:.2%} ({m['tpr_count']}) | {m['fpr']:.2%} ({m['fpr_count']}) |")
    lines+=['','## 预先固定的九个比较','','pAUC差区间按9比较校正，每个99.444444%；区间跨0视为未确认。','', '| 比较（左−右） | pAUC差 | 校正区间 | 判定 |','|---|---:|---|---|']
    for c in r['comparisons']:lines.append(f"| {c['left']} − {c['right']} | {c['delta_pauc']:+.6f} | [{c['pauc_delta_ci994444'][0]:+.6f}, {c['pauc_delta_ci994444'][1]:+.6f}] | {c['decision']} |")
    lines+=['','## 三个种子全部结果','','| 组/头 | 种子 | AUROC | pAUC | 检出率 | 误报率 |','|---|---:|---:|---:|---:|---:|']
    for seed in SEEDS:
        for m in r['seed_metrics'][str(seed)][1:]:lines.append(f"| {m['id']} | {seed} | {m['auroc']:.6f} | {m['pauc']:.6f} | {m['tpr']:.2%} | {m['fpr']:.2%} |")
    lines+=['','## 解读限制','','or_small对old_full同时改变来源和样本量；来源作用主要看同规模or_small对old_small。old_small对old_full仍包含训练子集与注意力步数变化，不是纯因果规模效应。只有一个固定训练子集，三个种子不覆盖抽样不确定性。OR标签来自公开基准，不等同真实部署。','','全部模型/分数/计数复算通过；10000次独立直方图ROC复算18个区间。原文、逐条F、分数、权重留在学校，外传仅汇总。','']
    (OUT/'report-2026-11-04.md').write_bytes('\n'.join(lines).encode());print('OC_H3_REPORT_COMPLETE',flush=True)
if __name__=='__main__':main()
