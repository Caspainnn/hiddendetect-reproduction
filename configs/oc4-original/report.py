from core import *
def main():
    r=load(OUT/'evaluation.json');v=load(OUT/'verification.json');t=load(OUT/'training.json');assert v['passed'] and v['evaluation_sha256']==digest(OUT/'evaluation.json')
    lines=['# OC-O1：原OC-4数据上的SVM与注意力进阶版验证','','2026-11-04。用户指定原积分AUROC0.756547对应的RV1082；该批为项目Aegis历史数据，不是论文XSTest。原训练1600、选择600、正常校准400；固定原14层F。','',
       '本轮重新训练SVM与注意力两组；原积分B及H1原SVM/MLP/注意力/Transformer同时保留。H1与本轮神经训练程序不同，结构改进需对本轮匹配程序基线解释。无独立新确认。','']
    decision={'positive':'本历史批支持提高','worse':'本历史批支持下降','unconfirmed':'改善未确认'}
    for key,cohort in r['cohorts'].items():
        lines += [f'## {key}：'+('主历史测试，27比较修正' if key=='RV' else '补充历史测试，区间仅描述性'),'', '|方案|AUROC|标准化pAUC|检出率（条数）|误报率（条数）|','|---|---|---|---|---|']
        for m in cohort['metrics']:lines.append(f"|{m['id']}|{m['auroc']:.6f}|{m['pauc']:.6f}|{m['tpr']:.2%}（{m['tp']}）|{m['fpr']:.2%}（{m['fp']}）|")
        lines += ['',f"主pAUC差区间覆盖率{cohort['pauc_ci_coverage']:.9%}；标签分层配对bootstrap{REPEATS}次。",'', '|比较|pAUC差|差区间|判断|AUROC差95%|检出率差95%|误报率差95%|','|---|---|---|---|---|---|---|']
        ci=lambda pair:f'[{pair[0]:+.6f},{pair[1]:+.6f}]'
        for c in cohort['comparisons']:lines.append(f"|{c['id']}|{c['delta_pauc']:+.6f}|{ci(c['pauc_delta_ci'])}|{decision[c['decision']]}|{ci(c['auroc_delta_ci95'])}|{ci(c['tpr_delta_ci95'])}|{ci(c['fpr_delta_ci95'])}|")
    lines += ['', '## 所有种子','', '|种子|批次|方案|AUROC|pAUC|检出率|误报率|','|---|---|---|---|---|---|---|']
    for seed,cohorts in r['seed_metrics'].items():
        for key,rows in cohorts.items():
            for m in rows:lines.append(f"|{seed}|{key}|{m['id']}|{m['auroc']:.6f}|{m['pauc']:.6f}|{m['tpr']:.2%}|{m['fpr']:.2%}|")
    lines += ['', '## 选型与成本','', '|方案|所选参数|dev均值pAUC|选定模型训练秒|参数量|','|---|---|---|---|---|']
    for name,param in t['selected'].items():
        candidate=next(c for c in t['search'][name] if c['param']==param);runs=candidate['runs']
        lines.append(f"|{name}|{json.dumps(param)}|{candidate['mean_dev_pauc']:.6f}|{sum(x['cost']['seconds'] for x in runs):.3f}|{runs[0]['cost'].get('parameters','—')}|")
    lines += ['',f"共{t['total_fits']}次拟合（SVM45、神经99），选定{t['selected_models']}模型。训练墙钟{t['seconds']:.3f}秒。SVM固定确定性种子，神经三种子；SVM每头9候选、神经每头3候选，预算不同。",'',
       f"复核：{v['model_reloads']}新模型重载、{v['point_metric_sets']}指标组，{v['independent_roc_replicates']}次独立ROC及两批全部54差区间重算通过；H1四旧头全部种子逐分数复现。最大点指标差{v['maximum_point_difference']:.3g}。",'',
       '## 解释边界','',
       '- RV和C都已经多次观察，仅历史诊断；不能声称未见数据确认。',
       '- 神经主机制比较使用匹配全批240更新基线，H1旧头只是历史参照。',
       '- PLE注意力收益需结合线性接口、MLP与差中差判断；门控需结合无门控消融。pAUC头损失、优化器、裁剪和正则共同变化。FT是受控规模官方实现整体迁移。',
       '- 区间条件于训练/选择与固定校准，不覆盖这些环节的不确定性；20000次bootstrap极端分位点仍有Monte Carlo误差。',
       '- 未按测试选种子、翻转风险方向或进一步堆叠组件。原文、F、模型及逐条分数留校。','',
       '[协议](../../docs/oc4-original-protocol-2026-11-04.md) · [训练](training.json) · [指标](evaluation.json) · [复核](verification.json) · [源封印](source-lock.json)']
    (OUT/'report-2026-11-04.md').write_bytes(('\n'.join(lines)+'\n').encode());print('OC_O1_REPORT_READY',flush=True)
if __name__=='__main__':main()
