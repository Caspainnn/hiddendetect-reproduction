import json
from core import *

TITLES={'B':'原积分','SVM':'OC-S1线性SVM',ATT:'原注意力（本轮重训）',TRANS:'原小Transformer（本轮重训）',MLP:'原MLP（本轮重训）',
 GATE:'门控汇总',UNGATED:'无门控汇总',PLE_A:'分段编码＋注意力',LIN_A:'线性编码＋注意力',PLE_M:'分段编码＋MLP',LIN_M:'线性编码＋MLP',PAUC:'CVaR-pAUC注意力',FT:'官方FT-Transformer'}
DIRECT={GATE:ATT,PLE_A:ATT,PLE_M:LIN_M,PAUC:ATT,FT:TRANS}
QUESTIONS={GATE:'换计算权重的方法能否改善？',PLE_A:'更细的F编码能否帮助注意力？',PLE_M:'编码收益是否也能由MLP获得？',PAUC:'改用低误报训练目标是否有效？',FT:'成熟表格注意力架构是否有额外收益？'}

def main():
    r=json.loads((OUT/'evaluation.json').read_text());t=json.loads((OUT/'training.json').read_text());v=json.loads((OUT/'verification.json').read_text());lock=json.loads((OUT/'source-lock.json').read_text())
    assert v['passed'] and v['evaluation_sha256']==digest(OUT/'evaluation.json')
    metrics={m['id']:m for m in r['metrics']};decision={'positive':'本历史批支持提高','worse':'本历史批支持下降','unconfirmed':'改善未确认'}
    fmt=lambda x:f'{x:.6f}';rate=lambda x:f'{x:.2%}';ci=lambda x:f'[{x[0]:+.6f}, {x[1]:+.6f}]'
    lines=['# OC-A2：注意力改进历史诊断','',
      '日期：2026-11-04。固定原14层F；OR训练300（正常/危险各150），训练侧5折选择；同400正常校准，历史测试1000正常/500危险。所有模型先封印再评分。本轮没有独立新确认。','',
      '## 同批全部结果','', '|编号／方案|AUROC|标准化pAUC≤0.1|检出率（条数/500）|误报率（条数/1000）|','|---|---|---|---|---|']
    for m in r['metrics']:lines.append(f"|{m['id']}／{TITLES[m['id']]}|{fmt(m['auroc'])}|{fmt(m['pauc'])}|{rate(m['tpr'])}（{m['tp']}/500）|{rate(m['fpr'])}（{m['fp']}/1000）|")
    lines+=['','## 五项问题闭环','']
    for name,base in DIRECT.items():
        m,b=metrics[name],metrics[base];comparison=next(c for c in r['comparisons'] if c['id']==name)
        lines+=[f'### {name}：{QUESTIONS[name]}','',f"直接对照为{base}（{TITLES[base]}）。同批AUROC {fmt(b['auroc'])} → {fmt(m['auroc'])}；pAUC {fmt(b['pauc'])} → {fmt(m['pauc'])}；检出率 {rate(b['tpr'])} → {rate(m['tpr'])}；误报率 {rate(b['fpr'])} → {rate(m['fpr'])}（{b['fp']} → {m['fp']}条）。主pAUC差{comparison['delta_pauc']:+.6f}，15比较校正99.666667%区间{ci(comparison['pauc_delta_ci'])}，{decision[comparison['decision']]}。",'']
    lines+=['## 15项预设主比较','', '|比较|pAUC差|99.666667%主差区间|判断|AUROC差95%|检出率差95%|误报率差95%|','|---|---|---|---|---|---|---|']
    for c in r['comparisons']:
        label=' + '.join(f'{w:+g}×{n}' for n,w in c['terms'])
        lines.append(f"|{c['id']}：{label}|{c['delta_pauc']:+.6f}|{ci(c['pauc_delta_ci'])}|{decision[c['decision']]}|{ci(c['auroc_delta_ci95'])}|{ci(c['tpr_delta_ci95'])}|{ci(c['fpr_delta_ci95'])}|")
    lines+=['','差中差为(分段编码注意力−线性编码注意力)−(分段编码MLP−线性编码MLP)。只有该区间全正才支持编码收益对注意力额外有利；直接比较两个终点模型不能替代它。主pAUC以外的95%区间为描述性，率差单位为比例。','',
      '## 训练侧选型、容量与成本','', '|方案|所选学习率|5折均值pAUC|参数量|最终3种子耗时秒|','|---|---|---|---|---|']
    for name in NAMES:
        runs=[x for x in t['runs'] if x['id']==name];lr=t['selected'][name];candidate=next(c for c in t['cv'][name] if c['lr']==lr)
        lines.append(f"|{name}|{lr:g}|{candidate['mean_pauc']:.6f}|{runs[0]['parameters']}|{sum(x['seconds'] for x in runs):.3f}|")
    lines += ['',f"成功运行共{t['total_fits']}次拟合（165折内、33最终、1旧程序回归），训练阶段墙钟{t['seconds']:.3f}秒。各神经头3学习率、相同5折、240次全批更新、宽度16。SVM参照来自OC-S1的9候选选择，跨家族预算不同。门控、PLE与FT参数量不同，不宣称全部等容量。",'',
      '## 全种子结果','', '|种子|方案|AUROC|pAUC|检出率|误报率|','|---|---|---|---|---|---|']
    for seed,rows in r['seed_metrics'].items():
        for m in rows:lines.append(f"|{seed}|{m['id']}|{fmt(m['auroc'])}|{fmt(m['pauc'])}|{rate(m['tpr'])}|{rate(m['fpr'])}|")
    old=r['historical_replay']
    lines+=['','## 复现与实现边界','',
      f"H3原程序注意力逐参数与逐样本分数复现：AUROC {old['auroc']:.6f}、pAUC {old['pauc']:.6f}、检出{old['tp']}/500、误报{old['fp']}/1000。本轮改为全批240更新，主比较因此使用本轮匹配程序重训基线，不将训练程序差别算作新结构作用。",
      'FT使用作者rtdl0.0.13发行实现；16宽/1块是本次受控配置，不是作者默认大模型。pAUC使用官方LibAUC1.4.0的pAUC_CVaR_Loss和SOPA，保留逐正样本辅助状态。损失、优化器、梯度裁剪与正则共同变化，不能单归因损失。PLE与门控是依据论文公式的本项目迁移，不称作者原任务精确复现。',
      f"{v['model_reloads']}模型重载，{v['point_metric_sets']}组指标/阈值/计数，{v['independent_roc_replicates']}次独立直方图ROC及15项全部区间复算通过。最大点指标差{v['maximum_point_difference']:.3g}。辅助状态、完整分数及重采样分布留校。",
      '合成检查包含PLE边界与外推、线性编码信息等价、批内样本独立性、梯度有限、官方CVaR逐样本状态/代理损失和模型重载。数据封印复用H3近重复审计并重查四类身份字段互斥。','',
      '## 限制与下一步','',
      '- 本次只是已观察OR-Bench上的历史诊断，不是独立新确认，也不代表真实部署或多模态泛化。',
      '- 5折每折正常30，低误报选择离散；CV只用主种子。三个最终种子反映初始化变化，不反映训练样本抽样。所有种子均报告，不按测试选最好者。',
      '- 主区间以本次训练/选择和固定400正常校准为条件，不覆盖选型与校准不确定性。阈值目标5%不保证测试误报5%。',
      '- 保存失败或未确认项，不根据本轮测试继续搜索或堆叠组件。任何正向结果仍需未用数据独立确认。',
      '- 跨结构表现来自此固定宽度、步数、学习率网格，不否定所有注意力、数值编码或pAUC方法。','',
      '环境：'+json.dumps(lock['environment'],ensure_ascii=False)+'。官方依赖归档及源码哈希见source-lock.json；下载仅在隔离目录。','',
      '[执行协议](../../docs/oc-attention-protocol-2026-11-04.md) · [训练记录](training.json) · [指标与区间](evaluation.json) · [复核](verification.json) · [源码与依赖封印](source-lock.json)']
    (OUT/'report-2026-11-04.md').write_bytes(('\n'.join(lines)+'\n').encode());print('OC_A2_REPORT_READY',flush=True)
if __name__=='__main__':main()
