import json
from pathlib import Path
from core import ROOT,OUT,BASE,COST,RBF,PAUC,CTRL,SEEDS,digest

def main():
    r=json.loads((OUT/'evaluation.json').read_text());t=json.loads((OUT/'training.json').read_text());v=json.loads((OUT/'verification.json').read_text())
    assert v['passed'] and v['evaluation_sha256']==digest(OUT/'evaluation.json')
    titles={'B':'原积分B',BASE:'调参线性SVM',COST:'代价敏感线性SVM',RBF:'RBF-SVM',PAUC:'线性pAUC-SVM',CTRL:'线性核SVC','H3-C10':'H3固定C10复核','cost-fixed-C-unweighted':'代价头同C、不加权'}
    def fmt(x):return f'{x:.6f}'
    def rate(x):return f'{x:.2%}'
    def interval(x):return f'[{x[0]:+.6f}, {x[1]:+.6f}]'
    verdict={'positive':'本历史批支持提高','worse':'本历史批支持下降','unconfirmed':'改善未确认'}
    lines=['# OC-S1：SVM改进历史诊断','',
        '日期：2026-11-04。固定14层F；同来源训练300（正常/危险各150），5折训练侧选择，400正常校准，历史OR-Bench测试1000正常/500危险。三项采用用户指定的四级编号。没有独立新确认，不将历史批结果称为泛化证明。','',
        '## 同批结果','', '|编号／方案|AUROC|标准化pAUC≤0.1|检出率（条数/500）|误报率（条数/1000）|', '|---|---|---|---|---|']
    for m in r['metrics']:lines.append(f"|{m['id']}／{titles[m['id']]}|{fmt(m['auroc'])}|{fmt(m['pauc'])}|{rate(m['tpr'])}（{m['tp']}/500）|{rate(m['fpr'])}（{m['fp']}/1000）|")
    lines+=['','## 四项预设主比较','', '|候选−直接对照|pAUC差|四比较校正98.75%差区间|结论|AUROC差95%区间|检出率差95%区间|误报率差95%区间|','|---|---|---|---|---|---|---|']
    for c in r['comparisons']:lines.append(f"|{c['left']} − {c['right']}|{c['delta_pauc']:+.6f}|{interval(c['pauc_delta_ci9875'])}|{verdict[c['decision']]}|{interval(c['auroc_delta_ci95'])}|{interval(c['tpr_delta_ci95'])}|{interval(c['fpr_delta_ci95'])}|")
    lines+=['','除主pAUC差区间外，其余95%区间为描述性补充；检出率/误报率差采用比例单位，例如0.01表示1个百分点。所有区间以本次训练/选型和冻结校准为条件。','',
        '## 逐问题解释','']
    metrics={m['id']:m for m in r['metrics']};b=metrics[BASE]
    for name,question,change in [(COST,'更重罚误拦正常输入能否改善低误报表现？','类别加权，其他LinearSVC结构不变；超参数均在训练侧重新选择'),
            (RBF,'两类之间需要弯曲的边界吗？','RBF核，并保留线性核SVC对照；与LinearSVC还有损失/求解器差异'),
            (PAUC,'直接关注低误报区间能否改善线性排序？','top-negative成对hinge代理目标与cutting-plane求解器，不增加输入信息')]:
        m=metrics[name];c=next(c for c in r['comparisons'] if c['left']==name and c['right']==BASE)
        lines += [f'### {name}：{question}','',f"改动：{change}。同批调参线性SVM → 候选：AUROC {fmt(b['auroc'])} → {fmt(m['auroc'])}；pAUC {fmt(b['pauc'])} → {fmt(m['pauc'])}；检出率 {rate(b['tpr'])} → {rate(m['tpr'])}；误报率 {rate(b['fpr'])} → {rate(m['fpr'])}（{b['fp']} → {m['fp']}条）。主差区间{interval(c['pauc_delta_ci9875'])}，{verdict[c['decision']]}。",'']
    fixed=metrics['cost-fixed-C-unweighted'];weighted=metrics[COST]
    lines += [f"代价头固定C消融：正常权重1 → 所选权重{t['selected'][COST]['safe_weight']:g}，pAUC {fmt(fixed['pauc'])} → {fmt(weighted['pauc'])}；检出率 {rate(fixed['tpr'])} → {rate(weighted['tpr'])}；误报率 {rate(fixed['fpr'])} → {rate(weighted['fpr'])}。仅点值描述，不另称校正显著。",'',
        '## 训练侧选型与成本','', '|方案|选择参数|5折平均pAUC|CV拟合数|最终拟合数|CV＋最终耗时秒|','|---|---|---|---|---|---|']
    for name,param in t['selected'].items():
        candidate=next(c for c in t['cv'][name] if c['param']==param);runs=[v for v in t['runs'] if v['id']==name]
        seconds=sum(f['seconds'] for c in t['cv'][name] for f in c['folds'])+sum(v['seconds'] for v in runs)
        lines.append(f"|{name}|{json.dumps(param,ensure_ascii=False)}|{candidate['mean_pauc']:.6f}|45|{len(runs)}|{seconds:.3f}|")
    lines+=['',f"共{t['total_fits']}次拟合，训练阶段墙钟{t['seconds']:.3f}秒。5折验证每折仅30正常/30危险，低误报估计离散。各方案9候选但搜索空间不同，不宣称穷尽最优。C在sklearn求和损失与pAUC平均损失之间含义不同。",'',
        '## 全种子结果','', '|种子|方案|AUROC|pAUC|检出率|误报率|','|---|---|---|---|---|---|']
    for seed,rows in r['seed_metrics'].items():
        for m in rows:lines.append(f"|{seed}|{m['id']}|{fmt(m['auroc'])}|{fmt(m['pauc'])}|{rate(m['tpr'])}|{rate(m['fpr'])}|")
    solvers=[f['solver'] for candidates in t['cv'].values() for c in candidates for f in c['folds'] if f['solver']]+[v['solver'] for v in t['runs'] if v['solver']]
    lines+=['','当前SVM求解器为确定性；重复种子相同不代表独立训练抽样的稳定性。','', '## 实现与复核','',
        f"pAUC方案是本项目的论文思路迁移，不是作者求解器精确复现。{len(solvers)}个pAUC拟合均通过收敛检查，最大oracle间隙{max(x['surrogate_gap'] for x in solvers):.9g}，最大primal-dual gap{max(x['primal_dual_gap'] for x in solvers):.9g}。自检包含枚举负样本子集oracle、有限差分子梯度、r=1与基线一致性、QP证书及模型重载。",
        f"H3固定C10逐样本分数、AUROC0.959700、pAUC0.857305和382检出/40误报复现通过。{v['model_reloads']}个模型重载，{v['point_metric_sets']}组指标与阈值/计数复核，{v['independent_roc_replicates']}次独立直方图ROC复算，四组全部区间复核通过。原H3清单与输入hash封印、训练与校准/测试ID/规范化组/文本及输入hash互斥检查通过；复用已有近重复排除证据。",'',
        '## 证据与限制','',
        '- 主种子20261104，10000次标签分层配对bootstrap，四比较Bonferroni校正。阈值为400正常最终分数第381顺序统计量、严格大于告警；实际误报率并不被固定为5%。',
        '- 所有测试均已观察。仅同一OR-Bench任务的300条训练数据，标签来自公开基准；不能扩大为真实部署或跨领域结论。',
        '- 本轮没有额外训练抽样、校准不确定性或嵌套CV置信区间。pAUC模型仅beta=0.1；不自动推广到其他误报区间。',
        '- 结构和参数没有按本轮测试成绩追加搜索。失败或未确认项保留；注意力实验待单独执行。',
        '- 本地交付仅汇总JSON和中文报告；原文、逐条F、模型、逐条分数与bootstrap数组留学校机器。','',
        '[协议](../../docs/oc-svm-protocol-2026-11-04.md) · [训练选型与求解记录](training.json) · [结果与区间](evaluation.json) · [复核](verification.json) · [封印](source-lock.json)']
    (OUT/'report-2026-11-04.md').write_bytes(('\n'.join(lines)+'\n').encode())
    print('OC_S1_REPORT_READY',flush=True)
if __name__=='__main__':main()
