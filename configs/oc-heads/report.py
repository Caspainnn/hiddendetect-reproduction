"""Render completed evidence only; this file is not a training dependency."""
import json
from core import ROOT,OUT,IDS,SEEDS,digest
LABELS={'B':'原积分B','OC-4-1-1':'逻辑回归','OC-4-1-2':'岭回归','OC-4-1-3':'线性SVM','OC-4-2-1':'小型MLP','OC-4-3-1':'浅层梯度提升树','OC-4-3-2':'随机森林','OC-4-4-1':'注意力汇总','OC-4-4-2':'小型Transformer'}
def get(name):return json.loads((OUT/name).read_text())
def main():
    selection=get('selection.json');verified=get('verification.json');extra=get('verification-additional.json');manifest=get('manifest.json')
    assert verified['status']==extra['status']=='passed'
    lines=['# OC-4：相同14层F，八种检测头的历史验证','',
        '日期：2026-11-04；运行OC-H1。八方案均已训练、选择、校准并完成两批历史评估；没有独立新确认。原模型、末位、参照及F不变，不扩充检测器输入。','',
        '训练1600（800正常/800危险）、选择600（300/300），八个头完全相同。每头3候选×3固定种子，共72次拟合；按三种子dev低误报pAUC均值选择参数。主结果固定种子20261104，另外两种子不用于挑测试赢家。各方案另用400正常定阈值（升序381，严格大于告警）。原积分没有本轮新增训练标签，监督头需要额外1600训练+600选择标签。','',
        '核验上次全池审计的数据源未改变，未用正常仍仅10条（危险212），不足每类30门槛。C1000和RV1082均为已观测历史诊断；区间为正也不称新确认，不能直接据此部署替换原方法。','',
        f"8条训练样本鲜提取32F与旧缓存逐值差0；模型重载重放{verified['score_values_replayed']}个分数最大差{verified['maximum_score_difference']}；另复算全部24个选定模型的dev分数，并核对bootstrap依赖Git blob及输入封印。",'',
        f"原逐位相等检查因随机森林并行求和的浮点末位差异失败；保留原校验及生产分数，追加1e-12容差复核后，点指标最大差{verified['maximum_metric_difference']}、阈值最大差{verified['maximum_threshold_difference']}、混淆条数精确一致。dev恢复训练时的math-SDPA计算上下文重放，最大差{extra['maximum_dev_score_difference']}。该追加审计不改变模型、选型或生产评分。",'',
        '## 主种子结果','',
        'pAUC为FPR≤0.1的标准化值。主差区间采用10000次标签分层配对bootstrap、8比较Bonferroni（各99.375%）；条件于已训练头、选择规则与固定校准。加粗仅表示点值改善，不表示统计确认。']
    for batch in ['C','RV']:
        e=get(f'evaluation-{batch}.json');b=e['metrics'][0]
        lines+=['',f"### 历史{batch}（{e['counts'][0]}正常/{e['counts'][1]}危险）",'',
            '|实验|方法|AUROC|低误报pAUC|检出率|误报率（条数）|对B的pAUC差及99.375%区间|历史解释|',
            '|---|---|---|---|---|---|---|---|']
        for m in e['metrics']:
            def value(key,percent=False):
                text=f"{m[key]:.2%}" if percent else f"{m[key]:.6f}"
                improvement=m['id']!='B' and (m[key]<b[key] if key=='fpr' else m[key]>b[key])
                return '**'+text+'**' if improvement else text
            if m['id']=='B':interval='—';decision='基线'
            else:
                lo,hi=m['pauc_delta_ci99375'];interval=f"{m['delta_pauc']:+.6f} [{lo:+.6f}, {hi:+.6f}]"
                decision={'positive_interval':'本批正差区间；仍属历史诊断','worse':'本批低误报表现更差','unconfirmed':'改善未获区间支持'}[m['historical_decision']]
            lines.append(f"|{m['id']}|{LABELS[m['id']]}|{value('auroc')}|{value('pauc')}|{value('tpr',True)}|{value('fpr',True)}（{m['fpr_count']}/{m['fpr_n']}）|{interval}|{decision}|")
        lines+=['','机制比较只作描述性95%，不借用主家族显著性：','', '|左侧方案|右侧直接对照|pAUC差|描述性95%差区间|','|---|---|---|---|']
        for m in e['mechanisms']:
            lo,hi=m['descriptive_ci95'];lines.append(f"|{m['left']}|{m['right']}|{m['pauc_delta']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|")
    lines+=['','## 三种子稳定性','',
        '以下是同一选定参数下三个分别训练的头；范围不是置信区间，也不是集成模型成绩。主种子已经预先固定。', '',
        '|批次|实验|AUROC最小～最大|pAUC最小～最大|检出率最小～最大|误报率最小～最大|','|---|---|---|---|---|---|']
    for batch in ['C','RV']:
        seeds=[get(f'seed-{seed}-{batch}.json') for seed in SEEDS]
        for name in IDS:
            values=[next(m for m in seed if m['id']==name) for seed in seeds]
            def span(key,percent=False):
                a,z=min(v[key] for v in values),max(v[key] for v in values)
                return f'{a:.2%}～{z:.2%}' if percent else f'{a:.6f}～{z:.6f}'
            lines.append(f"|{batch}|{name}|{span('auroc')}|{span('pauc')}|{span('tpr',True)}|{span('fpr',True)}|")
    lines+=['','## 选型与成本','', '|实验|选定参数|平均dev pAUC|3候选×3种子累计拟合秒数|选定神经头参数量|','|---|---|---|---|---|']
    for row in get('search.json'):
        name=row['id'];best=selection['selected'][name];cost=sum(r['cost']['seconds'] for c in row['candidates'] for r in c['runs']);params=best['models'][0]['cost']['parameters']
        lines.append(f"|{name}|{best['parameter']}|{best['mean_dev_pauc']:.6f}|{cost:.2f}|{params if params is not None else '—'}|")
    lines+=['', '神经头固定120epochs、batch256、AdamW、dropout0，具体参数见冻结协议。注意力汇总与Transformer使用相同16维嵌入、层标识、查询池化及输出，但Transformer额外含编码器，参数量更多；其差异不能单独归因于自注意力。树与线性头参数量不按神经权重数量直接类比。', '',
        '## 结论边界','',
        '三种线性头（逻辑回归、岭回归、线性SVM）及MLP在两批历史数据上，相对B的主pAUC修正差区间分别为正；这支持固定F下学习判分规则的历史验证价值，但仍不等于独立新确认。', '',
        '复杂结构尚未稳定超过简单线性头：注意力汇总对逻辑回归的pAUC差，C和RV两批描述性95%区间均含0；Transformer对注意力汇总在C未获额外收益支持，在RV描述性区间为负。MLP及两种树在RV的pAUC点值低于逻辑回归，相关机制比较只作描述性判断。此结论限于本次固定小规模结构和参数预算，不否定整类方法。', '',
        '注意力汇总在RV的误报6.00%降至4.67%、检出31.46%升至47.31%；但在C误报5.40%升至6.80%，不能声称误报改善跨批稳定。优先将线性头作为后续新来源验证基线，注意力汇总保留为候选；不按历史最高点值直接选定部署赢家。', '',
        '本轮完成固定F输入下的八个检测头验证；是否比简单线性头更有价值，需同时看两批低误报指标、直接机制对照、误报和种子波动，而非只挑最高AUROC。历史诊断不能证明跨模型或跨来源泛化；不因本轮点值上升自动替换原积分。后续独立验证需要新来源/新批次，不能把当前已评分记录重复当作新确认。', '',
        '## 复核入口','',
        '- [冻结协议](../../docs/oc-heads-protocol-2026-11-04.md)、[源码封印](source-lock.json)、[数据封印](manifest.json)。',
        '- [全部选择记录](search.json)、[测试前封印](pre-evaluation-seal.json)、[选定头](selection.json)。',
        '- [历史C完整统计](evaluation-C.json)、[历史RV完整统计](evaluation-RV.json)、[逐条三种子分数](scores.jsonl)。',
        '- [模型与分数复核](verification.json)、[补充来源与dev复核](verification-additional.json)、[训练F鲜提取](identity-training.json)、[无新确认说明](confirmation-unavailable.json)。',
        '', '源码与数据在训练前封印，selection在测试前封印。结果文件不含原始提示或SSH凭据。']
    if (OUT/'verification-statistics-independent.json').exists():
        independent=get('verification-statistics-independent.json');assert independent['status']=='passed'
        lines+=['','## 独立统计复核','',f"另一套按并列分数分档的ROC实现重做两批各10000次bootstrap，复核{independent['point_metrics_checked']}组点指标、{independent['bootstrap_intervals_checked']}个主区间；点指标最大差{independent['maximum_point_metric_difference']}，区间最大差{independent['maximum_interval_difference']}，阈值差0且混淆条数精确一致。逐条分数始终在学校机器内读取，仅交付[汇总审计](verification-statistics-independent.json)。"]
    (OUT/'report-2026-11-04.md').write_bytes(('\n'.join(lines)+'\n').encode())
    print('OC4_REPORT_WRITTEN')
if __name__=='__main__':main()
