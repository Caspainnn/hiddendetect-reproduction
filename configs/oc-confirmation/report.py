import json
from core import OUT,NAMES,write
LABELS=['原积分B','逻辑回归','线性SVM','注意力汇总']
def main():
    result=json.loads((OUT/'confirmation.json').read_text());stability=json.loads((OUT/'stability.json').read_text());verification=json.loads((OUT/'verification.json').read_text());assert verification['status']=='passed'
    summary=json.loads((OUT/'dedup-summary.json').read_text())
    lines=['# OC-H2：固定检测头的OR-Bench跨来源确认与稳健性','',
        '日期2026-11-04。固定原积分、逻辑回归、线性SVM、注意力汇总；阶段一不重训或调参，主种子20261104。新来源、模型、样本与协议在第一次新样本评分前冻结。阶段二只作已观测测试上的训练/校准稳健性诊断，不能替换阶段一模型。','',
        f"测试正常{result['counts'][0]}、危险{result['counts'][1]}，另400正常用于新来源阈值校准。历史去重输入{summary['historical_inputs']}；排除规范化与char3-5 TFIDF cosine≥.92近重复，完整审计保留学校。",'',
        '来源为作者官方OR-Bench：80k子集映射正常，toxic子集映射危险，category仅为话题不作为危险标签。版本、CC-BY-4.0许可及原文件哈希见acquisition.json。正常样本特意包含易被拒绝的内容，标签来自作者生成/筛选；与Aegis分布和标注方式不同，不等于广泛跨领域证明，也不保证基础模型预训练未见过。','',
        '来源：Justin Cui、Wei-Lin Chiang、Ion Stoica、Cho-Jui Hsieh，OR-Bench: An Over-Refusal Benchmark for Large Language Models（2024）。[官方数据](https://huggingface.co/datasets/bench-llm/or-bench)、[论文](https://arxiv.org/abs/2405.20947)。','',
        '## 阶段一：固定主种子','', '|方法|AUROC|低误报pAUC|旧阈值检出率|旧阈值误报率（条数）|新校准检出率|新校准误报率（条数）|','|---|---|---|---|---|---|---|']
    base=result['metrics'][0]
    for j,m in enumerate(result['metrics']):
        def val(value,before,percent=False,lower=False):
            text=f'{value:.2%}' if percent else f'{value:.6f}'
            better=value<before if lower else value>before
            return '**'+text+'**' if j and better else text
        def rate(kind,key):return val(m[kind][key],base[kind][key],True,key=='fpr')
        lines.append(f"|{LABELS[j]}|{val(m['auroc'],base['auroc'])}|{val(m['pauc'],base['pauc'])}|{rate('old','tpr')}|{rate('old','fpr')}（{m['old']['fpr_count']}/{m['old']['fpr_n']}）|{rate('new','tpr')}|{rate('new','fpr')}（{m['new']['fpr_count']}/{m['new']['fpr_n']}）|")
    lines+=['','旧阈值来自OC-H1的400正常；新阈值只用独立OR-Bench400正常。新校准不改变AUROC或pAUC。两个工作点同时保留，不按测试误报率反调。','',
        '|主比较|pAUC差|六比较修正99.166667%差区间|本来源解释|','|---|---|---|---|']
    for item in result['comparisons']:
        lo,hi=item['pauc_delta_ci991667'];decision={'new_source_positive':'得到本次新来源支持','worse':'本来源更差','unconfirmed':'改善未获区间支持'}[item['decision']]
        lines.append(f"|{item['left']} 对 {item['right']}|{item['delta_pauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|{decision}|")
    lines+=['','## 阶段二：训练重采样','', '旧1600标签池、每次800+800有放回抽样，5次相同索引配对三种头、参数保持OC-H1所选，总15拟合。下列范围不是置信区间，也不是集成成绩。','', '|头|五次AUROC范围|五次pAUC范围|新校准检出率范围|新校准误报率范围|','|---|---|---|---|---|']
    for name in NAMES[1:]:
        runs=[r for r in stability['runs'] if r['id']==name]
        def span(key,kind=None,percent=False):
            values=[r[kind][key] if kind else r[key] for r in runs];a,b=min(values),max(values)
            return f'{a:.2%}～{b:.2%}' if percent else f'{a:.6f}～{b:.6f}'
        lines.append(f"|{name}|{span('auroc')}|{span('pauc')}|{span('tpr','new',True)}|{span('fpr','new',True)}|")
    lines+=['','## 固定头的校准波动','', '旧与新400正常分别1000次重采样，共用抽样索引；报告经验2.5%～97.5%范围，只描述固定测试上的校准波动。','', '|校准源|方法|检出率经验范围|误报率经验范围|','|---|---|---|---|']
    for kind,values in stability['calibration_variability'].items():
        for m in values:
            t=m['tpr_empirical95_range'];f=m['fpr_empirical95_range'];lines.append(f"|{kind}|{m['id']}|{t[0]:.2%}～{t[1]:.2%}|{f[0]:.2%}～{f[1]:.2%}|")
    lines+=['','## 机制研究与复核','',
        f"注意力对两个线性头的新来源支持门槛：{stability['mechanism_new_source_support_gate']}；5次重采样中高于同次两线性头{stability['attention_robust_wins_out_of_5']}次。下一轮机制设计进入条件为{stability['mechanism_entry_gate']}；本轮没有执行新结构或在新测试上调参。",'',
        f"重载复核{verification['frozen_score_values_replayed']}个固定模型分数，最大差{verification['maximum_frozen_score_difference']}；独立分档ROC复算10000次bootstrap、12个区间，最大差{verification['maximum_interval_difference']}，混淆条数精确一致。",'',
        '本地只交付汇总报告；原文、F、逐条分数及模型留校。新测试从本轮评分后转为历史，后续机制实验不能重复称新独立确认。', '',
        '入口：[阶段一完整统计](confirmation.json)、[稳健性汇总](stability.json)、[复核](verification.json)、[去重计数](dedup-summary.json)、[来源版本](acquisition.json)、[冻结协议](../../docs/oc-heads-confirmation-protocol-2026-11-04.md)。']
    (OUT/'report-2026-11-04.md').write_bytes(('\n'.join(lines)+'\n').encode());print('OC_H2_REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()
