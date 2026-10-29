"""Render the frozen comparison without hand-selecting favorable seeds."""
import json
import numpy as np
from routing_core import ROOT,write

def main():
    out=ROOT/'results/routing';f=json.loads((out/'freeze.json').read_text());e=json.loads((out/'evaluation.json').read_text());cv=json.loads((out/'cv.json').read_text());seal=json.loads((ROOT/'logs/routing-cache-seal.json').read_text());s=f['selection']
    lines=['# R16–R18：数值形状控制、模板邻位与逐层位置选择','',
        '本轮从上一轮“整体换位置没有稳定收益”继续：检查各层采用不同位置，是否有同维度的信息增益。',
        '训练1600、开发600、固定v1比较基准2760行不变；另取此前未使用的同源Aegis1000行确认。它不是新领域测试。','',
        '## 设计与防泄漏','',
        '- R16：变长/固定512执行，内容与有效token一致、右补齐并传attention mask。相同lambda1岭回归；新头统一float64，原float32 B单列。',
        '- R17：固定512读取模板最后、倒数第二、倒数第三个token，均取32层×4096完整坐标。',
        '- R18：32位掩码逐层选用户末位或模板末位；始终131072维。没有拼接增维、没有投影，也没有训练基础模型。',
        '- train-only两次5折CV；目标为两种执行方式中较差的(mean pAUC−0.25 SD)，减0.002×替换比例。dev仅诊断。',
        '- random/BPSO/GA/DE/scan各5种子×64个唯一掩码，先共享并计费34个基准/单层方案，再30个。scan展开最佳单层的30个双层组合。',
        '- 最终选择只用CV，一SE内优先全模板；这个重叠折SE是启发式，不是正式置信区间。所有算法种子均报告。',
        '- 模型、源代码、样本和缓存先封存并提交，再统一打分；安全校准400条的第380个排序值，严格大于阈值报警。','',
        f"共计费{f['charged_evaluations']}次、跨运行{f['unique_cv_masks']}个不同掩码。最终CV选择：`{s['selected_label']}`；原始最佳掩码 `{s['raw_best_mask']}`。",
        f"固定512全部{seal['rows']}行的受保护前缀最大差异={seal['state']['prefix_max_difference']}；固定/变长读出最大坐标差异={seal['state']['max_fixed_variable_difference']:.9g}。",
        '前缀在固定形状下相同，支持上轮的浮点执行形状解释；不能据此声称所有数值误差已经消除。','',
        '## 逐项结果','']
    selected=['legacy_B','variable_template','fixed_template','variable_all_user','fixed_all_user','fixed_suffix_minus1','fixed_suffix_minus2','variable_single','fixed_single']
    selected+=['variable_'+s['selected_label'],'fixed_'+s['selected_label']];selected=list(dict.fromkeys(selected))
    for condition in ('test_aegis_holdout:global','confirmation_aegis:global'):
        lines += ['### '+condition,'','|方案|AUROC|pAUC(.1)|召回|误报|','|---|---:|---:|---:|---:|']
        table=e['tests'][condition]
        for name in selected:
            v=table[name];lines.append(f"|{name}|{v['auroc']:.6f}|{v['partial_auroc']:.6f}|{v['tpr']:.1%}|{v['fpr']:.1%}|")
        lines+=['','|算法（5种子均值）|变长AUROC|固定AUROC|固定pAUC|固定召回|固定误报|','|---|---:|---:|---:|---:|---:|']
        for method in ('random','bpso','ga','de','scan'):
            a=[table[f'variable_{method}_{seed}'] for seed in range(20261201,20261206)];b=[table[f'fixed_{method}_{seed}'] for seed in range(20261201,20261206)]
            avg=lambda seq,k:np.mean([v[k] for v in seq])
            lines.append(f"|{method}|{avg(a,'auroc'):.6f}|{avg(b,'auroc'):.6f}|{avg(b,'partial_auroc'):.6f}|{avg(b,'tpr'):.1%}|{avg(b,'fpr'):.1%}|")
        lines+=['']
    lines+=['## 新1000条的预注册对照','',
        '5000次分层配对bootstrap；以下AUROC/pAUC区间按8个对照作Bonferroni修正。算法比较是5种子指标差的均值，不是分数集成；区间条件于现有训练与校准样本，未涵盖重训不确定性。','',
        '|对照|ΔAUROC|修正区间|ΔpAUC|修正区间|','|---|---:|---|---:|---|']
    contrasts=e['contrasts']['confirmation_aegis:global']
    for name,v in contrasts.items():
        a=v['auroc'];p=v['partial_auroc'];lines.append(f"|{name}|{a['delta']:+.6f}|{a['family8_ci']}|{p['delta']:+.6f}|{p['family8_ci']}|")
    positive=[n for n,v in contrasts.items() if v['auroc']['family8_ci'][0]>0]
    negative=[n for n,v in contrasts.items() if v['auroc']['family8_ci'][1]<0]
    lines+=['','## 结论与边界','',f"按预注册AUROC规则，有证据的正向对照：{positive or '无'}；负向对照：{negative or '无'}。其余对照尚不能确认方向。",
        '需要同时看pAUC和校准阈值下召回/误报；单次最高分不能证明算法优越。',
        '本轮只研究用户末位与模板末位之间的逐层切换及两个模板邻位，没有穷举所有token、动态位置或注意力/MLP分量，不能证明全局最优。',
        '下一步仅在出现可重复信息增益时扩大位置空间；若选不出稳定替换层，应优先跨数据源验证、训练样本重复抽样和更严格的误报控制，而非继续增大搜索空间。','',
        '## 实际选中的层（1起算）','', '|方法|种子|替换为用户末位的层|CV目标|','|---|---|---|---:|']
    for r in cv['runs']:
        layers=[i+1 for i,b in enumerate(r['best_mask']) if b];lines.append(f"|{r['method']}|{r['seed']}|{layers}|{r['best_fitness']:.6f}|")
    lines+=['','完整各源结果、全部种子与区间：evaluation.json；逐样本分数：heldout-scores.jsonl；各掩码训练CV：cv.json/oof.npz。',
        '复核：`python configs/routing/verify.py --bootstrap`；计算服务器另加`--actual`校验私有缓存字节。','']
    (out/'report-2026-10-29.md').write_text('\n'.join(lines),encoding='utf8')
    print('REPORT_WRITTEN',out/'report-2026-10-29.md')

if __name__=='__main__':main()
