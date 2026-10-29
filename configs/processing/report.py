"""Reproducible full-family result tables, no test-based winner selection."""
import json
from processing_core import ROOT

def main():
    out=ROOT/'results/processing';f=json.loads((out/'freeze.json').read_text());e=json.loads((out/'evaluation.json').read_text());seal=json.loads((ROOT/'logs/processing-cache-seal.json').read_text())
    lines=['# R19—R22：用户信息加工实验交接','',
        '固定基准v1与train1600/dev600不变；新的Aegis1000确认排除所有旧输入及近重复，仍是同源确认，不是新领域。所有候选在评分前提交冻结。','',
        '## 这次加工什么','',
        '第8、16、24、32层的全部有效用户token，保留完整4096维方向，不作随机投影。普通均值、末位对用户的注意力加权、相对残差变化加权、变化加权的单位跨层增量、Attention模块增量、MLP模块增量，以及确定性随机权重对照，共7类摘要。局部—整体关系共116项，包括余弦、距离、范数、权重熵和模块方向关系。',
        '跨层／模块增量是对内部向量的观察，不是因果归因；没有训练底座、没有生成回复。固定512、有正确mask，排除前缀／后缀／补齐token。','',
        '## 匹配控制与训练','',
        '- R19 mean：末位完整向量＋普通用户均值。',
        '- R20 dynamic：注意力加权＋变化加权的用户状态。',
        '- R21 processed：dynamic再加入跨层／Attention／MLP增量。',
        '- R22 relations：processed再加入局部—整体关系。',
        '- 随机权重对照random；重复5份普通均值duplicate_mean，归一化后应与mean数学等价。',
        '全部使用同一lambda1中心化岭回归。G=G_B+gamma G_E，gamma只取预设.25和1；关系块固定权重.1。13个候选、重复5折两次，共130折；关系标准化仅拟合训练折。开发集仅诊断。',
        f"最终CV选择 `{f['selection']['selected']}`，原始CV最佳 `{f['selection']['raw_best']}`；一SE规则是启发式，不是正式置信区间。",
        f"新缓存{seal['additional_bytes']}字节；旧末位读出最大差{seal['state']['maximum_old_B_difference']}；固定前缀差{seal['state']['prefix_difference']}。",'']
    for condition in ('test_aegis_holdout:global','confirmation_aegis:global'):
        lines+=['## '+condition,'','|方案|AUROC|pAUC(.1)|召回|误报|','|---|---:|---:|---:|---:|']
        for c in f['candidates']:
            v=e['tests'][condition][c['name']];lines.append(f"|{c['name']}|{v['auroc']:.6f}|{v['partial_auroc']:.6f}|{v['tpr']:.1%}|{v['fpr']:.1%}|")
        lines+=['']
    lines+=['## 新确认：预注册十个对照','',
        '以下为gamma1；gamma.25及固定旧基准比较为描述性。5000次分层配对bootstrap，按十个对照作每种指标内Bonferroni修正，AUROC与pAUC分开解释；条件于固定训练和校准，不涵盖完整重训不确定性。','',
        '|对照|ΔAUROC|修正区间|ΔpAUC|修正区间|','|---|---:|---|---:|---|']
    for name,v in e['contrasts']['confirmation_aegis:global'].items():
        a=v['auroc'];p=v['partial_auroc'];lines.append(f"|{name}|{a['delta']:+.6f}|{a['family10_ci']}|{p['delta']:+.6f}|{p['family10_ci']}|")
    lines+=['','## 证据边界','',
        '比较加工方法是否增加信息，需同时看相对B、普通均值和随机权重对照；不能把候选中的测试最高值当成预选赢家。维度／正则化也会影响结果；重复均值只是维度冗余对照，并未完全匹配所有统计容量。',
        '这里只检查四个固定层的指定汇总，不能否定其他层、学习型聚合或用户片段方案。注意力权重不自动代表token的因果重要性。','',
        f"旧固定末位分数最大差{e['maximum_old_B_score_difference']}，重复均值分数最大差{e['maximum_duplicate_mean_score_difference']}。",
        '完整候选、各来源和召回／误报区间见evaluation.json；逐样本归档heldout-scores.jsonl；训练折与选择见cv.json/oof.npz/selection.json。',
        '复核：`python configs/processing/verify.py --bootstrap`；服务器加`--actual`校验私有缓存。','']
    (out/'report-2026-10-29.md').write_text('\n'.join(lines),encoding='utf8');print('REPORT_WRITTEN')

if __name__=='__main__':main()
