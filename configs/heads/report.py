"""Render all preselected head winners without test-based winner picking."""
import json
from head_core import ROOT

def main():
    out=ROOT/'results/heads';f=json.loads((out/'freeze.json').read_text());e=json.loads((out/'evaluation.json').read_text());cv=json.loads((out/'cv.json').read_text());seal=json.loads((ROOT/'logs/heads-cache-seal.json').read_text())
    lines=['# R23—R26：相同完整表示上的检测头替代验证','',
        '固定完整32层模板末位131072维，不改变位置、不投影、不训练底座。相同train1600/dev600、固定基准v1，另取1000条此前未使用的同源Aegis确认。','',
        '## 为什么选这三类替代','',
        '- R23调参岭回归：先给原方法公平调参，避免只优化替代方法。',
        '- R24 L2逻辑回归：保留线性形式，改为二分类损失。',
        '- R25线性SVM：标准hinge间隔目标，线性核SVC，不是非线性SVM。',
        '- R26 RBF核岭回归：检验样本间非线性相似度是否有用。',
        '各9组参数、相同两次五折、训练CV pAUC选择，每类预算相同但运行代价不同。原lambda1单独保留。整体CV选择一SE内优先调参岭回归。开发集仅诊断，不挑最终测试最高头。','',
        '## 数学与数据控制','',
        '逻辑回归使用训练样本张成空间的完整特征重参数化，保留全部非数值零特征值；不是64维投影。训练折独立中心化和分解；RBF宽度只由拟合折距离中位数确定。线性头最终转回131072坐标系。',
        f"CV最大相对Gram重构误差{max(r['relative_gram_error'] for r in cv['span_checks']):.3g}；全训练张成空间秩{cv['full_span']['rank']}。",
        f"新特征缓存{seal['bytes']}字节；32条旧训练探针最大读出差{seal['pilot']['maximum_old_B_difference']}，公共前缀差{seal['state']['prefix_difference']}。",
        f"CV最终选择 `{f['selection']['selected_head']}`，原始最佳参数 `{f['selection']['raw_best']}`。一SE是重叠折启发式，不是正式置信区间。",'',
        '|最终头|训练侧选择参数|CV pAUC|开发AUROC（仅诊断）|','|---|---|---:|---:|']
    for c in f['candidates']:lines.append(f"|{c['name']}|{c['parameters']}|{c['cv']['cv_mean']:.6f}|{c['dev_auroc']:.6f}|")
    for condition in ('test_aegis_holdout:global','confirmation_aegis:global'):
        lines+=['','## '+condition,'','|方案|AUROC|pAUC(.1)|召回|误报|','|---|---:|---:|---:|---:|']
        for c in f['candidates']:
            v=e['tests'][condition][c['name']];lines.append(f"|{c['name']}|{v['auroc']:.6f}|{v['partial_auroc']:.6f}|{v['tpr']:.1%}|{v['fpr']:.1%}|")
    lines+=['','## 新确认五个预注册对照','',
        '5000次分层配对bootstrap，按5对照作每种指标内Bonferroni校正；条件于当前训练和校准。AUROC和pAUC分开解释，不宣称同时控制全部指标。','',
        '|对照|ΔAUROC|修正区间|ΔpAUC|修正区间|','|---|---:|---|---:|---|']
    for n,v in e['contrasts']['confirmation_aegis:global'].items():
        a=v['auroc'];p=v['partial_auroc'];lines.append(f"|{n}|{a['delta']:+.6f}|{a['family5_ci']}|{p['delta']:+.6f}|{p['family5_ci']}|")
    lines+=['','## 边界与复核','',
        '新确认为同源，固定旧基准为描述性比较；Toxic任务为毒性辅助任务，不混合算平均。保持原始标签，未根据上一轮误报重新调方法。',
        '只验证三个明确实现与有限网格，不能证明整个模型类别最优。相同调参候选数不等于相同CPU/GPU时间，RBF推理需读取封存的训练参考表示。',
        f"原B跨轮分数最大差{e['maximum_old_B_score_difference']:.3g}。全部36设置、360折指标、35测试条件、逐样本分数和bootstrap可重放。",
        '复核：`python configs/heads/verify.py --bootstrap`；服务器加`--actual`校验私有缓存。完整源任务、标签、阈值与各来源结果见evaluation.json。','']
    (out/'report-2026-10-29.md').write_text('\n'.join(lines),encoding='utf8');print('REPORT_WRITTEN')

if __name__=='__main__':main()
