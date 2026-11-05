"""Build OC question chains from the registered methods and saved results."""
import json
import hashlib
from copy import deepcopy
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT / 'experiments/HiddenDetect'


def node(label, children=None, **kwargs):
    return dict(label=label, children=children or [], **kwargs)


def confirmation_evidence(ident):
    path=REPO/'results/oc-confirmation/confirmation.json'
    if not path.exists():return []
    result=json.loads(path.read_text(encoding='utf-8'))
    if ident not in result['names']:return []
    metrics={m['id']:m for m in result['metrics']};m,b=metrics[ident],metrics['B']
    leaves=[node(f"结果（新来源OR-Bench）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}",kind='result')]
    for kind,label in [('old','沿用旧阈值'),('new','新来源400正常校准')]:
        a,z=b[kind],m[kind]
        leaves.append(node(f"{label}：检出率 {a['tpr']:.2%} → {z['tpr']:.2%}；误报率 {a['fpr']:.2%} → {z['fpr']:.2%}（{z['fpr_count']}/{z['fpr_n']}）",kind='result'))
    comparison=next(c for c in result['comparisons'] if c['left']==ident and c['right']=='B')
    lo,hi=comparison['pauc_delta_ci991667']
    decision={'new_source_positive':'本次新来源相对B获区间支持','worse':'本次新来源相对B更差','unconfirmed':'本次新来源改善未获区间支持'}[comparison['decision']]
    leaves.append(node(f"证据：{result['counts'][0]}正常/{result['counts'][1]}危险，主种子{result['primary_seed']}；六比较99.166667%主差区间[{lo:+.6f},{hi:+.6f}]；{decision}。OR-Bench为生成并筛选的过度拒绝基准，不代表广泛泛化。",kind='evidence'))
    stability=REPO/'results/oc-confirmation/stability.json'
    if stability.exists():
        stats=json.loads(stability.read_text(encoding='utf-8'));runs=[r for r in stats['runs'] if r['id']==ident]
        if runs:
            values=[r['pauc'] for r in runs]
            leaves.append(node(f"稳健性：固定参数、旧1600标签池重采样5次，pAUC范围{min(values):.6f}～{max(values):.6f}；这是已观测测试上的稳定性诊断，不是另一批新确认。",kind='evidence'))
    return [node('OC-H2：跨来源确认与稳健性',leaves,id=ident.lower()+'-h2',kind='direction')]


def retraining_evidence(ident):
    path=REPO/'results/oc-retraining/retraining.json'
    if not path.exists():return []
    result=json.loads(path.read_text(encoding='utf-8'))
    metrics={m['id']:m for m in result['metrics']}
    if 'or_small:'+ident not in metrics:return []
    summary=json.loads((REPO/'results/oc-retraining/data-summary.json').read_text(encoding='utf-8'))
    new=metrics['or_small:'+ident]
    leaves=[node(f"方法：原14层F不变，旧完整1600训练头保留；旧来源和OR来源各取{summary['new_training_count']}条分别从头训练，同参数、三种子；同一400正常校准、1000正常/500危险测试。",kind='method')]
    for group,label in [('old_full','旧完整头 → OR重训头'),('old_small','同规模旧源重训 → OR重训')]:
        old=metrics[group+':'+ident]
        leaves.append(node(f"结果（{label}）：AUROC {old['auroc']:.6f} → {new['auroc']:.6f}；低误报pAUC {old['pauc']:.6f} → {new['pauc']:.6f}\n检出率 {old['tpr']:.2%} → {new['tpr']:.2%}；误报率 {old['fpr']:.2%} → {new['fpr']:.2%}（{new['fpr_count']}/{new['fpr_n']}）",kind='result'))
        comparison=next(c for c in result['comparisons'] if c['left']=='or_small:'+ident and c['right']==group+':'+ident)
        lo,hi=comparison['pauc_delta_ci994444']
        decision={'positive':'本次历史诊断支持提高','worse':'本次历史诊断支持下降','unconfirmed':'本次改善尚未获区间支持'}[comparison['decision']]
        leaves.append(node(f"证据（{label}）：pAUC差{comparison['delta_pauc']:+.6f}，九比较99.444444%区间[{lo:+.6f},{hi:+.6f}]；{decision}。",kind='evidence'))
    leaves.append(node('边界：测试在H2已观察，本轮是历史重训诊断；完整旧头与OR重训同时有来源、样本量及部分训练步数差异；同规模旧源对照用于拆分。三种子全量结果见OC-H3报告，不挑最好种子。',kind='evidence'))
    experiment=node('OC-H3 / '+ident+'：三组重训对照',leaves,id=ident.lower()+'-h3-run',kind='experiment')
    question=node('换成新来源的数据重新训练，能比直接沿用旧头更好吗？',[experiment],id=ident.lower()+'-h3-question',kind='question')
    return [node('OC-H3：重新训练检测头',[question],id=ident.lower()+'-h3',kind='direction')]


def svm_results():
    path=REPO/'results/oc-svm/evaluation.json'
    verification=REPO/'results/oc-svm/verification.json'
    if not path.exists() or not verification.exists():return None
    checked=json.loads(verification.read_text(encoding='utf-8'))
    assert checked['passed'] and checked['evaluation_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def svm_baseline_evidence(ident):
    result=svm_results()
    if ident!='OC-4-1-3' or not result:return []
    metrics={m['id']:m for m in result['metrics']};m=metrics[ident];b=metrics['B'];old=metrics['H3-C10']
    return [node('OC-S1：SVM改进的同批基线',[
        node('方法：OR来源300训练，固定5折训练侧选型；基线与各候选均9组参数，标准化在折内拟合；400正常校准、1000正常/500危险历史测试。',kind='method'),
        node(f"结果（原积分B → 本轮调参线性SVM C=1）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）",kind='result'),
        node(f"复核：H3固定C10逐条分数复现，AUROC {old['auroc']:.6f}、pAUC {old['pauc']:.6f}、检出{old['tp']}/500、误报{old['fp']}/1000。C1由训练侧选择，不按测试挑选。",kind='evidence'),
        node('结论：三项改进均未确认优于同批调参线性基线；RBF两项主pAUC差区间全负。242拟合、17模型重载、8万次独立ROC及全部区间复核。历史诊断，无独立新确认。',kind='evidence')],id='oc-s1-baseline',kind='direction')]


def attention_results():
    path=REPO/'results/oc-attention/evaluation.json';check=REPO/'results/oc-attention/verification.json'
    if not path.exists() or not check.exists():return None
    verified=json.loads(check.read_text(encoding='utf-8'))
    assert verified['passed'] and verified['evaluation_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def original_oc4_evidence(ident):
    path=REPO/'results/oc4-original/evaluation.json';check=path.with_name('verification.json')
    if not path.exists() or not check.exists():return []
    verified=json.loads(check.read_text(encoding='utf-8'))
    assert verified['passed'] and verified['evaluation_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    result=json.loads(path.read_text(encoding='utf-8'))
    candidates=['OC-4-1-3-1','OC-4-1-3-2','OC-4-1-3-3','OC-4-4-1-1','OC-4-4-1-2','OC-4-4-1-3','OC-4-4-2-1','OC-4-2-1-1']
    if ident not in candidates+['OC-4-1-3','OC-4-4-1','OC-4-4-2','OC-4-2-1']:return []
    direct={'OC-4-1-3-1':'OC-4-1-3','OC-4-1-3-2':'OC-4-1-3','OC-4-1-3-3':'OC-4-1-3','OC-4-4-1-1':'OC-4-4-1','OC-4-4-1-2':'OC-4-4-1','OC-4-4-1-3':'OC-4-4-1','OC-4-4-2-1':'OC-4-4-2','OC-4-2-1-1':'linear-encoded-mlp'}
    leaves=[node('方法：回到原OC-4的Aegis训练1600、选择600、正常校准400；RV1082为主历史测试（原积分AUROC0.756547），C1000为补充。两组重新拟合；固定原14F。原H1旧头与本轮匹配程序基线均保留。',kind='method')]
    for cohort,label in [('RV','原OC-4主历史批次RV1082'),('C','补充历史批次C1000')]:
        rows={m['id']:m for m in result['cohorts'][cohort]['metrics']};m=rows[ident]
        for base,title in [('B','原积分B')]+([(direct[ident],'本轮直接对照'+direct[ident]),('H1:'+ident.rsplit('-',1)[0],'H1原检测头')] if ident in direct else [('H1:'+ident,'H1原检测头')]):
            b=rows[base]
            leaves.append(node(f"结果（{label}；{title} → 候选）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）",kind='result'))
    if ident in candidates:
        for suffix,label in [('-vs-B','候选−原积分'),('-vs-direct','候选−本轮直接对照')]:
            c=next(c for c in result['cohorts']['RV']['comparisons'] if c['id']==ident+suffix);lo,hi=c['pauc_delta_ci'];decision={'positive':'本历史批支持提高','worse':'本历史批支持下降','unconfirmed':'改善未确认'}[c['decision']]
            leaves.append(node(f"证据（RV；{label}）：pAUC差{c['delta_pauc']:+.6f}；27比较99.814815%区间[{lo:+.6f},{hi:+.6f}]，{decision}。",kind='evidence'))
    leaves.append(node('边界：144拟合、38新选定模型；H1旧头全部种子分数复现。原神经头为小批训练，本轮统一240全批更新，不将程序变化归因结构；SVM9候选、神经3候选，预算不同。RV主家族27项，C区间仅描述性；两批均历史，无独立新确认。',kind='evidence'))
    return [node('回到原OC-4数据，变式还有收益吗？',[node('OC-O1：原OC-4划分验证',leaves,id=ident.lower()+'-o1-run',kind='experiment')],id=ident.lower()+'-o1-question',kind='question')]


def attention_baseline_evidence(ident):
    result=attention_results()
    if ident not in ['OC-4-4-1','OC-4-4-2','OC-4-2-1'] or not result:return []
    metrics={m['id']:m for m in result['metrics']};m=metrics[ident];b=metrics['B'];svm=metrics['SVM']
    leaves=[node('方法：本轮训练300，5折只选择学习率；每头240次全批更新，3种子；原14F、400正常校准及1000正常/500危险历史测试。新结构的直接对照使用本轮重训模型。',kind='method'),
        node(f"结果（原积分B → 本轮重训头）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）",kind='result'),
        node(f"参照（OC-S1线性SVM → 本轮重训头）：AUROC {svm['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {svm['pauc']:.6f} → {m['pauc']:.6f}\n检出率 {svm['tpr']:.2%} → {m['tpr']:.2%}；误报率 {svm['fpr']:.2%} → {m['fpr']:.2%}（基线{svm['fp']}条；候选{m['fp']}条）",kind='result'),
        node('边界：原H3程序逐参数及逐分数复现，本轮统一全批训练程序与旧小批不同；不将训练程序变化归因新结构。SVM先前9候选、神经头本轮3候选，跨家族预算不同。历史诊断，无独立新确认。',kind='evidence')]
    if ident=='OC-4-4-1':
        c=next(c for c in result['comparisons'] if c['id']=='attention-vs-SVM');lo,hi=c['pauc_delta_ci']
        leaves.append(node(f"证据（原注意力−SVM）：pAUC差{c['delta_pauc']:+.6f}；15比较99.666667%区间[{lo:+.6f},{hi:+.6f}]。",kind='evidence'))
    return [node('OC-A2：本轮匹配训练基线',leaves,id=ident.lower()+'-a2-baseline',kind='direction')]


def build_oc():
    reference = json.loads((REPO / 'results/oc-reference/confirmation-evaluation.json').read_text(encoding='utf-8'))
    scoring = json.loads((REPO / 'results/oc23/historical-RV-evaluation.json').read_text(encoding='utf-8'))
    metrics = {m['direction_id']: m for d in (reference, scoring) for m in d['metrics']}
    groups = [
        node('OC-1：拒绝词参照：拿什么代表“拒绝”？', id='oc1'),
        node('OC-2：单层评分：怎样把词表结果变成一个分数？', id='oc2'),
        node('OC-3：跨层汇总：怎样把各层分数合起来？', id='oc3'),
        node('OC-4：不同检测头判分：固定14层F，换个头会更好吗？', id='oc4'),
    ]
    definitions = [
        ('OC-1-1', '有些字符串的首token，会不会不能清楚代表拒绝？', '清理参照', '预先清理含义过泛的拒绝token，保留10个原坐标；保持原余弦与积分', 'AUROC提高，但低误报pAUC改善未确认；检出率下降'),
        ('OC-1-2', '原来的拒绝表达，会不会覆盖不全？', '扩充参照', '预先在原集合加入10个拒绝表达，去重后29个坐标；保持原余弦与积分', '本固定扩充方案更差，不代表所有扩充方法都无效'),
        ('OC-1-3', '所有拒绝token，都应该一样重要吗？', '参照加权', '保持原20个坐标，5个拒绝语气词权重为2，其余为1；保持原余弦与积分', '本固定加权方案更差，保留原等权参照'),
        ('OC-2-1', '直接看拒绝token的平均分，会不会更清楚？', '拒绝logit均值', '每层取原20个拒绝logits的均值，替代余弦，再按原14层梯形积分', '两批历史诊断均未确认低误报改善；尚无独立新确认'),
        ('OC-2-2', '某一种拒绝表达很强，是否就值得重视？', '平滑最大值', '每层对原拒绝logits计算tau=1的log-mean-exp，再按原14层梯形积分；不搜索温度', '两批历史诊断的低误报pAUC均更差；尚无独立新确认'),
        ('OC-2-3', '拒绝表达在整个词表里，占了多大概率？', '拒绝概率总和', '每层完整词表做softmax，累加原拒绝token概率，再按原14层梯形积分', '两批历史诊断的低误报pAUC均更差；尚无独立新确认'),
    ]
    # OC-3 uses the existing question chain; OC-4 follows its new three-level plan.
    text = (REPO / 'research-map.md').read_text(encoding='utf-8')
    text = text[text.index('      - OC-3：跨层汇总'):text.index('      - OC-4：')]
    current = None
    question = None
    for line in text.splitlines():
        m = re.match(r'^(\s*)- (.*)$', line)
        if not m:
            continue
        depth, label = len(m[1]), m[2]
        if depth == 8:
            question = label
        elif depth == 10:
            ident, title = label.split('：', 1)
            current = [ident, question, title, '', '']
            definitions.append(current)
        elif depth == 12 and current:
            if label.startswith('方法：'):
                current[3] = label.removeprefix('方法：')
            elif label.startswith(('证据与结论：', '历史线索：', '归因边界：')):
                current[4] += ('；' if current[4] else '') + label.split('：', 1)[1]
    rows = []
    for ident, question, title, method, conclusion in definitions:
        group = int(ident.split('-')[1])
        topic = ['reference', 'scoring', 'aggregation', 'heads'][group - 1]
        if ident in metrics:
            m = metrics[ident]
            result = (f"结果：AUROC 0.756547 → {m['auroc']:.6f}；低误报pAUC 0.612375 → {m['pauc']:.6f}\n"
                      f"检出率 31.46% → {m['tpr']:.2%}；误报率 6.00% → {m['fpr']:.2%}（{m['fpr_count']}/300）")
            batch = '同批RV1082：300正常、782危险。' + ('OC-1执行时为新确认。' if group == 1 else '本方法在此批为历史诊断。')
            if group == 3:
                # Preserve the prior confirmation separately from the shared historical batch.
                batch += '此前独立2000条证据见结论。' if ident != 'OC-3-5' else ''
            source = 'results/oc-reference/report-2026-10-30.md' if group == 1 else 'results/oc23/report-2026-10-30.md'
            status = '已验证' if group == 1 else '历史诊断'
        else:
            result = '结果：待验证，暂无严格直接对照数值。'
            batch = '固定14层的检测头实验尚未执行；训练标签预算须单记。'
            source = 'docs/oc-cluster-plan-2026-10-30.md'
            status = '待验证'
        if not conclusion:
            conclusion = '待按固定输入、同标签预算及独立评估进行验证。'
        experiment = node(f'{ident}：{title}', [node('方法：' + method, kind='method'), node(result, kind='result'), node('证据与结论：' + conclusion + '\n' + batch, kind='evidence')], id=ident.lower(), kind='experiment')
        groups[group - 1]['children'].append(node(question, [experiment], id=ident.lower() + '-question', kind='question'))
        rows.append(dict(topic=topic, group='OC', name=escape(experiment['label']), why=escape(question), how=escape(method), metrics=escape(result).replace('\n', '<br>'), batch=escape(batch), verdict=escape(conclusion), status=status, source=source))
    plan=ROOT/'docs/oc-head-design.md'
    result_path=REPO/'results/oc-heads/evaluation-RV.json'
    result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
    head_metrics={m['id']:m for m in result['metrics']} if result else {}
    svm=svm_results()
    svm_metrics={m['id']:m for m in svm['metrics']} if svm else {}
    attention=attention_results()
    attention_metrics={m['id']:m for m in attention['metrics']} if attention else {}
    classes={}
    head_nodes={}
    for line in plan.read_text(encoding='utf-8').splitlines():
        fields=[value.strip() for value in line.strip().strip('|').split('|')]
        if len(fields)!=8 or not re.fullmatch(r'OC-4-[1-4]',fields[0]) or not re.fullmatch(r'OC-4-\d-\d(?:-\d)?',fields[3]):
            continue
        category,title,question,ident,name,method,priority,_=fields
        if category not in classes:
            classes[category]=node(category+'：'+title,id=category.lower(),kind='direction')
            groups[3]['children'].append(classes[category])
        extra=[]
        if ident.count('-')==4 and ident in attention_metrics:
            direct={'OC-4-4-1-1':'OC-4-4-1','OC-4-4-1-2':'OC-4-4-1','OC-4-4-1-3':'OC-4-4-1','OC-4-4-2-1':'OC-4-4-2','OC-4-2-1-1':'linear-encoded-mlp'}
            baseline=direct[ident];m=attention_metrics[ident];b=attention_metrics[baseline]
            result_text=(f"结果（OC-A2历史OR；直接对照{baseline} → 候选）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
                f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）")
            c=next(c for c in attention['comparisons'] if c['id']==ident);lo,hi=c['pauc_delta_ci']
            decision={'positive':'本历史批支持提高','worse':'本历史批支持下降','unconfirmed':'改善未确认'}[c['decision']]
            conclusion=f"{decision}；主pAUC差{c['delta_pauc']:+.6f}，15比较99.666667%区间[{lo:+.6f},{hi:+.6f}]。"
            batch='OC-A2；训练300、5折选择、全批240更新；400正常校准、1000正常/500危险历史测试，三种子全部保留。无独立新确认。'
            source='results/oc-attention/report-2026-11-04.md';status='历史诊断完成'
            for control,label in [('SVM','同批OC-S1线性SVM'),('B','同批原积分B')]:
                z=attention_metrics[control]
                extra.append(node(f"结果（{label} → 候选）：AUROC {z['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {z['pauc']:.6f} → {m['pauc']:.6f}\n检出率 {z['tpr']:.2%} → {m['tpr']:.2%}；误报率 {z['fpr']:.2%} → {m['fpr']:.2%}（基线{z['fp']}条；候选{m['fp']}条）",kind='result'))
            svm_c=next(c for c in attention['comparisons'] if c['id']==ident+'-vs-SVM');l,h=svm_c['pauc_delta_ci']
            extra.append(node(f"证据（候选−SVM）：pAUC差{svm_c['delta_pauc']:+.6f}，15比较99.666667%区间[{l:+.6f},{h:+.6f}]。跨家族候选预算不同，详见协议。",kind='evidence'))
            controls={'OC-4-4-1-1':['gate-vs-ungated'],'OC-4-4-1-2':['encoding-vs-linear-interface','encoding-attention-specific'],'OC-4-2-1-1':['encoding-attention-specific'],'OC-4-4-2-1':['ft-vs-attention']}
            for key in controls.get(ident,[]):
                c2=next(c for c in attention['comparisons'] if c['id']==key);l,h=c2['pauc_delta_ci']
                term=' '.join(f'{w:+g}×{n}' for n,w in c2['terms'])
                extra.append(node(f"机制比较（{key}）：{term}；pAUC差{c2['delta_pauc']:+.6f}，99.666667%区间[{l:+.6f},{h:+.6f}]。差中差用于检查编码收益是否对注意力额外有利。",kind='evidence'))
            if ident=='OC-4-4-1-3':extra.append(node('实现：官方LibAUC1.4.0 pAUC_CVaR_Loss＋SOPA，beta0.1；辅助状态保留并验证更新。损失、优化器、裁剪与正则共同变化，不单归因损失。',kind='evidence'))
            if ident=='OC-4-4-2-1':extra.append(node('实现：作者rtdl0.0.13 FT-Transformer，16宽、1块、8头、CLS汇总。整体架构迁移，不等于作者默认规模或单组件消融。',kind='evidence'))
            extra.append(node('复核：199拟合、33新模型重载、39指标组、13万独立ROC及15项全部区间复算通过。原H3参数与分数复现。',kind='evidence'))
        elif ident.count('-')==4 and ident in svm_metrics:
            m=svm_metrics[ident];b=svm_metrics['OC-4-1-3'];original=svm_metrics['B']
            result_text=(f"结果（OC-S1历史OR-Bench；同批调参线性SVM → 候选）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
                         f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）")
            c=next(c for c in svm['comparisons'] if c['left']==ident and c['right']=='OC-4-1-3')
            lo,hi=c['pauc_delta_ci9875']
            decision={'positive':'本历史批支持提高','worse':'本历史批支持下降','unconfirmed':'改善未确认'}[c['decision']]
            conclusion=f"{decision}；pAUC差{c['delta_pauc']:+.6f}，四比较98.75%区间[{lo:+.6f},{hi:+.6f}]；不以点值或检出率单项替代主结论。"
            batch='OC-S1；同来源训练300，5折选择；400正常校准，1000正常/500危险历史测试。主种子20261104；历史诊断，无独立新确认。'
            source='results/oc-svm/report-2026-11-04.md';status='历史诊断完成'
            extra=[node(f"结果（同批原积分B → 候选）：AUROC {original['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {original['pauc']:.6f} → {m['pauc']:.6f}\n检出率 {original['tpr']:.2%} → {m['tpr']:.2%}；误报率 {original['fpr']:.2%} → {m['fpr']:.2%}（基线{original['fp']}条；候选{m['fp']}条）",kind='result'),
                   node('复核：H3固定C10逐分数复现，17模型重载、8万独立ROC及全部区间通过。前两次求解失败均在评分前停止并归档；最终求解通过数值误差上限。',kind='evidence')]
            if ident=='OC-4-1-3-1':
                fixed=svm_metrics['cost-fixed-C-unweighted']
                extra.append(node(f"固定C消融：不加权 → 所选正常权重4，低误报pAUC {fixed['pauc']:.6f} → {m['pauc']:.6f}；检出率 {fixed['tpr']:.2%} → {m['tpr']:.2%}；误报率 {fixed['fpr']:.2%} → {m['fpr']:.2%}。仅描述性，不称校正显著。",kind='result'))
            if ident=='OC-4-1-3-2':
                control=svm_metrics['linear-kernel-SVC'];c2=next(c for c in svm['comparisons'] if c['right']=='linear-kernel-SVC');l,h=c2['pauc_delta_ci9875']
                extra.append(node(f"核机制对照（线性核SVC → RBF）：低误报pAUC {control['pauc']:.6f} → {m['pauc']:.6f}；98.75%差区间[{l:+.6f},{h:+.6f}]，本历史批支持下降。",kind='result'))
            if ident=='OC-4-1-3-3':extra.append(node('边界：top-negative成对hinge凸代理目标＋cutting-plane QP为本项目迁移实现，不是原论文作者求解器的精确复现；目标和求解器共同变化。',kind='evidence'))
        elif ident in head_metrics:
            m=head_metrics[ident];b=head_metrics['B']
            result_text=(f"结果（历史RV1082）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
                         f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（{m['fpr_count']}/{m['fpr_n']}）")
            lo,hi=m['pauc_delta_ci99375']
            conclusion={'positive_interval':'本批低误报pAUC差区间为正，仍属历史诊断','unconfirmed':'本批低误报改善未获区间支持','worse':'本批低误报pAUC更差'}[m['historical_decision']]
            batch=f"主种子{result['primary_seed']}；300正常/782危险；99.375%主差区间[{lo:+.6f}, {hi:+.6f}]；无独立新确认。"
            source='results/oc-heads/report-2026-11-04.md';status='历史验证完成'
        else:
            result_text='结果：待验证，暂无严格直接对照数值。'
            conclusion='固定同样的14层F、训练与选择标签预算，比较不同检测头。'
            batch=priority+'；具体配置与评估规则在评分前冻结。'
            source='docs/oc-heads-protocol-2026-11-04.md' if (REPO/'docs/oc-heads-protocol-2026-11-04.md').exists() else 'docs/oc4-boundary-update-2026-11-04.md'
            status='待验证'
            source='../../docs/oc-svm-attention-design.md'
        experiment=node(ident+'：'+name,[node('方法：'+method,kind='method'),node(result_text,kind='result'),node('证据与结论：'+conclusion+'\n'+batch,kind='evidence')]+extra+confirmation_evidence(ident)+retraining_evidence(ident)+svm_baseline_evidence(ident)+attention_baseline_evidence(ident)+original_oc4_evidence(ident),id=ident.lower(),kind='experiment')
        target=head_nodes[ident.rsplit('-',1)[0]]['children'] if ident.count('-')==4 else classes[category]['children']
        target.append(node(question,[experiment],id=ident.lower()+'-question',kind='question'))
        head_nodes[ident]=experiment
        rows.append(dict(topic='heads',group='OC',name=escape(experiment['label']),why=escape(question),how=escape(method),metrics=escape(result_text).replace('\n','<br>'),batch=escape(batch),verdict=escape(conclusion),status=status,source=source))
    assert [len(g['children']) for g in groups] == [3, 3, 5, 4]
    assert [len(g['children']) for g in groups[3]['children']]==[3,1,2,2]
    return groups, rows


def comparison_html(text):
    html=escape(text).replace('\n','<br>')
    def emphasize(match):
        before,after=match.group(1),match.group(2)
        lower=bool(re.search(r'(?:误报率|误报|FPR)\s*$',html[:match.start()]))
        improved=float(after.rstrip('%'))<float(before.rstrip('%')) if lower else float(after.rstrip('%'))>float(before.rstrip('%'))
        return before+' → '+('<b>'+after+'</b>' if improved else after)
    return re.sub(r'(\d+(?:\.\d+)?%?)\s*→\s*(\d+(?:\.\d+)?%?)',emphasize,html)


def pca_uncapped_result(ident):
    """Attach a verified new run without replacing the frozen PA1 result nodes."""
    folder=REPO/'results/ei2-pca-uncapped'
    if not (folder/'verification.json').exists():return None
    verification=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
    assert verification['passed'] and verification['evaluation_sha256']==hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest()
    result=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'));assert result['run_id']=='EI-PA2' and result['numbering_version']==3
    chosen=result['selected'][ident];control=ident+'-cap32';leaves=[];texts=[];decisions=[]
    leaves.append(node('实际选择：'+chosen['variant']['key']+'；最终C='+str(chosen['C'])+'；各头种子每层数量'+str(chosen['counts_by_seed'])+'；完整训练各层数值秩'+str(result['ranks']),kind='method'))
    leaves.append(node('同基底直接对照：本方法限制32；选择'+result['selected'][control]['variant']['key']+'，各头种子数量'+str(result['selected'][control]['counts_by_seed'])+'。不是把新完整PCA与旧随机PCA直接相减。',kind='evidence'))
    if 'coverage' in chosen['variant']:
        coverage=chosen['variant']['coverage'];capped=chosen['variant']['capped']
        leaves.append(node(f"解释率检查：目标{chosen['variant']['tau']:.0%}；实际各层覆盖率{min(coverage):.2%}～{max(coverage):.2%}；未达标{sum(capped)}/14层。分母为原中心化残差总能量。",kind='evidence'))
    for role,label in [('C','历史C1000'),('RV','历史RV1082'),('OR_test','历史OR-Bench1500')]:
        data=result['cohorts'][role];m=data[ident][0];b=data[control][0]
        text=(f"结果（EI-PA2，{label}；同基底本方法限制32 → 解除上限）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
              f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）")
        texts.append(text);leaves.append(node(text,kind='result'));c=data['comparisons'][ident];lo,hi=c['ci983333'];decision='差区间为正' if lo>0 else ('差区间为负' if hi<0 else '差区间含零');decisions.append(label+'：'+decision)
        leaves.append(node(f"证据（解除−限制32）：AUROC差{c['delta_auroc']:+.6f}，三比较98.333333%配对区间[{lo:+.6f},{hi:+.6f}]；{decision}。",kind='evidence'))
    m=result['cohorts']['OR_test'][ident][0]['source_calibrated'];b=result['cohorts']['OR_test'][control][0]['source_calibrated']
    leaves.append(node(f"结果（OR来源400正常重新校准）：检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}/1000；候选{m['fp']}/1000）",kind='result'))
    conclusion='；'.join(decisions)+'。完整方向范围不保证更准；全部为历史诊断，没有独立新来源确认。'
    leaves.append(node('结论：'+conclusion,kind='evidence'))
    leaves.append(node('边界：CV为覆盖至自然秩的离散网格，非全整数穷举。门控采用平均开度惩罚，方向变多时每方向惩罚会变化；初始化匹配末端开度。有效输入容量随方向数增加，额外选择预算单列。旧PA1冻结成绩保留。',kind='evidence'))
    return leaves,'\n'.join(texts),conclusion


def pca_joint_result(ident):
    folder=REPO/'results/ei2-pca-joint'
    if not (folder/'final-verification.json').exists():return None
    proof=json.loads((folder/'final-verification.json').read_text(encoding='utf-8'))
    assert proof['passed'] and proof['evaluation_sha256']==hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest()
    for filename,digest in proof['files'].items():assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==digest
    result=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'));assert result['run_id']=='EI-PJ1' and result['numbering_version']==3
    chosen=result['selected'][ident];controls=result['controls'][ident];leaves=[];texts=[];decisions=[]
    labels={'Q-CV':'仅PCA前缀CV选数量','Q-VAR':'仅解释率选数量','Q-GATE':'仅原顺序门控','S-EN':'仅EN方向筛选','S-L1':'仅L1方向筛选'}
    leaves.append(node('实际选择：'+chosen['variant']['key']+'；最终C='+str(chosen['C'])+'；各seed每层数量'+str(chosen['counts_by_seed'])+'。'+('本次包含原PCA序号的跳选。' if not all(chosen['is_pca_prefix_by_seed']) else '本次恰好保留原PCA前缀。')+'具体坐标见本轮JSON。',kind='method'))
    if 'pool_coverage' in chosen['variant']:
        before=chosen['variant']['pool_coverage'];after=chosen['variant']['selected_coverage']
        leaves.append(node(f"解释率边界：筛前池覆盖{min(before):.2%}～{max(before):.2%}，筛后所选原PCA坐标保留{min(after):.2%}～{max(after):.2%}；两者分母均为原残差总能量，筛后不保证仍达到池目标。",kind='evidence'))
    def metric(label,b,m):
        return (f"结果（{label}）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
                f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；组合{m['fp']}条）")
    for role,label in [('C','历史C1000'),('RV','历史RV1082'),('OR_test','历史OR-Bench1500')]:
        data=result['cohorts'][role];m=data[ident][0];intervals=[]
        for control in controls:
            text=metric(label+'；'+labels[control]+' → 本组合',data[control][0],m);texts.append(text);leaves.append(node(text,kind='result'));c=data['comparisons'][ident][control];lo,hi=c['ci991667'];intervals.append((lo,hi))
            leaves.append(node(f"证据（组合−{labels[control]}）：AUROC差{c['delta_auroc']:+.6f}；本批六比较99.166667%区间[{lo:+.6f},{hi:+.6f}]。",kind='evidence'))
        decision='同时优于两单组件获本批区间支持' if all(lo>0 for lo,hi in intervals) else ('至少比一个单组件更差获本批区间支持' if any(hi<0 for lo,hi in intervals) else '尚未确认同时优于两单组件');decisions.append(label+'：'+decision)
        leaves.append(node(metric(label+'；同数量原PCA前缀 → 本组合，描述性、同C',data[ident+'-prefix'][0],m),kind='result'))
    for control in controls:
        b=result['cohorts']['OR_test'][control][0]['source_calibrated'];m=result['cohorts']['OR_test'][ident][0]['source_calibrated']
        leaves.append(node(f"结果（OR来源400正常重新校准；{labels[control]} → 组合）：检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}/1000；组合{m['fp']}/1000）",kind='result'))
    conclusion='；'.join(decisions)+'。全部为历史诊断，不代表未使用来源泛化；点值改善不等于组合增益确认。'
    leaves.append(node('结论：'+conclusion,kind='evidence'))
    leaves.append(node('边界：原14F/层/末位/最终L2头/标签池固定；PCA和排序/筛选只fit训练侧，CV每折重做排序和标准化。588 LR＋18稀疏选择器＋12 GPU门，额外选择预算不同；原坐标映射后重训，旧成绩不改。',kind='evidence'))
    return leaves,'\n'.join(texts),conclusion


def pca_automatic_children():
    """Register planned adaptive variants from one maintained design table."""
    path=ROOT/'docs/pca-candidate-design.md'
    groups={};rows=[];results={}
    for group,run in [('EI-2-3-2','ei2-pca-auto'),('EI-2-3-3','ei2-pca-select')]:
        folder=REPO/'results'/run
        if (folder/'verification.json').exists():
            verified=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
            assert verified['passed'] and verified['evaluation_sha256']==hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest()
            results[group]=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'));assert results[group]['numbering_version']==3
    aliases={r['current']:r['old'] for r in json.loads((ROOT/'docs/ei-pca-numbering-v3.json').read_text(encoding='utf-8'))['automatic']}
    for line in path.read_text(encoding='utf-8').splitlines():
        fields=[s.strip() for s in line.strip().strip('|').split('|')]
        if len(fields)!=7 or not re.fullmatch(r'EI-2-3-[234]-[1-3]',fields[2]):continue
        group,question,ident,title,method,control,status=fields
        assert status in ['待验证','历史匹配验证完成']
        status='待验证'
        gid=group.split(' ',1)[0]
        result=results.get(gid);selecting=gid=='EI-2-3-3';baseline='K32' if selecting else 'K8';baseline_label='固定32方向' if selecting else '固定8方向'
        if gid not in groups:groups[gid]=node(group.replace(' ', '：',1),id=gid.lower(),kind='direction')
        leaves=[
            node('方法：'+method,kind='method'),
            node('直接对照：'+control,kind='evidence'),
            node('结果：待验证；AUROC、低误报pAUC、检出率、误报率及误报条数暂无新成绩。',kind='result'),
            node('旧设计编号：'+aliases[ident]+'（版本2，未执行设计）；当前编号按版本3，不改任何旧运行键。' if ident in aliases else '新组合方向：'+ident+'；不是旧结果改名，不借用单组件成绩。',kind='evidence')]
        metric_text='待验证；AUROC、低误报pAUC、检出率、误报率及误报条数暂无新成绩。';batch='方案登记，未启动实验；原14F保留，变换只在训练侧确定，选择及训练预算执行前冻结。';conclusion='是否有效待匹配验证；不借用固定方向数结果。';source='../../docs/pca-candidate-design.md'
        if result and ident in result['selected']:
            chosen=result['selected'][ident];texts=[];decisions=[];leaves=[leaf for leaf in leaves if leaf.get('kind')!='result']
            counts=chosen['counts'] if selecting else chosen['counts_by_seed'][0]
            leaves.append(node('实际选择：'+str(chosen['variant']['key'])+'；各层数量'+str(counts)+'；最终C='+str(chosen['C']),kind='method'))
            if selecting:
                leaves.append(node('具体方向：'+str(chosen['direction_indices_1based_by_layer'])+'；按原0基层16～29排列，每层PCA方向从1开始。'+('恰好为前缀。' if chosen['is_prefix'] else '含跳选，不是只取前几个。'),kind='evidence'))
                stability=chosen['stability'];leaves.append(node('训练重采样5次：保留数量'+str(stability['directions'])+'；与原mask的Jaccard '+str([round(v,3) for v in stability['jaccard']])+'。固定PCA和标准化的条件检查；留得越多本就越容易重合。',kind='evidence'))
                if ident.endswith('-3'):
                    leaves.append(node('ARD边界：分类MAP通过梯度核验；外层alpha更新在100轮上限仍未达固定点，是有迭代上限的近似结果。',kind='evidence'))
            if 'capped' in chosen['variant']:
                capped=sum(chosen['variant']['capped']);leaves.append(node(f"解释率边界：14层中{capped}层因32方向上限未达到目标比例；未达标不代表已保留足够方差。",kind='evidence'))
            for role,label in [('C','历史C1000'),('RV','历史RV1082'),('OR_test','历史OR-Bench1500')]:
                data=result['cohorts'][role];m=data[ident][0];b=data[baseline][0]
                text=(f"结果（{label}；同基底{baseline_label} → 本项）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
                      f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）")
                texts.append(text);leaves.append(node(text,kind='result'));c=data['comparisons'][ident][baseline] if selecting else data['comparisons'][ident];lo,hi=c['ci991667' if selecting else 'ci983333'];decision='差区间为正' if lo>0 else ('差区间为负' if hi<0 else '差区间含零');decisions.append(label+'：'+decision)
                interval_label='六比较99.166667%' if selecting else '三比较98.333333%'
                leaves.append(node(f"证据（本项−{baseline_label}）：AUROC差{c['delta_auroc']:+.6f}，{interval_label}区间[{lo:+.6f},{hi:+.6f}]；{decision}。",kind='evidence'))
                if selecting:
                    p=data[ident+'-prefix'][0];c=data['comparisons'][ident][ident+'-prefix'];lo,hi=c['ci991667']
                    text=(f"结果（{label}；每层同数量前几个 → 本项；同C）：AUROC {p['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {p['pauc']:.6f} → {m['pauc']:.6f}\n检出率 {p['tpr']:.2%} → {m['tpr']:.2%}；误报率 {p['fpr']:.2%} → {m['fpr']:.2%}（基线{p['fp']}条；候选{m['fp']}条）")
                    texts.append(text);leaves.append(node(text,kind='result'));leaves.append(node(f"证据（本项−同数量前缀）：AUROC差{c['delta_auroc']:+.6f}，六比较99.166667%区间[{lo:+.6f},{hi:+.6f}]；前缀沿用本项C，不是单独调参冠军。",kind='evidence'))
            m=result['cohorts']['OR_test'][ident][0]['source_calibrated'];b=result['cohorts']['OR_test'][baseline][0]['source_calibrated']
            leaves.append(node(f"结果（OR来源400正常重新校准）：检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}/1000；候选{m['fp']}/1000）",kind='result'))
            metric_text='\n'.join(texts);batch='EI-PA1，编号版本3；360次LR＋12次门选择器训练；原14F/PCA32/末位及标签池固定，但额外选择预算不同。全为历史诊断，无独立新确认。'
            if selecting:batch='EI-PS1，编号版本3；90次最终LR＋22次选择器拟合；14F/PCA32/末位/最终L2头固定。另有选择预算；全部为历史诊断，无独立新确认。'
            conclusion='；'.join(decisions)+('。筛选能压缩输入，但是否提高判断效果取决于批次；不能据此认定普遍优于全部32方向。' if selecting else '。只说明本次有上限的选择规则；不代表自动数量普遍更好。');status='历史匹配验证完成';source='results/'+('ei2-pca-select' if selecting else 'ei2-pca-auto')+'/report-2026-11-05.md'
            if 'capped' in chosen['variant'] and any(chosen['variant']['capped']):conclusion+=' 解释率法已触及32上限，部分层未达目标，不能称已找到满足阈值的数量。'
            leaves.append(node('结论：'+conclusion,kind='evidence'))
        if gid=='EI-2-3-4':
            batch='组合设计登记，未启动实验；依赖EI-PA2完整训练数值秩核验。原14F保留，最终仍用原L2头；所有排序/筛选/标准化只fit训练侧，CV各折重新拟合。'
            conclusion='待验证：须分别对比数量单独、方向单独，不能只超过较弱对照就认定组合收益；测试不用于选择方案。'
            source='docs/ei2-pca-joint-design-2026-11-05.md'
            leaves.append(node('结论：'+conclusion,kind='evidence'))
            leaves.append(node('设计来源：EN排序与CV选数量优先；解释率池＋L1次优先；EN重排＋门控探索。ARD外层未收敛，首轮不加入。L1/EN本来就会产生数量，此处增加明确组合结构，避免重复立项。',kind='evidence'))
            joint=pca_joint_result(ident)
            if joint:
                leaves=[leaf for leaf in leaves if leaf.get('kind')!='result' and not leaf['label'].startswith('结论：')]+joint[0]
                metric_text=joint[1];conclusion=joint[2];status='历史匹配验证完成';batch='EI-PJ1，历史组合诊断；588 LR、18稀疏选择器、12 GPU门；两单组件和同数量前缀分别比较，未使用新来源仍待确认。';source='results/ei2-pca-joint/report-2026-11-05.md'
            elif (REPO/'docs/ei2-pca-joint-progress.json').exists():
                progress=json.loads((REPO/'docs/ei2-pca-joint-progress.json').read_text(encoding='utf-8'))
                if progress.get('status') in ['running','verifying']:
                    leaves=[leaf for leaf in leaves if leaf.get('kind')!='result']+[node('结果：待验证；EI-PJ1已冻结并启动，正在训练或独立核验；暂无通过核验的新成绩。',kind='result')];batch='EI-PJ1执行中；通过独立核验后才回填新分数，旧成绩不改。'
        if gid=='EI-2-3-2' and (REPO/'docs/ei2-pca-uncapped-protocol-2026-11-05.md').exists():
            progress_path=REPO/'docs/ei2-pca-uncapped-preflight-2026-11-05.json'
            progress=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.exists() else {}
            progress_text='EI-PA2已启动；GPU扫描梯度与确定性矩阵乘法契约通过，正在完整谱和数量选择训练。' if progress.get('status')=='running' else '新真实训练尚未启动，等待SSH恢复后执行GPU契约。'
            extension={
                'EI-2-3-2-1':'候选从1～32扩到训练数值秩；预先固定离散网格，含完整秩75%与100%，不是穷举每个整数。三折内重新拟合PCA，增加同新基底限制32的匹配对照。',
                'EI-2-3-2-2':'计算完整数值非零PCA谱，按原总方差逐层保留到80%／90%／95%，不再硬截断至32；增加同新基底限制32对照。',
                'EI-2-3-2-3':'每层全部数值方向都进入前缀门控；初始化匹配初始末端开度，避免后部先天关闭；增加同新基底限制32对照。'}[ident]
            new_result=pca_uncapped_result(ident)
            new_leaves=[
                node('方法：'+extension,kind='method'),
                node('共同条件：原14F、14层、读取末位、标签池、最终L2头保留；范围放开到训练数值秩，不另设任意固定上限。',kind='evidence'),
                node('结果：待验证；'+progress_text+'上方为旧EI-PA1有32上限的历史结果，不能当作本轮成绩。',kind='result')
                ]
            if new_result:
                new_leaves=new_leaves[:2]+new_result[0];metric_text='旧EI-PA1（32方向上限）：\n'+metric_text+'\n'+new_result[1];conclusion='旧EI-PA1：'+conclusion+' 新EI-PA2：'+new_result[2];source='results/ei2-pca-uncapped/report-2026-11-05.md'
            leaves.append(node('补充验证：解除32方向限制（EI-PA2，'+('历史匹配验证完成' if new_result else '待验证')+'）',new_leaves,id=ident.lower()+'-uncapped',kind='direction'))
            batch+='；解除32方向上限的EI-PA2'+('历史匹配验证完成' if new_result else '补跑待验证')+'，旧成绩保留。'
        leaves.append(node('边界：'+batch,kind='evidence'))
        experiment=node(ident+'：'+title,leaves,id=ident.lower(),kind='experiment')
        groups[gid]['children'].append(node(question,[experiment],id=ident.lower()+'-question',kind='question'))
        rows.append(dict(topic='logits',group='EI',name=ident+'：'+title,why=question,how=method+'；直接对照：'+control,
            metrics=comparison_html(metric_text),batch=batch,verdict=conclusion,status=status,source=source))
    assert len(rows)==9 and [len(n['children']) for n in groups.values()]==[3,3,3]
    return list(groups.values()),rows


def pca_direction_children(original_four,original_row):
    """Children of the existing EI-2-3; parent frozen results stay unchanged."""
    evaluations={}
    for group in ['ei2-pca-dimensions','ei2-pca-extended']:
        folder=REPO/'results'/group;evaluations[group]=None
        if (folder/'verification.json').exists():
            verification=json.loads((folder/'verification.json').read_text(encoding='utf-8'))
            assert verification['passed'] and verification['evaluation_sha256']==hashlib.sha256((folder/'evaluation.json').read_bytes()).hexdigest()
            evaluations[group]=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    questions,rows=[],[]
    definitions=[('EI-2-3-1',1,'只保留最主要的1个方向，信息够不够？'),('EI-2-3-2',2,'保留2个方向，能否兼顾精简与判断？'),('EI-2-3-3',8,'增加到8个方向，额外细节会不会有用？'),('EI-2-3-4',16,'增加到16个方向，比8个方向更有帮助吗？'),('EI-2-3-5',32,'继续增加到32个方向，还能获得额外收益吗？')]
    for ident,k,question in definitions:
        title=f'{ident}：每层{k}个残差PCA方向'
        extended=k>=16;group='ei2-pca-extended' if extended else 'ei2-pca-dimensions';evaluation=evaluations[group]
        basis=32 if extended else 8;control_name='K8' if extended else 'K4'
        controls=[('old-K8','前轮EI-2-3-3旧8方向'),('K8','主对照：同一PCA32基底的前8方向')] if extended else [('old-K4','父级EI-2-3旧4方向'),('K4','主对照：同一PCA8基底的前4方向')]
        if k==32:controls.append(('EI-2-3-4','额外对照：本轮16方向'))
        method=f'原14F保留；每层只取训练侧统一PCA{basis}基底的前{k}个方向，共{14+14*k}维，补零至512。各k的额外坐标都按原4方向系数缩放，不根据测试选方向。'
        leaves=[node('方法：'+method,kind='method')]
        batch='EI-PCA1；同EI-V1的训练1600/dev600/cal400和历史C/RV/OR；同逻辑回归、3个C×3种子，主AUROC选参。旧4维不覆盖，另设同基底K4。'
        if extended:batch='EI-PCA2；同EI-V1的数据与条件；统一PCA32前4/8/16/32方向、同一逻辑回归与每臂3C×3种子。主比较16−8、32−8、32−16；旧4/8结果不覆盖。'
        conclusion='待验证；主要对本轮同基底'+control_name+'，旧方向数结果另列追溯。';texts=['结果：待验证。'];status='待验证'
        source='docs/'+group+'-protocol-2026-11-05.md'
        if evaluation:
            texts=[];decisions=[]
            for role,label in [('C','历史C1000'),('RV','历史RV1082'),('OR_test','历史OR-Bench1500')]:
                data=evaluation['cohorts'][role];m=data[ident][0]
                for control,trace in controls:
                    b=data[control][0]
                    text=(f"结果（{label}；{trace} → 本项）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
                          f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）")
                    leaves.append(node(text,kind='result'));texts.append(text)
                c=data['comparisons'][ident];lo,hi=c['ci983333'];decision='差区间为正' if lo>0 else ('差区间为负' if hi<0 else '差区间含零')
                decisions.append(label+'：'+decision)
                leaves.append(node(f"证据（本项−同基底{control_name}）：AUROC差{c['delta_auroc']:+.6f}，三比较98.333333%配对区间[{lo:+.6f},{hi:+.6f}]；{decision}。",kind='evidence'))
                if k==32:
                    extra=data['comparisons']['EI-2-3-5-vs-16'];lo2,hi2=extra['ci983333']
                    leaves.append(node(f"证据（32−16方向）：AUROC差{extra['delta_auroc']:+.6f}，同一家族98.333333%区间[{lo2:+.6f},{hi2:+.6f}]。",kind='evidence'))
            conclusion='；'.join(decisions)+'。全部为已观察的历史数据，不能称独立新确认。方向增加也增加有效容量；噪声控制与三种子见报告。'
            status='历史匹配验证完成';source='results/'+group+'/report-2026-11-05.md'
        else:leaves.append(node(texts[0],kind='result'))
        leaves.extend([node('直接对照：主要与本轮同基底'+control_name+'比较；旧结果单列，避免把随机化PCA基底改变混同为方向数改变。',kind='evidence'),node('结论：'+conclusion,kind='evidence'),node('边界：'+batch,kind='evidence')])
        questions.append(node(question,[node(title,leaves,id=ident.lower(),kind='experiment')],id=ident.lower()+'-question',kind='question'))
        rows.append(dict(topic='logits',group='EI',name=title,why=question,how=method,metrics=comparison_html('\n'.join(texts)),batch=batch,verdict=conclusion,status=status,source=source))
    mapping=json.loads((ROOT/'docs/ei-pca-numbering-v3.json').read_text(encoding='utf-8'))['fixed']
    for entry in mapping:
        if entry['directions_per_layer']==4:continue
        old,new=entry['old'],entry['current']
        question=next(n for n in questions if n['children'][0]['id']==old.lower());experiment=question['children'][0]
        question['id']=new.lower()+'-question';experiment['id']=new.lower();experiment['label']=experiment['label'].replace(old+'：',new+'：',1)
        experiment['children'].append(node('旧编号追溯：'+old+'（版本2／'+entry['run']+'）；仅改变展示归属，原运行键和结果不改。',kind='evidence'))
        row=next(n for n in rows if n['name'].startswith(old+'：'));row['name']=row['name'].replace(old+'：',new+'：',1);row['batch']+=' 旧编号：'+old+'（'+entry['run']+'，版本2）。'
        for leaf in experiment['children']:
            leaf['label']=leaf['label'].replace('前轮EI-2-3-3旧8方向','前轮8方向（旧EI-2-3-3／EI-PCA1）')
        row['metrics']=row['metrics'].replace('前轮EI-2-3-3旧8方向','前轮8方向（旧EI-2-3-3／EI-PCA1）')
    original_four=deepcopy(original_four);original_four['id']='ei-2-3-1-3';original_four['label']='EI-2-3-1-3：每层4方向（原方案／对照）'
    original_four['children'].append(node('旧编号追溯：EI-2-3（EI-V1，原4方向）；本分支不代表新增4方向优化，也不混同各轮匹配基底的K4对照。',kind='evidence'))
    questions.insert(2,node('原方案每层4方向，作为对照表现如何？',[original_four],id='ei-2-3-1-3-question',kind='question'))
    original_row=deepcopy(original_row);original_row['name']=original_four['label'];original_row['why']='原方案每层4方向，作为对照表现如何？';original_row['batch']+=' 原EI-2-3／EI-V1四方向追溯，非新增运行。';rows.insert(2,original_row)
    return node('EI-2-3-1：手动指定方向数量',questions,id='ei-2-3-1',kind='direction'),rows


def build_ei2():
    """Keep the registered six questions; attach matched results when verified."""
    path=ROOT/'docs/vocabulary-design.md'
    result_path=REPO/'results/ei2-vocabulary/evaluation.json'
    verified_path=REPO/'results/ei2-vocabulary/verification.json'
    result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() and verified_path.exists() else None
    if result:
        verification=json.loads(verified_path.read_text(encoding='utf-8'))
        assert verification['passed'] and verification['evaluation_sha256']==hashlib.sha256(result_path.read_bytes()).hexdigest(), 'EI-V1 verification does not match evaluation'
    questions,rows=[],[]
    for line in path.read_text(encoding='utf-8').splitlines():
        fields=[v.strip() for v in line.strip().strip('|').split('|')]
        if len(fields)!=8 or not re.fullmatch(r'EI-2-[1-6]',fields[0]):continue
        ident,title,question,features,method,control,stage,status=fields
        result_nodes=[node('结果：待验证；暂无严格匹配的直接对照数值。',kind='result')]
        result_text='待验证；AUROC、低误报pAUC、检出率、误报率及误报条数暂无新成绩。'
        batch=stage+'；旧OC-4-3／迁入EI。非已执行、非已冻结协议。'
        conclusion='是否有增益待匹配对照验证；历史logits及OC-4检测头成绩不得倒填。'
        source='../../docs/vocabulary-design.md'
        if result:
            baseline='EI-2-5' if ident=='EI-2-6' else 'B-F'
            result_nodes=[];texts=[];decisions=[]
            for role,label in [('C','历史C1000'),('RV','历史RV1082'),('OR_test','历史OR-Bench1500')]:
                cohort=result['cohorts'][role];m=cohort[ident][0];b=cohort[baseline][0]
                text=(f"结果（{label}；对照{baseline}）：AUROC {b['auroc']:.6f} → {m['auroc']:.6f}；低误报pAUC {b['pauc']:.6f} → {m['pauc']:.6f}\n"
                      f"检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}条；候选{m['fp']}条）")
                result_nodes.append(node(text,kind='result'));texts.append(text)
                if ident in ['EI-2-5','EI-2-6']:
                    comparison=cohort['full_auroc_differences'][ident+'_vs_'+baseline];lo,hi=comparison['ci975'];confidence='97.5%'
                else:
                    comparison=cohort['primary_auroc_differences'][ident];lo,hi=comparison['ci9875'];confidence='98.75%'
                decision='差区间为正' if lo>0 else ('差区间为负' if hi<0 else '差区间含零')
                decisions.append(label+'：'+decision)
                result_nodes.append(node(f"证据（{label}）：AUROC差{comparison['delta']:+.6f}，{confidence}配对区间[{lo:+.6f},{hi:+.6f}]；{decision}。",kind='evidence'))
            m=result['cohorts']['OR_test'][ident][0]['source_calibrated'];b=result['cohorts']['OR_test'][baseline][0]['source_calibrated']
            result_nodes.append(node(f"结果（OR来源400正常样本重新校准阈值）：检出率 {b['tpr']:.2%} → {m['tpr']:.2%}；误报率 {b['fpr']:.2%} → {m['fpr']:.2%}（基线{b['fp']}/1000；候选{m['fp']}/1000）",kind='result'))
            result_text='\n'.join(texts)
            dim=result['dimensions'][ident]
            batch=f"EI-V1；train1600/dev600/cal400；固定原14层、模板末位、同一逻辑回归、每臂3个C×3种子。输入{dim['raw_dimension']}维，接口{dim['interface_dimension']}维；全部测试均已被历史实验使用，没有独立新确认。"
            conclusion='；'.join(decisions)+'。仅说明这些历史批次，不能据此宣称新增信息已确认或新来源泛化。容量与正则化也会变化，噪声和F复制对照见报告。'
            status='历史匹配验证完成';source='results/ei2-vocabulary/report-2026-11-04.md'
        experiment=node(ident+'：'+title,[
            node('输入：'+features,kind='method'),
            node('方法：'+method,kind='method'),
            node('直接对照：'+control,kind='evidence'),
            *result_nodes,
            node('结论：'+conclusion,kind='evidence'),
            node('边界：固定原14层F对应的层、模板末位、同一检测头及标签/调参预算；变换只在训练侧确定。'+batch+'；旧OC-4-3／迁入EI，不是当前树模型编号。',kind='evidence')],id=ident.lower(),kind='experiment')
        if ident=='EI-2-3':
            original_row=dict(topic='logits',group='EI',name=experiment['label'],why=question,how='原14F＋每层4方向，原EI-V1方案；标准化和PCA均仅训练侧拟合。',metrics=comparison_html(result_text),batch=batch,verdict=conclusion,status=status,source=source)
            children,child_rows=pca_direction_children(experiment,original_row)
            experiment['label']='EI-2-3：补充残差PCA';experiment['kind']='direction';experiment['children']=[children]
            automatic,automatic_rows=pca_automatic_children();experiment['children'].extend(automatic)
        questions.append(node(question,[experiment],id=ident.lower()+'-question',kind='question'))
        metric_html=comparison_html(result_text)
        rows.append(dict(topic='logits',group='EI',name=escape(experiment['label']),why=escape(question),
            how=escape('输入：'+features+'；方法：'+method+'；直接对照：'+control),
            metrics=metric_html,batch=batch,verdict=conclusion,status=status,source=source))
        if ident=='EI-2-3':rows[-1].update(status='方向分组',metrics='按手动指定、自动数量、自动方向、数量与方向组合四类展开；原4方向成绩见EI-2-3-1-3。',how='原14F保留并补充残差PCA信息，各具体候选见子分支。',verdict='原4方向记录迁入对照叶子，旧结果不改写；自动及组合候选须分别验证。')
    assert len(questions)==len(rows)==6
    rows.extend(child_rows)
    rows.extend(automatic_rows)
    return questions,rows


def build_ei1():
    """Current registry IDs, preserving frozen C32/all64 keys and legacy DOM anchors."""
    ids=json.loads((REPO/'docs/experiment-numbering.json').read_text(encoding='utf-8'))['centroid']
    metrics=json.loads((REPO/'results/signal-selection/evaluation.json').read_text(encoding='utf-8'))['tests']['test_aegis']
    base=metrics['F32'];questions,rows=[],[]
    variants=[('C32','只看离安全、危险样本中心的接近程度，能判断危险吗？','只看质心线索',
        '在词表投影之前读取32层末位隐藏状态；各层用带标签训练样本形成安全/危险单位中心，计算与危险中心的余弦减去与安全中心的余弦，得到32个C；只输入C32训练逻辑回归。',
        '本批AUROC和检出率点值更高，但误报增加；未单独登记C32相对F32的配对差区间，不能只凭点值确认优越。'),
       ('all64','F和质心线索一起看，能否补上彼此漏掉的信息？','F＋质心线索',
        '同样在投影前提取C32，与原32层F32拼成64维，交给逻辑回归；训练侧拟合质心和标准化。实际对照为F32＋逻辑回归。',
        '相对F32，本历史批AUROC差95%区间[+0.02116,+0.05138]；误报也增加。相对只看C32的额外AUROC收益未获区间支持，不能直接断言二者互补。')]
    for key,question,title,method,conclusion in variants:
        ident=ids['replace_history'] if key=='C32' else ids['add_history']
        m=metrics[key]
        result=(f"结果（历史同批1000条）：AUROC {base['auroc']:.5f} → {m['auroc']:.5f}\n"
                f"检出率 {base['tpr']:.1%} → {m['tpr']:.1%}；误报率 {base['fpr']:.1%} → {m['fpr']:.1%}（基线{base['fp']}/500；候选{m['fp']}/500安全输入）\n"
                '低误报pAUC：历史归档未登记，本次未补算。')
        evidence='历史监督框架：32层特征，不是当前固定14层的严格匹配验证。质心使用训练标签；相同头家族不等于输入容量相同。区间未作多重比较修正；旧结果原样追溯。'
        exp=node(ident+'：'+title+'（历史候选'+key+'）',[
            node('方法：'+method,kind='method'),
            node('实际对照：同批F32＋逻辑回归，AUROC起点不是原论文无训练积分。',kind='evidence'),
            node(result,kind='result'),node('结论：'+conclusion+'\n'+evidence,kind='evidence'),
            node('待验证：固定同样14层、读取位置、检测头与标签/调参预算后，新增C是否仍有用；没有新成绩。',kind='evidence')],id='ei1-'+key.lower(),kind='experiment')
        questions.append(node(question,[exp],id='ei1-'+key.lower()+'-question',kind='question'))
        rows.append(dict(topic='centroid',group='EI',name=ident+'：'+title+'（历史'+key+'）',why=question,how=method,
            metrics=escape(result).replace('\n','<br>'),batch=evidence,verdict=conclusion,status='历史监督结果',source='results/signal-selection/report-2026-10-28.md'))
    return questions,rows
