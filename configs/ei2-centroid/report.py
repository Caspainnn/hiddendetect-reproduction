"""Render only independently verified EI-C1 results."""
from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[2];d=r/'results/ei2-centroid';p=json.loads((d/'verification.json').read_text());assert p['passed'] and p['selected_entries_bound']==9
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for file,key in [('evaluation.json','evaluation_sha256'),('scores.npz','scores_sha256'),('source-lock.json','source_lock_sha256')]:assert sha(d/file)==p[key]
e=json.loads((d/'evaluation.json').read_text());arm='EI-2-7-2'
lines=['# EI-C1：原14F＋质心C14（EI-2-7-2）','',
'本轮回答“词表拒绝分数F之外，原隐藏状态的安全/危险质心线索能否补充信息”。使用EI-PJ1同一6582条清单与角色，保留原14层、模板末位、标签预算与最终L2头。新前向提取每层隐藏状态，逐条检查对应F及token哈希；训练侧拟合两类单位质心，C为危险余弦减安全余弦。原F不重定义。','',
'历史EI-2-7-1的F32＋C32保留，不搬用旧成绩。仅C14的EI-3-1-2没有执行，仍待验证。全部测试均已用于此前实验，是历史诊断，不是独立新确认。','',
'## 方法与控制','',
'- train1600、dev600、cal400、C1000、RV1082、OR_cal400、OR_test1500，沿用原顺序、ID与输入哈希。原0基层16～29，对应hidden_states[17:31]，读取模板末位投影前的原隐藏状态。',
'- 训练隐藏向量单位化，分别对安全/危险类取均值再单位化。train样本也使用这份训练质心，包含自身；dev和测试只变换，不拟合。不是原RCS完整复现。',
'- 原14F、14F＋C14、14F＋14维确定性高斯噪声三臂。噪声只作同维度描述性检查，不做额外主区间。每臂同C=[0.1,1,10]、三种子20261104/05/06、L2 liblinear、最多5000迭代，共27拟合。',
'- 标准化均值/标准差只fit训练；F和补充块分别除sqrt14，补零至512。真实有效输入14或28维。候选多14个有效系数，不能把全部变化解释成纯信息收益；同维噪声辅助检查。',
'- 只按三种子平均dev AUROC选C，精确平局按网格顺序。训练和选择后封存所有头，再提取其他角色、校准和评分。不按测试挑C、种子或阈值。','',
'|方案|最终C|平均dev AUROC|','|---|---|---|']
for name,s in e['selected'].items():lines.append(f"|{name}|{s['C']}|{s['dev_auroc_mean']:.6f}|")
decisions=[]
for role,label,n0 in [('C','历史C1000',500),('RV','历史RV1082',300),('OR_test','历史OR1500',1000)]:
 data=e['cohorts'][role];b=data['B-F'][0];a=data[arm][0];comp=data['comparison'];lo,hi=comp['ci983333'];verdict='AUROC改善获本次区间支持' if lo>0 else ('AUROC下降获本次区间支持' if hi<0 else 'AUROC差异尚未确认');decisions.append(label+'：'+verdict)
 lines+=['','## '+label,'','主展示为预设首种子20261104；其余种子全部保留。误报分母为'+str(n0)+'条正常输入。','', '|指标|仅原14F|原14F＋C14|','|---|---|---|']
 for key,title in [('auroc','AUROC'),('pauc','低误报pAUC'),('tpr','检出率'),('fpr','误报率')]:
  before,after=(f'{b[key]:.6f}',f'{a[key]:.6f}') if key in ['auroc','pauc'] else (f'{b[key]:.2%}',f'{a[key]:.2%}')
  if (a[key]<b[key] if key=='fpr' else a[key]>b[key]):after='**'+after+'**'
  if key=='fpr':before+=f"（{b['fp']}/{n0}）";after+=f"（{a['fp']}/{n0}）"
  lines.append('|'+title+'|'+before+'|'+after+'|')
 lines += ['',f"F＋C14−仅F的AUROC差：{comp['delta_auroc']:+.6f}；98.333333%区间[{lo:+.6f}, {hi:+.6f}]。{verdict}。",'', '|描述性同维噪声检查|AUROC|低误报pAUC|检出率|误报|','|---|---|---|---|---|']
 for name in ['Noise',arm]:
  m=data[name][0];lines.append(f"|{name}|{m['auroc']:.6f}|{m['pauc']:.6f}|{m['tpr']:.2%}|{m['fpr']:.2%}（{m['fp']}条）|")
 lines+=['','三种子范围（不按测试挑种子）：','', '|方案|AUROC|pAUC|检出率|误报率|','|---|---|---|---|---|']
 for name in ['B-F',arm,'Noise']:
  out=[]
  for key in ['auroc','pauc','tpr','fpr']:
   low,high=min(m[key] for m in data[name]),max(m[key] for m in data[name]);out.append(f'{low:.6f}～{high:.6f}' if key in ['auroc','pauc'] else f'{low:.2%}～{high:.2%}')
  lines.append('|'+name+'|'+'|'.join(out)+'|')
 if role=='OR_test':
  lines+=['','OR来源400条正常样本重新校准（不重训）：','', '|方案|检出率|误报|','|---|---|---|']
  for name in ['B-F',arm,'Noise']:
   m=data[name][0]['source_calibrated'];lines.append(f"|{name}|{m['tpr']:.2%}|{m['fpr']:.2%}（{m['fp']}/1000）|")
lines+=['','## 结论与边界','', '；'.join(decisions)+'。总体AUROC和误报表现分开判断，点值提高不自动表示显著改善。','',
'pAUC为sklearn标准化FPR≤10%部分面积。阈值由独立400条正常cal的第381顺序统计量确定，严格score>threshold。每批10000次分层配对bootstrap，三批主AUROC比较Bonferroni，单区间98.333333%；未校正整个历史多轮探索，也未包含重新采样训练集的不确定性。','',
f"独立核验通过：全部6582条F、训练质心/标准化、27原头、9最终条目与胜出原文件绑定、全部小分数、四指标与阈值，以及3个区间。精确评分重放最大差{p['prediction_max_difference']:.3g}；独立求和实现的余弦评分最大差{p['independent_cosine_prediction_max_difference']:.3g}；基线与EI-V1最大差{e['baseline_reproduction_max_difference']:.3g}。",'',
'[冻结协议](../../docs/ei2-centroid-protocol-2026-11-05.md) · [源码锁](source-lock.json) · [提取记录](extraction.json) · [评分前封印](pre-evaluation-seal.json) · [核验](verification.json) · [完整指标](evaluation.json)','',
f"evaluation SHA256：`{p['evaluation_sha256']}`。模型、质心向量、原文、隐藏状态与凭据不发布；结果只包含自有代码、小分数、ID和元数据。"]
(d/'report-2026-11-05.md').write_bytes(('\n'.join(lines)+'\n').encode());print('VERIFIED_EI_C1_REPORT_RENDERED')
