"""Render the verified historical diagnostic without fitting or choosing anything."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];D=R/'results/ei2-pca-joint'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((D/'final-verification.json').read_text());assert p['passed']
for name,h in p['files'].items():assert sha(D/name)==h
e=json.loads((D/'evaluation.json').read_text());families=list(e['controls'])
labels={'Q-CV':'仅原PCA前缀CV','Q-VAR':'仅解释率','Q-GATE':'仅原顺序门控','S-EN':'仅EN筛选','S-L1':'仅L1筛选','ALL':'保留全部方向','B-F':'仅原14F','same-tau-pool':'与组合2同tau的完整池','ranked-gate-zero':'EN重排＋零惩罚门'}
lines=['# EI-PJ1：同时自动决定方向数量与具体方向','',
'本轮已完成588次LR拟合、18次稀疏选择器训练、12次GPU门训练，并完成独立重算及30个最终头与原试验文件的严格绑定。原14F、14层（0基16～29）、模板末位、训练标签与最终L2逻辑回归保持一致。特征变换及筛选只在训练侧确定；验证集选择超参数，历史测试不参与选择。', '',
'**结论：组合2最值得继续做独立确认，但不能称为已经普遍优于两个单组件。** 它在历史C/RV上优于完整空间L1获得区间支持，相对仅解释率的提升尚未确认；在历史OR上同时超过两组件获得区间支持。组合1未在三批数据上稳定超过两组件；组合3在OR上显著差于原顺序门控。不能依据这些测试表现重新选超参数或改写主方案。','',
'全部数据均为之前已观察过的历史诊断，不是新的独立确认。本轮先冻结源码和训练协议，再封存搜索、方向掩码和模型，最后评分。与单组件的额外搜索预算不同，不能把变化全归因于组合本身。','',
'## 实际方法与选择','',
'|编号|大白话方法|验证集选中的设置|保留方向数（三种子）|','|---|---|---|---|']
methods=['先按EN用途重排，再用折内CV与one-SE决定每层留几个','先按解释率缩候选池，再让L1筛具体方向','先按EN用途重排，再让每层门控自己决定停止位置']
for n,m in zip(families,methods):
 s=e['selected'][n];lines.append(f"|{n}|{m}|{s['variant']['key']}；最终C={s['C']}|{[sum(x) for x in s['counts_by_seed']]}|")
lines+=['','完整训练谱每层1599个数值方向；CV折内为1065／1066。不是Transformer层数增加。排名按逐层标准化系数绝对值，平局原坐标优先；包括零权重方向。最终统一映射回原PCA坐标升序，丢弃临时头与软门，再训练原L2头。原坐标及每层数量保留在[evaluation.json](evaluation.json)。','']
v=e['selected'][families[1]]['variant'];lines += [f"组合2选τ={v['tau']}：筛前14层覆盖范围{min(v['pool_coverage']):.2%}～{max(v['pool_coverage']):.2%}，筛后{min(v['selected_coverage']):.2%}～{max(v['selected_coverage']):.2%}。分母均为原残差总能量，缩池达到80%不表示筛后仍达到80%。",'']
def f(m):return f"{m['auroc']:.6f}|{m['pauc']:.6f}|{m['tpr']:.2%}|{m['fpr']:.2%}（{m['fp']}条）"
for role,title in [('C','历史C1000'),('RV','历史RV1082'),('OR_test','历史OR-Bench1500')]:
 d=e['cohorts'][role];lines += ['## '+title,'','主展示为预定种子20261104；阈值均由独立的400条正常校准样本取同一分位数，使用严格大于阈值判阳性。pAUC为sklearn标准化FPR≤10%部分面积。误报分母依次为500、300、1000。','', '|方案|AUROC|低误报pAUC|检出率|误报|','|---|---|---|---|---|']
 for n in ['B-F','ALL','Q-CV','Q-VAR','Q-GATE','S-EN','S-L1']+families+[n+'-prefix' for n in families]+['same-tau-pool','ranked-gate-zero']:
  lines.append('|'+labels.get(n,n)+'|'+f(d[n][0])+'|')
 lines+=['','### 六项主比较','', '|组合|直接对照|AUROC变化|差异99.166667%区间|判断|','|---|---|---|---|---|']
 for n in families:
  for control in e['controls'][n]:
   c=d['comparisons'][n][control];lo,hi=c['ci991667'];b=d[control][0]['auroc'];a=d[n][0]['auroc'];val=f'**{a:.6f}**' if a>b else f'{a:.6f}';decision='支持改善' if lo>0 else ('支持下降' if hi<0 else '尚未确认差异')
   lines.append(f"|{n}|{labels[control]}|{b:.6f} → {val}|[{lo:+.6f}, {hi:+.6f}]|{decision}|")
 lines+=['','每批使用相同10000次分层配对bootstrap重采样；六项比较作Bonferroni调整，每个区间99.166667%。此校正未覆盖跨批次、历史多轮探索；仅作历史诊断。同数量前缀、同tau完整池、ALL、BF与零惩罚门仅为描述性检查，不参与主区间判定。','', '### 三种子的稳定性（范围）','', '|组合|AUROC范围|pAUC范围|检出率范围|误报率范围|','|---|---|---|---|---|']
 for n in families:
  ms=d[n];values=[]
  for key in ['auroc','pauc','tpr','fpr']:
   lo,hi=min(m[key] for m in ms),max(m[key] for m in ms);values.append(f'{lo:.6f}～{hi:.6f}' if key in ['auroc','pauc'] else f'{lo:.2%}～{hi:.2%}')
  lines.append('|'+n+'|'+'|'.join(values)+'|')
 if role=='OR_test':
  lines+=['','### 使用OR来源400条正常样本重新校准（不重训）','', '|方案|检出率|误报|','|---|---|---|']
  for n in ['Q-CV','Q-VAR','Q-GATE','S-EN','S-L1']+families:
   m=d[n][0]['source_calibrated'];lines.append(f"|{labels.get(n,n)}|{m['tpr']:.2%}|{m['fpr']:.2%}（{m['fp']}/1000）|")
lines+=['','## 预算、复核与限制','',
'CV数量网格44项×3折×3C＝396个头；最终候选及诊断192个头，总588。EN排名器λ=.0001、ρ=.5固定；EN/L1单组件各自搜索3λ，最终C均在0.1、1、10选择。池3τ×3λ，门3惩罚×3种子另加零惩罚诊断。主选择只按三种子平均验证AUROC，精确平局先更少方向再候选顺序。CV每折重拟合排序和标准化；PCA复用经索引与hash核验的父训练基底。', '',
'18个选择器KKT检查、396个CV头、192个最终候选头、12个GPU门重放、全部预测/阈值/四项指标、18个主区间已独立核验。追加检查将30个最终条目与胜出原文件的头、系数、标准化与坐标逐项绑定，重新计算验证AUROC。CPU篡改测试能拒绝同维度系数、标准化和坐标变更。', '',
'[逐层零权重比例](selector-audit.json)同时列出严格零和绝对值≤1e-8的比例；两者不可混用。[门数值审计](gate-numerics-audit.json)保留浮点乘积误差界和临界坐标记录；不靠放宽误差重选掩码。', '',
'独立只读审查提出“最终头与胜出文件绑定”及“补逐层零比例”两项缺口，已追加只读审计补齐。冻结训练源码与原复核未覆盖；追加文件见[说明](../../docs/ei2-pca-joint-audit-2026-11-05.md)。组合2虽在三批AUROC点值均超过两对照，但检出率相对仅解释率均略降；不能只报AUROC。组合3不能据较少误报掩盖其OR检出率与AUROC下降。', '',
'[冻结协议](../../docs/ei2-pca-joint-protocol-2026-11-05.md) · [训练源码锁](source-lock.json) · [评分前封印](pre-evaluation-seal.json) · [原复核](verification.json) · [最终发布门槛](final-verification.json) · [模型绑定审计](selected-entry-audit.json)','',
f"evaluation SHA256：`{p['evaluation_sha256']}`；scores SHA256：`{p['scores_sha256']}`。公开内容只含自有脚本、小分数、ID、元数据与报告；训练模型、原始文本和凭据留在本地／服务器。"]
(D/'report-2026-11-05.md').write_bytes(('\n'.join(lines)+'\n').encode())
print('VERIFIED_JOINT_REPORT_RENDERED')
