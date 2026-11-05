"""CPU-only independent checks and concise experiment report from frozen outputs."""
import ast,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from core import INTERVALS,auc,choose,interval_scores,select
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/layer-selection'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=json.loads((OUT/'manifest.json').read_text());e=json.loads((OUT/'evaluation.json').read_text())
 lock=json.loads((OUT/'run-lock.json').read_text());seal=json.loads((OUT/'pre-evaluation-seal.json').read_text())
 completion=json.loads((OUT/'completion.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
 assert completion['status']=='completed'
 for n in ['scores','evaluation']:assert sha(OUT/(n+'.jsonl' if n=='scores' else n+'.json'))==completion[n+'_sha256']
 assert sha(OUT/'manifest.json')==lock['manifest_sha256']==e['manifest_sha256']
 assert sha(OUT/'selection.json')==seal['selection_sha256']==e['selection_sha256']
 assert sha(OUT/'run-lock.json')==seal['run_lock_sha256']
 for name,h in lock['sources'].items():assert sha(ROOT/name)==h,name
 assert json.loads((OUT/'identity.json').read_text())['maximum_F_difference']==0
 audit=json.loads((OUT/'final-audit.json').read_text())
 assert audit['same_full_history_fuzzy_group_audit_pass'] and audit['unique_global_groups']==3256
 assert audit['dependency_bytes_match_pre_run_head']
 assert sha(OUT/'model-lock.json')==lock['model_lock_sha256']
 with (OUT/'scores.jsonl').open(encoding='utf-8') as stream:rows=[json.loads(x) for x in stream]
 assert len(rows)==3256 and len({r['id'] for r in rows})==len({r['group_id'] for r in rows})==3256
 assert [{k:v for k,v in r.items() if k!='F'} for r in rows]==m['samples']
 assert Counter(r['role'] for r in rows)=={'sel':256,'cal':400,'check':600,'test':2000}
 tests=[r for r in rows if r['role']=='test'];y=np.array([r['label'] for r in tests])
 assert Counter(y)=={0:1000,1:1000}
 sel=[r for r in rows if r['role']=='sel'];sy=np.array([r['label'] for r in sel]);sf=np.array([r['F'] for r in sel])
 assert select(sf,sy,[r['group_id'] for r in sel])==selection
 ss=interval_scores(sf)
 for j in range(496):assert abs(roc_auc_score(sy,ss[:,j])-selection['full_selection_auroc'][j])<1e-12
 rng=np.random.default_rng(selection['seed']);wins=Counter();layer_counts=np.zeros(32,dtype=int)
 classes=[np.flatnonzero(sy==k) for k in (0,1)]
 for _ in range(100):
  shuffled=[rng.permutation(ix) for ix in classes]
  for side in (0,1):
   ix=np.concatenate([v[side*(len(v)//2):(side+1)*(len(v)//2)] for v in shuffled])
   objective=auc(sy[ix],ss[ix]);a,b=INTERVALS[choose(objective,objective)]
   wins[f'{a}:{b}']+=1;layer_counts[a:b+1]+=1
 (OUT/'stability-diagnostics.json').write_text(json.dumps({'half_sample_winner_intervals':dict(wins),'layer_inclusion_frequency':(layer_counts/200).tolist(),'note':'Descriptive half-sample maxima; these frequencies do not select B.'},indent=2)+'\n')
 allscores=[]
 for n,(a,b) in selection['intervals'].items():
  s=np.array([np.trapz(np.array(r['F'])[a:b+1]) for r in tests])
  assert abs(roc_auc_score(y,s)-e['auroc'][n])<1e-12
  assert np.isfinite(s).all()
  allscores.append(s)
  cal=np.array([np.trapz(np.asarray(r['F'])[a:b+1]) for r in rows if r['role']=='cal'])
  threshold=float(np.sort(cal)[380]);secondary=e['secondary'][n]
  assert threshold==secondary['threshold_safe400_ordinal381']
  assert float((s[y==1]>threshold).mean())==secondary['tpr']
  assert float((s[y==0]>threshold).mean())==secondary['fpr']
 # Independent weighted ROC algorithm from the original-position evaluator.
 tree=ast.parse((ROOT/'configs/original-positions/core.py').read_text(encoding='utf-8'))
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bootstrap_metrics')
 ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent-weighted-roc','exec'),ns)
 rng=np.random.default_rng(e['seed']);classes=[np.flatnonzero(y==k) for k in (0,1)]
 indices=np.array([np.concatenate([rng.choice(v,len(v),replace=True) for v in classes]) for _ in range(10000)])
 distribution=np.empty((10000,4));names=['B0','B1','A','B']
 for start in range(0,10000,100):
  for j,s in enumerate(allscores):distribution[start:start+100,j]=ns['bootstrap_metrics'](y,s,indices[start:start+100])[0]
 for n,v in e['contrasts'].items():
  a,b=n.split('-');delta=distribution[:,names.index(a)]-distribution[:,names.index(b)]
  np.testing.assert_allclose(np.quantile(delta,[.005,.995]),v['ci99'],atol=1e-12,rtol=0)
 table=['|方法|固定区间（0基，含端点）|新确认集 AUROC|历史 dev AUROC|', '|---|---|---:|---:|']
 for n,ab in e['intervals'].items():table.append(f"|{n}|{ab}|{e['auroc'][n]:.6f}|{e['secondary'][n]['historical_check_auroc']:.6f}|")
 contrasts=['|配对比较|AUROC 差值|99% 区间|结论|','|---|---:|---|---|']
 for n,v in e['contrasts'].items():
  lo,hi=v['ci99'];verdict='正增益得到支持' if lo>0 else '下降得到支持' if hi<0 else '差异尚未确认'
  contrasts.append(f"|{n}|{v['delta']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|{verdict}|")
 report='\n'.join(['# 固定连续区间选层实验','', '主指标：总体 AUROC。位置固定为原模板末位，原 F 与梯形积分保持一致。', '', 'B0：默认 [16,29]；B1：同一256条选择集上的原 FDV 端点法；A：496区间直接最大化选择集 AUROC；B：100对互补分层半样本的第10百分位 AUROC 最大。', '', '选择集256条（128/类）、历史诊断600条、safe校准400条、新同源确认2000条（1000/类）。实际历史去重记录数：'+str(m['history_count'])+'。新样本的身份与近重复审计覆盖服务器主目录及最新 OC 工作树。', '', *table, '', *contrasts, '', '区间来自共享分层配对 bootstrap 10000次，每个比较给双侧99%百分位区间，对预定5个比较作近似家族95%控制。区间含零表示尚未确认差异。', '', '8条新提取原 F 与历史精确原规则缓存逐值一致（最大差0）。模型文件逐个校验 SHA256。历史 dev 只作诊断，选择结果在新确认评分前封存。全部3256条保存32层小分数，无提示词公开输出。', '', '局限：确认集来自 Aegis train 中此前未使用的样本；它支持同源结论，未验证新领域泛化。FDV 使用同等256条文本选择集，属于原端点规则的同预算比较，不能称为原12条多模态校准复现。', '', '次要 pAUC、冻结阈值 TPR/FPR 见 evaluation.json；总体 AUROC 为本轮主判据。'])+'\n'
 (OUT/'report-2026-10-30.md').write_text(report,encoding='utf-8')
 (OUT/'verification.json').write_text(json.dumps({'status':'passed','rows':3256,'independent_sklearn_auroc':True,'selectors_reproduced':True,'independent_weighted_roc_bootstrap_ci':True,'thresholds_recomputed':True,'group_role_disjoint':True,'frozen_source_hashes':True,'verifier_sha256':sha(Path(__file__))},indent=2)+'\n')
 print(report)
if __name__=='__main__':main()
