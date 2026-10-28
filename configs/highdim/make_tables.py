"""Render every frozen result and a compact scientific figure; no model selection."""
import json
import os
from pathlib import Path
import numpy as np
from features import ROOT

out=ROOT/'results/highdim'
freeze=json.loads((out/'freeze.json').read_text()); evaluation=json.loads((out/'evaluation.json').read_text())
lines=['# 全部冻结结果：高维信息与分组选择', '',
    '47个候选全部列出；TPR为危险／毒性召回，FPR为安全／非毒性误报。校准400条安全输入，严格score>阈值；测试不再调整阈值。AUROC区间为500次按标签分层的配对prompt-group bootstrap，以固定训练、选择和校准为条件，未做多重比较校正。毒性任务单独报告。', '']
for role,title in [('test_aegis','Aegis：500条新测试'),('test_toxicchat','ToxicChat：200条新毒性辅助测试')]:
    lines += ['## '+title,'','|候选|组数|标量维数|AUROC [95% CI]|AP|TPR|实际FPR|','|---|---:|---:|---:|---:|---:|---:|']
    for c in freeze['candidates']:
        v=evaluation['tests'][role][c['name']]; lo,hi=v['ci95']['auroc']
        lines.append('|%s|%s|%s|%.5f [%.5f, %.5f]|%.4f|%.1f%%|%.1f%%|'%(c['name'],c['groups'],c['dimensions'],
            v['auroc'],lo,hi,v['average_precision'],100*v['tpr'],100*v['fpr']))
    lines += ['','|五种子全部|平均AUROC ± 种子样本标准差|平均TPR|平均FPR|','|---|---:|---:|---:|']
    for head in ('linear','interaction'):
        for method in ('random','bpso'):
            values=[evaluation['tests'][role]['%s_%s_seed%d'%(head,method,seed)] for seed in range(5)]
            lines.append('|%s_%s|%.5f ± %.5f|%.2f%%|%.2f%%|'%(head,method,np.mean([v['auroc'] for v in values]),
                np.std([v['auroc'] for v in values],ddof=1),100*np.mean([v['tpr'] for v in values]),100*np.mean([v['fpr'] for v in values])))
    lines.append('')
lines += ['## 搜索实际计数','','所有运行至多100个不同掩码；旧6次为串行，恢复后为8个独立拟合线程，每个估计器单线程。秒数受共享缓存、恢复状态及线程数影响；fit_seconds是并发任务各自墙钟时间之和，不是独立优化器速度比较。','','|头／方法／种子|不同掩码|实际目标调用|新拟合|缓存命中|增量秒数|旧完成运行恢复|','|---|---:|---:|---:|---:|---:|---|']
for run in json.loads((out/'search.json').read_text()):
    lines.append('|%s/%s/%d|%d|%d|%d|%d|%.2f|%s|'%(run['head'],run['method'],run['seed'],run['unique_evaluations'],
        run['objective_calls'],run['new_fits'],run['cache_hits'],run['seconds'],run.get('resumed_completed_run',False)))
(out/'all-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
test=evaluation['tests']['test_aegis']
def bars(ax,names,labels,title):
    values=np.array([test[name]['auroc'] for name in names]); ci=np.array([test[name]['ci95']['auroc'] for name in names])
    ax.bar(np.arange(len(names)),values,color='#4682a9',width=.65)
    ax.errorbar(np.arange(len(names)),values,yerr=np.maximum(0,np.stack([values-ci[:,0],ci[:,1]-values])),fmt='none',ecolor='#263746',capsize=3)
    ax.set_xticks(np.arange(len(names)),labels,rotation=20,ha='right'); ax.set_ylim(.45,1.0); ax.set_ylabel('Held-out AUROC'); ax.set_title(title); ax.grid(axis='y',alpha=.2)
bars(axes[0,0],['raw_ridge_last','raw_ridge_user','raw_ridge_positions'],['Last','User regions','Last + regions'],'Unprojected vectors: fixed ridge head')
bars(axes[0,1],['linear_last_vector','linear_position_vectors','linear_position_trajectory','linear_all','interaction_all'],
    ['Last / linear','Positions / linear','+ trajectories','All / linear','All / trees'],'Projected/summary information and fixed heads')
for ax,head in zip(axes[1],('linear','interaction')):
    for x,method,color in [(0,'random','#4682a9'),(1,'bpso','#d17a22')]:
        values=[test['%s_%s_seed%d'%(head,method,seed)]['auroc'] for seed in range(5)]
        ax.scatter(x+np.linspace(-.08,.08,5),values,color=color,s=50,zorder=3)
        ax.plot([x-.15,x+.15],[np.mean(values)]*2,color=color,lw=2)
    ax.axhline(test[head+'_all']['auroc'],color='#444',linestyle='--',label='Full pool, same head')
    ax.set_xticks([0,1],['Random: all 5 seeds','BPSO: all 5 seeds']); ax.set_ylim(.65,1.0); ax.set_ylabel('Held-out AUROC')
    ax.set_title(head+' head: equal 100-mask budgets'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
fig.suptitle('Fresh Aegis test (n=500); bootstrap CIs conditional on fitted pipeline; seed dots share the same test',fontsize=12)
fig.savefig(out/'information-and-search.png',dpi=180); plt.close(fig)
print('All47 tables and figure generated')
