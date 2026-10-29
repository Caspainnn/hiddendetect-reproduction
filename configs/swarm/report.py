"""Tables/figures only, after immutable evaluation; never selects a model."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'results/swarm';r=json.loads((out/'evaluation.json').read_text());f=json.loads((out/'freeze.json').read_text());runs=json.loads((out/'search.json').read_text())
    lines=['# All unified-benchmark and confirmation results','',
        'The benchmark was previously observed. Fresh confirmation is separate and same-source, not official-test or unseen-domain evidence. All48 candidates retained.','',
        'pAUC is sklearn standardized partial AUROC over FPR0–10%; thresholds are independent empirical safe95% quantiles, not retuned to test FPR5%.',
        'Dimension column counts selected extra inputs; anchor_L1 instead records total nonzero coefficients including B and still reads all inputs.','']
    for condition,values in r['tests'].items():
        lines+=['## '+condition,'','|Candidate|AUROC|pAUC(.1) [95% CI]|Recall %|FPR %|Extra dimensions|',
            '|---|---:|---:|---:|---:|---:|']
        for c in f['candidates']:
            n=c['name'];v=values[n];lo,hi=v['ci95']['partial_auroc'];dims=c.get('dimensions','reference')
            lines.append(f"|{n}|{v['auroc']:.5f}|{v['partial_auroc']:.5f} [{lo:.5f}, {hi:.5f}]|{100*v['tpr']:.2f}|{100*v['fpr']:.2f}|{dims}|")
        lines+=['','### Equal-budget method means','',
            '|Method|AUROC mean ± seed SD|pAUC mean ± seed SD|Recall % mean|FPR % mean|pAUC delta vs random [conditional paired95% CI]|',
            '|---|---:|---:|---:|---:|---:|']
        for m in ('random','bpso','ga','de'):
            a=r['method_comparison'][condition][m];p=a['partial_auroc'];delta=p['vs_random'];lo,hi=delta['paired_ci95']
            lines.append(f"|{m}|{a['auroc']['mean']:.5f} ± {a['auroc']['seed_std']:.5f}|{p['mean']:.5f} ± {p['seed_std']:.5f}|{100*a['tpr']['mean']:.2f}|{100*a['fpr']['mean']:.2f}|{delta['mean_delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]|")
        lines+=['']
    (out/'all-results.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf8',newline='\n')
    with (out/'all-results.csv').open('w',encoding='utf8',newline='') as stream:
        w=csv.writer(stream);w.writerow(['condition','candidate','AUROC','pAUC0.1','AP','TPR','FPR','TP','FN','FP','TN'])
        for condition,values in r['tests'].items():
            for name,v in values.items():w.writerow([condition,name,*[v[k] for k in ('auroc','partial_auroc','average_precision','tpr','fpr','tp','fn','fp','tn')]])
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained');colors=['#777777','#0072b2','#009e73','#d55e00']
    for m,color in zip(('random','bpso','ga','de'),colors):
        curves=np.array([x['best_history'] for x in runs if x['method']==m]);mean=curves.mean(0);sd=curves.std(0)
        axes[0].plot(np.arange(1,121),mean,label=m,color=color);axes[0].fill_between(np.arange(1,121),mean-sd,mean+sd,alpha=.15,color=color)
    axes[0].set_xlabel('Unique development evaluations');axes[0].set_ylabel('Best development fitness');axes[0].set_title('Mean5 seeds ± SD');axes[0].legend()
    for ax,role,title in [(axes[1],'test_aegis_holdout:global','Fixed public benchmark (n=1000)'),(axes[2],'confirmation_aegis:global','Fresh confirmation (n=400)')]:
        for j,(m,color) in enumerate(zip(('random','bpso','ga','de'),colors)):
            entry=r['method_comparison'][role][m]['partial_auroc'];ax.errorbar(j,entry['mean'],yerr=entry['seed_std'],fmt='o',color=color,capsize=4)
        for name,style in [('expert_B','--'),('linear_BPAT',':'),('anchor_L1','-.')]:ax.axhline(r['tests'][role][name]['partial_auroc'],ls=style,lw=1,label=name)
        ax.set_xticks(range(4),['random','BPSO','GA','DE']);ax.set_ylabel('Standardized partial AUROC0–10% FPR');ax.set_title(title);ax.legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'swarm-comparison.png',dpi=180);plt.close(fig)
    summary={'best_dev_method':f['best_dev_method'],'method_comparison':r['method_comparison'],
        'sparsity':{m:[{'seed':i,'dimensions':next(c for c in f['candidates'] if c['name']==m+'_seed'+str(i))['dimensions']} for i in range(5)] for m in ('random','bpso','ga','de')},
        'budget_runs':[{k:x[k] for k in ('method','seed','unique_evaluations','proposal_count','duplicate_count','diversity_restarts','seconds','incremental_new_fits')} for x in runs]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8',newline='\n');print('TABLES_AND_FIGURE_WRITTEN')
    groups=json.loads((ROOT/'results/highdim/schema.json').read_text())
    selected={c['name']:[g for take,g in zip(c['mask'],groups) if take] for c in f['candidates'] if c['kind']=='anchored' and c['name'].split('_seed')[0] in ('random','bpso','ga','de')}
    (out/'selected-groups.json').write_text(json.dumps({'scope':'selection frequency is diagnostic, not causal importance','candidates':selected},indent=2)+'\n',encoding='utf8',newline='\n')

if __name__=='__main__':main()
