"""Reporting only; no candidate selection or refitting."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'results/positions';f=json.loads((out/'freeze.json').read_text());r=json.loads((out/'evaluation.json').read_text());lines=['# All actual-position results','',
        '17 predefined heads, same ridge and full32x4096 vectors. CV selection uses train1600 only. Benchmark is historical; fresh confirmation is separate.','',
        'AUROC bounds are conditional paired bootstrap Bonferroni upper bounds for12 fixed real alternatives. Margin=.005 is an engineering tolerance, not a universal standard.','']
    for condition,values in r['tests'].items():
        lines+=['## '+condition,'','|Readout|AUROC|pAUC(.1)|Recall %|FPR %|Delta AUROC [paired95% CI]|Simultaneous one-sided upper delta|',
            '|---|---:|---:|---:|---:|---:|---:|']
        base=values['template_last']['auroc'];bounds=r['restricted_optimality'][condition]['upper_delta_auroc']
        for c in f['candidates']:
            n=c['name'];v=values[n];lo,hi=v['paired_delta_vs_template_ci95']['auroc'];upper=f"{bounds[n]:+.5f}" if n in bounds else 'control'
            lines.append(f"|{n}|{v['auroc']:.5f}|{v['partial_auroc']:.5f}|{100*v['tpr']:.2f}|{100*v['fpr']:.2f}|{v['auroc']-base:+.5f} [{lo:+.5f}, {hi:+.5f}]|{upper}|")
        lines+=['',f"Restricted margin criterion supported: **{r['restricted_optimality'][condition]['supports_margin']}**. This is not a global-optimality proof.",'']
    (out/'all-results.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf8',newline='\n')
    with (out/'all-results.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['condition','candidate','auroc','partial_auroc','average_precision','tpr','fpr','tp','fn','fp','tn'])
        for condition,values in r['tests'].items():
            for n,v in values.items():writer.writerow([condition,n,*[v[k] for k in ('auroc','partial_auroc','average_precision','tpr','fpr','tp','fn','fp','tn')]])
    single=[c for c in f['candidates'] if len(c['positions'])==1];labels=[c['name'].replace('user_','').replace('template_','tpl_') for c in single]
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained');x=np.arange(len(single))
    axes[0,0].errorbar(x,[c['cv_pauc_mean'] for c in single],yerr=[c['cv_pauc_std'] for c in single],fmt='o-',capsize=3)
    axes[0,0].set_title('Train CV pAUC: mean10 folds ± SD (not CI)');axes[0,0].set_ylabel('Standardized pAUC0–10% FPR')
    for ax,condition,title in [(axes[0,1],'test_aegis_holdout:global','Fixed benchmark (n=1000)'),(axes[1,0],'confirmation_aegis:global','Fresh confirmation (n=1000)')]:
        values=r['tests'][condition];mean=np.array([values[c['name']]['auroc'] for c in single]);ci=np.array([values[c['name']]['ci95']['auroc'] for c in single])
        ax.errorbar(x,mean,yerr=np.maximum(0,[mean-ci[:,0],ci[:,1]-mean]),fmt='o-',capsize=3)
        ax.set_title(title);ax.set_ylabel('AUROC ± bootstrap95% CI')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_xticks(x,labels,rotation=35,ha='right');ax.grid(alpha=.2)
    bounds=r['restricted_optimality']['confirmation_aegis:global'];names=list(bounds['upper_delta_auroc']);y=np.arange(len(names))
    axes[1,1].scatter([bounds['upper_delta_auroc'][n] for n in names],y)
    axes[1,1].axvline(.005,color='r',ls='--',label='predeclared margin .005')
    axes[1,1].set_yticks(y,[n.replace('template_plus_user_','tpl+').replace('user_','') for n in names]);axes[1,1].invert_yaxis()
    axes[1,1].set_xlabel('Simultaneous upper bound on deltaAUROC vs tpl-last');axes[1,1].set_title('Restricted near-optimality check');axes[1,1].grid(alpha=.2);axes[1,1].legend(fontsize=8)
    fig.savefig(out/'token-position-comparison.png',dpi=180);plt.close(fig)
    summary={'selection':f['selection'],'cv':[{'name':c['name'],'mean':c['cv_pauc_mean'],'std':c['cv_pauc_std']} for c in f['candidates']],
        'restricted_optimality':r['restricted_optimality'],'old_B_parameters':f['old_B_parameter_max_differences'],
        'old_B_scores_max_diff':r['maximum_old_B_score_difference'],'duplicate_scores_max_diff':r['maximum_duplicate_score_difference']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8',newline='\n');print('POSITION_TABLES_AND_FIGURE_WRITTEN')

if __name__=='__main__':main()
