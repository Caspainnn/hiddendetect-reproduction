"""Generate complete tables and a static scientific figure from sealed metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]

def main():
    out=ROOT/'results/fusion'; r=json.loads((out/'evaluation.json').read_text()); f=json.loads((out/'freeze.json').read_text())
    lines=['# All frozen conditional-fusion results','',
        'Thresholds: global = independent Aegis safe400; source = independent ToxicChat non-toxic400. Strict score > threshold. Toxicity proxy is separate from harmful-input detection.','',
        'All23 candidates retained; best-dev additive subset: `'+f['best_dev_additive_subset']+'`. CIs conditional on frozen training/calibration; no multiplicity correction.','']
    for test,values in r['tests'].items():
        lines+=['## '+test,'','|Candidate|AUROC [95% CI]|AP|TPR %|FPR %|Delta AUROC vs B [paired95% CI]|',
            '|---|---:|---:|---:|---:|---:|']
        for c in f['candidates']:
            name=c['name']; v=values[name]; lo,hi=v['ci95']['auroc']; dl,dh=v['paired_delta_ci95']['expert_B']['auroc']
            delta=v['auroc']-values['expert_B']['auroc']
            lines.append(f"|{name}|{v['auroc']:.5f} [{lo:.5f}, {hi:.5f}]|{v['average_precision']:.5f}|{100*v['tpr']:.2f}|{100*v['fpr']:.2f}|{delta:+.5f} [{dl:+.5f}, {dh:+.5f}]|")
        lines+=['']
    (out/'all-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    names=['expert_B','expert_P','expert_A','expert_T','linear_BPAT','pairwise_BPAT',f['best_dev_additive_subset'],'equal_BPAT','raw_ridge_positions','linear_FC64']
    names=list(dict.fromkeys(names)); fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    values=r['tests']['test_aegis_holdout:global']; y=np.arange(len(names))
    auc=np.array([values[n]['auroc'] for n in names]); ci=np.array([values[n]['ci95']['auroc'] for n in names])
    axes[0].errorbar(auc,y,xerr=[auc-ci[:,0],ci[:,1]-auc],fmt='o',capsize=3)
    axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].set_xlabel('AUROC (paired bootstrap95% intervals)');axes[0].set_title('Fresh Aegis prospective holdout (n=1000)')
    selected=['expert_B','linear_BPAT','pairwise_BPAT',f['best_dev_additive_subset']];selected=list(dict.fromkeys(selected))
    for n in selected:
        a=r['tests']['test_toxic_holdout:global'][n];b=r['tests']['test_toxic_holdout:source'][n]
        axes[1].plot([100*a['fpr'],100*b['fpr']],[100*a['tpr'],100*b['tpr']],'-o',label=n)
        axes[1].annotate('source',(100*b['fpr'],100*b['tpr']),fontsize=8)
    axes[1].set_xlabel('Realized false-positive rate (%)');axes[1].set_ylabel('Recall (%)');axes[1].set_title('Toxicity proxy: global to source threshold (n=400)');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'conditional-fusion.png',dpi=180);plt.close(fig)
    print('TABLES_AND_FIGURE_WRITTEN')

if __name__=='__main__':main()
