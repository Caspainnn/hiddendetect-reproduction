"""Plot six prespecified paired contrasts; no fitting or candidate selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

directory=Path(__file__).resolve().parent
e=json.loads((directory/'evaluation.json').read_text())
contrasts=e['contrasts']['confirmation_aegis:global']
labels=list(contrasts);y=np.arange(len(labels))
fig,axes=plt.subplots(1,2,figsize=(12,4.7),sharey=True,layout='constrained')
for ax,metric,title in zip(axes,('partial_auroc','auroc'),('Low-FPR pAUC difference','AUROC difference')):
    ax.axvline(0,color='#7b858c',lw=1,ls='--')
    for i,label in enumerate(labels):
        v=contrasts[label][metric];low,high=v['family6_ci'];color='#267463' if label.startswith('ridge') else '#356fa5'
        ax.scatter(v['draw_deltas'],i+np.linspace(-.13,.13,5),s=15,color='#afb6bb',zorder=2)
        ax.plot([low,high],[i,i],color=color,lw=2)
        ax.scatter([v['delta']],[i],s=38,color=color,zorder=3)
    ax.set_title(title,fontsize=11);ax.set_xlabel('Improvement over paired control');ax.grid(axis='x',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(y,[v.replace('logistic_','LR: ').replace('ridge_','Ridge: ').replace('_vs_',' vs ') for v in labels]);axes[0].invert_yaxis()
fig.suptitle('R27: fresh confirmation, five fixed overlapping training draws',fontsize=13)
fig.supxlabel('Dots: mean metric difference; bars: six-contrast corrected paired bootstrap intervals.\nGrey dots: individual draws. Conditional on this training pool and calibration; not ensemble performance.',fontsize=9)
fig.savefig(directory/'stability-contrasts.png',dpi=190)
fig.savefig(directory/'stability-contrasts.pdf')
print('PLOT_WRITTEN')
