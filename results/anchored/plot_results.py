"""Scientific plot from frozen test output only; no fitting or selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

directory=Path(__file__).resolve().parent;e=json.loads((directory/'evaluation.json').read_text())
keys=['confirmation_aegis:global','confirmation_toxic:source'];labels=list(e['contrasts'][keys[0]])
short=['Gated dynamic - B','Gated - global dynamic','Gated dynamic - gated mean','Gated dynamic - gated random','Global dynamic - B','Gated mean - B','Gated dynamic - B refit2200']
fig,axes=plt.subplots(1,2,figsize=(12,5.1),sharey=True,layout='constrained')
for ax,key,title in zip(axes,keys,('Fresh Aegis harmfulness (n=1000)','Fresh ToxicChat toxicity (n=150), source calibration')):
    ax.axvline(0,color='#7c858b',ls='--',lw=1)
    for i,label in enumerate(labels):
        v=e['contrasts'][key][label]['partial_auroc'];lo,hi=v['family7_ci'];color='#287568' if i!=6 else '#356ca1'
        ax.plot([lo,hi],[i,i],color=color,lw=2);ax.scatter(v['delta'],i,color=color,s=34,zorder=3)
    ax.set_title(title,fontsize=10);ax.set_xlabel('Paired pAUC(.1) difference');ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(np.arange(len(labels)),short);axes[0].invert_yaxis()
fig.suptitle('R28-R29: prespecified anchored-correction comparisons',fontsize=13)
fig.supxlabel('Seven-contrast corrected paired bootstrap intervals, separately per task/metric.\nConditional on fixed training and calibration; toxicity and harmfulness are different tasks.',fontsize=9)
fig.savefig(directory/'anchored-contrasts.png',dpi=190);fig.savefig(directory/'anchored-contrasts.pdf');print('PLOT_WRITTEN')
