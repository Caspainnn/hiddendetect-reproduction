"""Render only frozen metric differences; no model fitting or selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out=Path(__file__).resolve().parent
e=json.loads((out/'evaluation.json').read_text())
family=e['contrasts']['confirmation_aegis:global']
names=['mean_vs_B','dynamic_vs_B','dynamic_vs_mean','dynamic_vs_random',
       'processed_vs_B','processed_vs_dynamic','relations_vs_B','relations_vs_processed']
labels=[n.replace('_vs_',' vs ').replace('_',' ') for n in names]
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
for ax,metric,title in zip(axes,['auroc','partial_auroc'],['AUROC difference','pAUC(.1) difference']):
    point=np.array([family[n][metric]['delta'] for n in names])
    ci=np.array([family[n][metric]['family10_ci'] for n in names])
    y=np.arange(len(names));ax.errorbar(point,y,xerr=[point-ci[:,0],ci[:,1]-point],fmt='o',color='#245f9b',capsize=3)
    ax.axvline(0,color='#555555',linestyle='--',linewidth=1);ax.set_title(title)
    ax.grid(axis='x',alpha=.2);ax.set_xlabel('Difference in metric units')
axes[0].set_yticks(np.arange(len(names)),labels);axes[0].invert_yaxis()
fig.suptitle('Fresh 1000: user-signal processing, gamma=1\nPaired bootstrap5000; ten-contrast correction within each metric',fontsize=12)
fig.savefig(out/'processing-contrasts.png',dpi=180)
fig.savefig(out/'processing-contrasts.pdf')
print('FROZEN_CONTRAST_PLOTS_WRITTEN')
