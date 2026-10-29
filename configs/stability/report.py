"""Generate tables only from frozen, replayable output; conclusions added after review."""
import json
from stability_core import ROOT

def main():
    out=ROOT/'results/stability';e=json.loads((out/'evaluation.json').read_text());m=json.loads((ROOT/'configs/stability/manifest.json').read_text());cv=json.loads((out/'cv.json').read_text())
    lines=['# R27 Training stability: frozen result tables','','Five overlapping balanced training subsets of 1600 from a 3200 pool; 40 models, 3600 inner fold fits.','Fresh confirmation is a new same-source holdout. Old fixed C and Toxic are historical comparisons.','Intervals condition on these five fixed training draws and calibration. They are not training-population confidence intervals.','','## Training overlap','',str(m['overlap']),'']
    for key,agg in e['aggregate'].items():
        lines+=['## '+key,'','|Variant|Mean AUROC|Mean pAUC(.1)|Mean recall|Mean FPR|','|---|---:|---:|---:|---:|']
        for n,v in agg.items():lines.append(f"|{n}|{v['auroc']:.6f}|{v['partial_auroc']:.6f}|{v['tpr']:.1%}|{v['fpr']:.1%}|")
        lines+=['','|Contrast|Mean delta pAUC|Six-contrast corrected interval|Positive draws /5|Mean delta AUROC|Corrected interval|','|---|---:|---|---:|---:|---|']
        for n,v in e['contrasts'][key].items():
            p=v['partial_auroc'];a=v['auroc'];fmt=lambda x:'['+', '.join(f'{z:+.6f}' for z in x)+']'
            lines.append(f"|{n}|{p['delta']:+.6f}|{fmt(p['family6_ci'])}|{p['positive_draws']}|{a['delta']:+.6f}|{fmt(a['family6_ci'])}|")
        lines+=['']
    lines+=['## Fresh per-draw results','','|Draw|Variant|AUROC|pAUC|Recall|FPR|','|---|---|---:|---:|---:|---:|']
    for n,v in e['tests']['confirmation_aegis:global'].items():lines.append(f"|{n.split('_')[0]}|{n[2:]}|{v['auroc']:.6f}|{v['partial_auroc']:.6f}|{v['tpr']:.1%}|{v['fpr']:.1%}|")
    lines+=['','## Train-only selected parameters','','|Draw|Variant|Parameter|CV pAUC|','|---|---|---:|---:|']
    for r in cv['records']:
        c=r['settings'][r['selected_index']];lines.append(f"|{r['draw']}|{r['variant']}|{c['value']}|{c['mean_pauc']:.6f}|")
    (out/'report-2026-10-29.md').write_bytes(('\n'.join(lines)+'\n').encode());print('REPORT_WRITTEN')

if __name__=='__main__':main()
