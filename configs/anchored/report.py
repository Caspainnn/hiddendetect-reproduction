"""Frozen result tables; do not run after adding reviewed conclusions."""
import json
from anchored_core import ROOT

def main():
    out=ROOT/'results/anchored';e=json.loads((out/'evaluation.json').read_text());f=json.loads((out/'freeze.json').read_text());m=json.loads((ROOT/'configs/anchored/manifest.json').read_text());cv=json.loads((out/'cv.json').read_text())
    lines=['# R28-R29: anchored summary score correction','','Models and meta parameters frozen before scoring. New tasks reported separately.','',f"Meta selection: {f['selection']}",f"Fresh counts per class: {m['fresh_class_count']}",'','## Train-side selected settings','','|Head|Parameters|','|---|---|']
    for c in f['models']:lines.append(f"|{c['name']}|{c['parameters']}|")
    for key,heads in e['tests'].items():
        lines+=['','## '+key,'','|Head|AUROC|pAUC(.1)|Recall|FPR|Changed fraction vs B|','|---|---:|---:|---:|---:|---:|']
        for n,v in heads.items():lines.append(f"|{n}|{v['auroc']:.6f}|{v['partial_auroc']:.6f}|{v['tpr']:.1%}|{v['fpr']:.1%}|{v['changed_fraction_vs_B']:.1%}|")
        lines+=['','|Contrast|Delta pAUC|Seven-contrast corrected interval|Delta AUROC|Corrected interval|','|---|---:|---|---:|---|']
        for n,v in e['contrasts'][key].items():
            p=v['partial_auroc'];a=v['auroc'];fmt=lambda z:'['+', '.join(f'{x:+.6f}' for x in z)+']'
            lines.append(f"|{n}|{p['delta']:+.6f}|{fmt(p['family7_ci'])}|{a['delta']:+.6f}|{fmt(a['family7_ci'])}|")
    lines+=['','## All train-only candidate settings','','|Candidate|Mean meta-CV pAUC|CV standard deviation|','|---|---:|---:|']
    for c in cv['candidates']:lines.append(f"|{c['name']}|{c['cv_mean']:.6f}|{c['cv_std']:.6f}|")
    (out/'report-2026-10-29.md').write_bytes(('\n'.join(lines)+'\n').encode());print('REPORT_WRITTEN')

if __name__=='__main__':main()
