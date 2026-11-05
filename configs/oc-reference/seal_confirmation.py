"""Seal completed historical calibration before prospectively scoring any fresh row."""
from core import ROOT,NAMES,digest,write,validate_confirmation

out=ROOT/'results/oc-reference'
assert not (out/'pre-confirmation-seal.json').exists()
assert not (out/'confirmation-scores.jsonl').exists()
names=['manifest.json','token-audit.json','source-lock-v2.json','thresholds.json','historical-evaluation.json',
       'history-scores.jsonl','extraction-history.json','identity-history.json']
write(out/'pre-confirmation-seal.json',{'status':'frozen_before_confirmation_scoring','family':NAMES[1:],
    'sha256':{n:digest(out/n) for n in names},'bootstrap_repeats':10000,'seed':20261030,'primary':'standardized pAUC FPR<=0.1'})
validate_confirmation(out)
print('RV_PRE_CONFIRMATION_SEALED')
