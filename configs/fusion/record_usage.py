"""Record actual roles without rewriting the immutable acquisition record."""
import json
from fusion_core import ROOT,digest

def main():
    manifest=json.loads((ROOT/'configs/fusion/manifest.json').read_text())
    record={'acquisition_record_sha256':digest(ROOT/'logs/fusion-extra-data.json'),
        'manifest_sha256':digest(ROOT/'configs/fusion/manifest.json'),
        'counts':manifest['counts'],
        'amendment':'Before new inference, count-only audit added disjoint prospective ToxicChat train holdout; acquisition intent was originally safe calibration only.',
        'roles':{'calibration_toxic':'400 human-annotated non-toxic train rows; thresholds only',
            'test_toxic_holdout':'disjoint human-annotated train rows of both classes; prospective toxicity proxy evaluation only'},
        'scope':'toxicity proxy is separate from harmful-input labels; prospective train-partition holdout is not official-test evidence'}
    path=ROOT/'logs/fusion-data-usage.json'
    if path.exists(): assert json.loads(path.read_text())==record
    else: path.write_text(json.dumps(record,indent=2)+'\n')
    print('DATA_USAGE_BOUND',manifest['counts'])

if __name__=='__main__': main()
