"""Lock a text smoke manifest before any score is observed."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def normalized(text):
    return ' '.join(text.split()).casefold()

def select_smoke(rows, excluded):
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate XSTest ID')
    excluded = {normalized(x) for x in excluded}
    selected = []
    for label in (0, 1):
        candidates = sorted((row for row in rows if int('contrast' in row['type']) == label
                             and normalized(row['prompt']) not in excluded), key=lambda row: row['id'])
        if len(candidates) < 4:
            raise ValueError('Insufficient samples for class ' + str(label))
        selected.extend({'id': row['id'], 'type': row['type'], 'label': label,
                         'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest()}
                        for row in candidates[:4])
    return selected

if __name__ == '__main__':
    path = ROOT / 'data/xstest_prompts.csv'
    rows = list(csv.DictReader(path.read_text(encoding='utf-8-sig').splitlines()))
    assert len(rows) == 450 and sum('contrast' in x['type'] for x in rows) == 200
    with zipfile.ZipFile(ROOT / 'source/HiddenDetect/data/few_shot.zip') as archive:
        calibration = json.loads(archive.read('few_shot/few_shot.json'))
    excluded = {x['txt'] for x in calibration}
    manifest = {'dataset': 'paul-rottger/xstest',
                'source': json.loads((ROOT / 'logs/xstest-source.json').read_text()),
                'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'total': len(rows), 'safe': 250, 'unsafe': 200,
                'source_difference': 'Original author CSV instead of HiddenDetect-linked HF gpt4 parquet; no model completions used. HF-copy equality not yet verified.',
                'selection': '4 per label, lexicographic ID order, excluding normalized exact calibration prompts',
                'calibration_prompt_overlap_ids': [x['id'] for x in rows if normalized(x['prompt']) in {normalized(s) for s in excluded}],
                'samples': select_smoke(rows, excluded)}
    target = ROOT / 'configs/smoke-manifest.json'
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print('LOCKED', [x['id'] for x in manifest['samples']], 'calibration overlaps', manifest['calibration_prompt_overlap_ids'])
