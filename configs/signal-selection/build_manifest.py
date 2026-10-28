"""Prospective split locking. No model scores are used here."""
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from experiment_core import normalize_prompt, prompt_label

ROOT = Path(__file__).resolve().parents[2]

def remove_later_duplicates(rows, similarity=0.92):
    from sklearn.feature_extraction.text import TfidfVectorizer
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)
    texts = [normalize_prompt(x['prompt']) for x in rows]
    exact = {}
    for i, text in enumerate(texts):
        if text in exact:
            union(i, exact[text])
        else:
            exact[text] = i
    matrix = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=65536,
                            dtype=np.float32).fit_transform(texts)
    for start in range(0, len(rows), 128):
        pairs = (matrix[start:start + 128] @ matrix.T).tocoo()
        for i, j, value in zip(pairs.row, pairs.col, pairs.data):
            absolute = start + int(i)
            if absolute < j and value >= similarity:
                union(absolute, int(j))
    kept, removed = [], []
    for i, row in enumerate(rows):
        root = find(i)
        if root == i:
            kept.append(row)
        else:
            removed.append({'id': row['id'], 'role': row['role'], 'duplicate_of': rows[root]['id']})
    return kept, removed

def main():
    from transformers import AutoTokenizer
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX
    raw = ROOT / 'data/signal-selection/raw'
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/llava-v1.6-vicuna-7b', use_fast=False, local_files_only=True)
    rejected = Counter()
    candidates = []
    def add(identifier, text, label, source, role, category=''):
        if not isinstance(text, str) or not text.strip() or text.strip().upper() == 'REDACTED':
            rejected['empty_or_redacted'] += 1
            return
        if '<image>' in text:
            rejected['literal_image_token_in_text_only_task'] += 1
            return
        candidates.append({'id': identifier, 'prompt': text, 'label': int(label),
                           'source': source, 'role': role, 'category': category})
    # Previously inspected prompts have precedence and cannot become fresh tests.
    old = list(csv.DictReader((ROOT / 'data/xstest_prompts.csv').read_text(encoding='utf-8-sig').splitlines()))
    for row in old:
        add('old-xstest:' + row['id'], row['prompt'], int('contrast' in row['type']), 'old', 'excluded')
    with zipfile.ZipFile(ROOT / 'source/HiddenDetect/data/few_shot.zip') as archive:
        old_calibration = json.loads(archive.read('few_shot/few_shot.json'))
    for i, row in enumerate(old_calibration):
        add('old-calibration:' + str(i), row['txt'], row.get('toxicity', 0), 'old', 'excluded')
    seed = 20261028
    def stable_key(row):
        return hashlib.sha256((str(seed) + row['id']).encode()).hexdigest()
    aegis = raw / 'nvidia__Aegis-AI-Content-Safety-Dataset-2.0'
    for filename, role, cap in [('train.json', 'train_pool', 1650), ('validation.json', 'calibration', 1000), ('test.json', 'test_aegis', 1000)]:
        pool = []
        for row in json.loads((aegis / filename).read_text()):
            if not row.get('prompt') or row['prompt'].strip().upper() == 'REDACTED':
                rejected['empty_or_redacted'] += 1
                continue
            pool.append({'id': 'aegis:' + filename + ':' + str(row['id']), 'row': row})
        for label in (0, 1):
            eligible = sorted((x for x in pool if prompt_label(x['row']['prompt_label']) == label), key=stable_key)
            for item in eligible[:cap]:
                row = item['row']
                add(item['id'], row['prompt'], label, 'aegis', role, row.get('violated_categories', ''))
    jbb = raw / 'JailbreakBench__JBB-Behaviors/data'
    for filename, label in [('benign-behaviors.csv', 0), ('harmful-behaviors.csv', 1)]:
        for row in csv.DictReader((jbb / filename).read_text(encoding='utf-8-sig').splitlines()):
            add('jbb:' + str(label) + ':' + row['Index'], row['Goal'], label, 'jbb', 'test_jbb', row.get('Category', ''))
    toxic_file = raw / 'lmsys__toxic-chat/data/1123/toxic-chat_annotation_test.csv'
    toxic = [x for x in csv.DictReader(toxic_file.read_text(encoding='utf-8-sig').splitlines()) if x['human_annotation'].lower() == 'true']
    for label in (0, 1):
        pool = [{'id': 'toxicchat:' + row['conv_id'], 'row': row} for row in toxic if int(row['toxicity']) == label]
        for item in sorted(pool, key=stable_key)[:400]:
            add(item['id'], item['row']['user_input'], label, 'toxicchat_toxicity_proxy', 'test_toxicchat')
    eligible = []
    for row in candidates:
        if row['role'] == 'excluded':
            eligible.append(row)
            continue
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], row['prompt'])
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX)
        if len(ids) > 512:
            rejected['over_512_templated_tokens'] += 1
            continue
        row['input_length'] = len(ids)
        row['input_ids_sha256'] = hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()
        eligible.append(row)
    print('ELIGIBLE_FOR_DEDUP', len(eligible), flush=True)
    kept, duplicates = remove_later_duplicates(eligible)
    grouped = defaultdict(list)
    for row in kept:
        if row['role'] != 'excluded':
            grouped[(row['role'], row['label'])].append(row)
    selected = []
    for label in (0, 1):
        pool = grouped[('train_pool', label)]
        if len(pool) < 1100:
            raise RuntimeError('Insufficient pre-score training/development candidates; record counts before revising target')
        for i, row in enumerate(pool[:1100]):
            row['role'] = 'train' if i < 800 else 'dev'
            selected.append(row)
    for role, maximum in [('calibration', 500), ('test_aegis', 500), ('test_jbb', 100), ('test_toxicchat', 200)]:
        for label in (0, 1):
            selected.extend(grouped[(role, label)][:maximum])
    assert len({x['id'] for x in selected}) == len(selected)
    assert len({normalize_prompt(x['prompt']) for x in selected}) == len(selected)
    manifest = []
    for index, row in enumerate(selected):
        row['index'] = index
        row['prompt_sha256'] = hashlib.sha256(row['prompt'].encode()).hexdigest()
        row['group_id'] = hashlib.sha256(normalize_prompt(row['prompt']).encode()).hexdigest()
        manifest.append({k: v for k, v in row.items() if k != 'prompt'})
    target = ROOT / 'data/signal-selection/records.jsonl'
    target.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in selected), encoding='utf-8')
    report = {'seed': seed, 'max_templated_tokens': 512, 'no_truncation': True,
              'duplicate_policy': 'NFKC/casefold/punctuation normalization, then char3-5gram TFIDF cosine>=0.92; later roles removed; semantic equivalence not guaranteed',
              'counts': dict(Counter(x['role'] + ':' + str(x['label']) for x in selected)),
              'exclusions': dict(rejected), 'near_or_exact_duplicate_count': len(duplicates),
              'raw_lock_sha256': hashlib.sha256((ROOT / 'logs/signal-selection-download.json').read_bytes()).hexdigest(),
              'records_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'samples': manifest,
              'label_scope': {'aegis': 'human prompt_label', 'jbb': 'behavior partition', 'toxicchat': 'human toxicity proxy only; report separately'},
              'prospective': 'Locked before new model extraction; optimizers use train/dev only; calibration/test evaluated after freeze.'}
    (Path(__file__).parent / 'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (ROOT / 'logs/signal-selection-duplicates.json').write_text(json.dumps(duplicates, indent=2) + '\n')
    print('MANIFEST_LOCKED', json.dumps(report['counts']), 'duplicates', len(duplicates), flush=True)

if __name__ == '__main__':
    main()
