"""Pure extraction contracts; all large arrays private and bounded."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = {'P': ((32,5,4096), 'float16'), 'X': ((32,5,64), 'float32'),
          'S': ((32,16), 'float32'), 'A': ((32,32,8), 'float32'),
          'M': ((32,8), 'float32'), 'F': ((32,), 'float32'),
          'L': ((32000,), 'float16'), 'Q': ((6,), 'float32')}
TRACE_NAMES = ['norm_mean','norm_std','norm_max','adjacent_cos_mean','adjacent_cos_min',
    'first_last_cos','previous_layer_cos_mean','previous_layer_cos_min',
    'refusal_mean','refusal_std','refusal_max','refusal_min','entropy_mean','entropy_std',
    'entropy_rank_trend','top12_margin_mean']
ATTENTION_NAMES = ['prefix_mass','user_q1_mass','user_q2_mass','user_q3_mass','user_q4_mass',
                   'suffix_mass','normalized_entropy','maximum_weight']
MODULE_NAMES = ['attention_last_norm','attention_user_mean_norm','attention_user_max_norm',
                'mlp_last_norm','mlp_user_mean_norm','mlp_user_max_norm',
                'attention_mlp_last_cos','attention_mlp_user_mean_cos']

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def regions_from_offsets(offsets, user_span):
    start, end = user_span
    indices = np.array([i for i,(a,b) in enumerate(offsets) if b > a and b > start and a < end], dtype=int)
    if not len(indices): raise ValueError('No user tokens')
    return np.array_split(indices, 4)

def pool_hidden(hidden, regions):
    unit = hidden / np.maximum(np.linalg.norm(hidden, axis=-1, keepdims=True), 1e-12)
    return np.stack([unit[-1]] + [unit[ix].mean(axis=0) if len(ix) else np.zeros(unit.shape[-1]) for ix in regions])

def projection_matrix(width=4096, projected=64):
    rng = np.random.default_rng(20261029)
    return (rng.integers(0,2,size=(width,projected))*2-1).astype(np.float32) / np.sqrt(projected)

def open_cache(directory, manifest_sha, count, provenance):
    directory.mkdir(parents=True, exist_ok=True)
    meta = {'manifest_sha256': manifest_sha, 'count':count, 'provenance':provenance,
            'layout':{key:{'shape':[count,*shape],'dtype':dtype} for key,(shape,dtype) in LAYOUT.items()}}
    expected_bytes = sum(count*np.prod(shape)*np.dtype(dtype).itemsize for shape,dtype in LAYOUT.values())
    if expected_bytes > 6*1024**3: raise ValueError('Cache exceeds preregistered 6GiB')
    path = directory/'metadata.json'; exists = path.exists()
    if exists and json.loads(path.read_text()) != meta: raise ValueError('Cache provenance/schema drift')
    arrays = {'directory':directory,'count':count}
    for name,item in meta['layout'].items():
        arr = np.lib.format.open_memmap(directory/(name+'.npy'), mode='r+' if exists else 'w+',
            dtype=item['dtype'], shape=tuple(item['shape']))
        if list(arr.shape) != item['shape'] or arr.dtype != np.dtype(item['dtype']): raise ValueError('Cache shape/dtype drift')
        arrays[name] = arr
    if not exists: path.write_text(json.dumps(meta,indent=2)+'\n')
    prefix = directory/'checkpoint.json'
    start = json.loads(prefix.read_text())['next_index'] if prefix.exists() else 0
    if not 0 <= start <= count: raise ValueError('Invalid completed prefix')
    return arrays,start

def checkpoint(arrays, next_index):
    if not 0 <= next_index <= arrays['count']: raise ValueError('Invalid prefix')
    for name in LAYOUT: arrays[name].flush()
    temporary = arrays['directory']/'checkpoint.tmp'
    temporary.write_text(json.dumps({'next_index':next_index}))
    temporary.replace(arrays['directory']/'checkpoint.json')
