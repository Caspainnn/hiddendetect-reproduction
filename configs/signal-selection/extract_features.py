"""Single frozen-model pass; prospective records -> resumable H/F/logit cache."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def open_cache(directory, key, count):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {'manifest_sha256': key, 'count': count,
                'shapes': {'H': [count, 32, 4096], 'F': [count, 32], 'L': [count, 32000]}}
    path = directory / 'metadata.json'
    exists = path.exists()
    if exists and json.loads(path.read_text()) != metadata:
        raise ValueError('Feature cache manifest/count drift')
    arrays = {}
    for name, shape in metadata['shapes'].items():
        dtype = np.float32 if name == 'F' else np.float16
        arrays[name] = np.lib.format.open_memmap(directory / (name + '.npy'), mode='r+' if exists else 'w+', dtype=dtype, shape=tuple(shape))
        if list(arrays[name].shape) != shape or arrays[name].dtype != dtype:
            raise ValueError('Corrupt cache shape/dtype')
    if not exists:
        path.write_text(json.dumps(metadata, indent=2))
    checkpoint = directory / 'checkpoint.json'
    start = json.loads(checkpoint.read_text())['next_index'] if checkpoint.exists() else 0
    if not 0 <= start <= count:
        raise ValueError('Invalid completed cache prefix')
    return {**arrays, 'directory': directory, 'count': count}, start

def checkpoint_cache(cache, next_index):
    if not 0 <= next_index <= cache['count']:
        raise ValueError('Invalid checkpoint boundary')
    for name in ('H', 'F', 'L'):
        cache[name].flush()
    target = cache['directory'] / 'checkpoint.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps({'next_index': next_index}))
    temporary.replace(target)

def main():
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from safetensors import safe_open
    from llava.conversation import conv_templates
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    torch.manual_seed(0)
    manifest_path = Path(__file__).parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records_path = ROOT / 'data/signal-selection/records.jsonl'
    assert hashlib.sha256(records_path.read_bytes()).hexdigest() == manifest['records_sha256']
    records = [json.loads(x) for x in records_path.read_text().splitlines()]
    key = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cache, start = open_cache(ROOT / 'cache/signal-selection', key, len(records))
    if start == len(records):
        print('CACHE_ALREADY_COMPLETE', len(records), flush=True)
        return
    author = ROOT / 'source/HiddenDetect'
    assert subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip() == '2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'
    assert not subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    alias = ROOT / 'openai/clip-vit-large-patch14-336'
    alias.parent.mkdir(exist_ok=True)
    if not alias.exists():
        alias.symlink_to(ROOT / 'models/clip-vit-large-patch14-336', target_is_directory=True)
    started = time.monotonic()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tokenizer, model, processor, context = load_pretrained_model(str(ROOT / 'models/llava-v1.6-vicuna-7b'), None,
        'llava-v1.6-vicuna-7b', device_map='auto', torch_dtype=torch.float16)
    assert torch.cuda.is_available() and not model.training
    assert not any(x.is_meta for x in model.parameters())
    assert model.config.num_hidden_layers == 32 and model.config.vocab_size == 32000
    weight_map = json.loads((ROOT / 'models/llava-v1.6-vicuna-7b/model.safetensors.index.json').read_text())['weight_map']
    vision = sorted(x for x in weight_map if 'vision_tower' in x)
    for name in (vision[0], vision[len(vision) // 2], vision[-1]):
        actual = model.state_dict()[name]
        with safe_open(ROOT / 'models/llava-v1.6-vicuna-7b' / weight_map[name], framework='pt', device='cpu') as stream:
            assert torch.equal(actual.detach().cpu(), stream.get_tensor(name).to(actual.dtype))
    sys.path.insert(0, str(author / 'code'))
    spec = importlib.util.spec_from_file_location('unchanged_author_test', author / 'code/test.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    reference = torch.zeros(32000, device=model.device, dtype=torch.float32)
    token_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in original.refusal_lst]
    reference[token_ids] = 1
    norm, head = model.model.norm, model.lm_head
    def forward(row):
        assert hashlib.sha256(row['prompt'].encode()).hexdigest() == row['prompt_sha256']
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], row['prompt'])
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        assert ids.shape[-1] == row['input_length'] <= 512
        assert hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest() == row['input_ids_sha256']
        with torch.no_grad():
            output = model(ids, images=None, image_sizes=None, output_hidden_states=True)
            assert len(output.hidden_states) == 33
            hidden = np.stack([state[0, -1].detach().cpu().numpy() for state in output.hidden_states[1:]])
            signals = []
            for state in output.hidden_states[1:]:
                # Preserve author full-sequence norm/head and final repeated norm.
                logits = head(norm(state))[:, -1, :]
                signals.append(torch.nn.functional.cosine_similarity(logits, reference).item())
            final = output.logits[0, -1].detach()
            assert torch.equal(final, final.half().to(final.dtype)), 'Unexpected loss in logit storage cast'
            logits_np = final.half().cpu().numpy()
        assert hidden.shape == (32, 4096) and np.isfinite(hidden).all() and np.isfinite(signals).all() and np.isfinite(logits_np).all()
        return hidden, np.asarray(signals, dtype=np.float32), logits_np
    # Verify eight prospective TRAIN rows against the unchanged author's function.
    smoke = []
    for label in (0, 1):
        smoke.extend([x for x in records if x['role'] == 'train' and x['label'] == label][:4])
    manual = [forward(x)[1] for x in smoke]
    captured = []
    old_loader, old_cosine = original.load_pretrained_model, original.N.cosine_similarity
    def capture(*args, **kwargs):
        result = old_cosine(*args, **kwargs)
        captured.append(float(result.item()))
        return result
    original.load_pretrained_model = lambda *args, **kwargs: (tokenizer, model, processor, context)
    original.N.cosine_similarity = capture
    try:
        labels, scores = original.test([{'txt': x['prompt'], 'img': None, 'toxicity': x['label']} for x in smoke], str(ROOT / 'models/llava-v1.6-vicuna-7b'), s=16, e=29)
    finally:
        original.load_pretrained_model, original.N.cosine_similarity = old_loader, old_cosine
    difference = float(np.max(np.abs(np.asarray(captured).reshape(8, 32) - np.asarray(manual))))
    assert difference <= 1e-7
    assert np.allclose(scores, np.trapz(np.asarray(manual, dtype=np.float64)[:, 16:30], axis=1), rtol=1e-10, atol=1e-12)
    smoke_report = {'samples': 8, 'train_only': True, 'max_F_difference_from_author': difference,
                    'scores_match': True, 'manifest_sha256': key, 'code_sha256': code_hash}
    (ROOT / 'logs/signal-selection-feature-smoke.json').write_text(json.dumps(smoke_report, indent=2))
    print('AUTHOR_F_SMOKE_PASS', difference, 'START', start, 'TOTAL', len(records), flush=True)
    torch.cuda.reset_peak_memory_stats()
    for i in range(start, len(records)):
        cache['H'][i], cache['F'][i], cache['L'][i] = forward(records[i])
        if (i + 1) % 16 == 0 or i + 1 == len(records):
            checkpoint_cache(cache, i + 1)
        if (i + 1) % 100 == 0:
            print('CACHED', i + 1, '/', len(records), 'SECONDS', round(time.monotonic() - started, 2), flush=True)
    report = {'status': 'complete', 'rows': len(records), 'manifest_sha256': key,
              'seconds_this_run': time.monotonic() - started, 'resumed_at': start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'code_sha256': code_hash, 'dtype': 'float16 H/logits; float32 F; logits cast verified lossless',
              'training': False, 'generation': False, 'author_F_smoke': smoke_report,
              'cache_bytes': sum((cache['directory'] / (name + '.npy')).stat().st_size for name in ('H', 'F', 'L'))}
    (ROOT / 'logs/signal-selection-extraction.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FEATURE_CACHE_COMPLETE', json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
