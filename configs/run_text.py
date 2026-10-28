"""Observe the unchanged author's text test() and persist its per-layer trace."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def validate_traces(traces, scores):
    if len(traces) != len(scores):
        raise ValueError('Trace and score counts differ')
    for trace, score in zip(traces, scores):
        if len(trace) != 32 or not all(math.isfinite(x) for x in trace):
            raise ValueError('Expected 32 finite decoder-layer signals')
        expected = sum((trace[i] + trace[i + 1]) / 2 for i in range(16, 29))
        if not math.isfinite(score) or not math.isclose(expected, score, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Author integral differs from captured closed [16,29] trace')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='configs/smoke-manifest.json')
    parser.add_argument('--output', default='results/smoke8')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    import transformers
    torch.manual_seed(0)
    np.random.seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    manifest = json.loads((ROOT / args.manifest).read_text())
    csv_path = ROOT / 'data/xstest_prompts.csv'
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == manifest['csv_sha256']
    rows = {x['id']: x for x in csv.DictReader(csv_path.read_text(encoding='utf-8-sig').splitlines())}
    selected = manifest['samples']
    samples = []
    for chosen in selected:
        row = rows[chosen['id']]
        assert hashlib.sha256(row['prompt'].encode()).hexdigest() == chosen['prompt_sha256']
        assert int('contrast' in row['type']) == chosen['label']
        samples.append({'txt': row['prompt'], 'img': None, 'toxicity': chosen['label']})
    author = ROOT / 'source/HiddenDetect'
    commit = subprocess.check_output(['git', '-C', str(author), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == '2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb'
    assert not subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    sys.path.insert(0, str(author / 'code'))
    spec = importlib.util.spec_from_file_location('author_hiddendetect', author / 'code/test.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Use the author's CSV loader as a separate check of our locked label mapping.
    loaded = module.load_XSTest(str(csv_path))
    assert len(loaded) == 450 and sum(x['toxicity'] for x in loaded) == 200
    author_labels = {x['txt']: x['toxicity'] for x in loaded}
    assert all(author_labels[x['txt']] == x['toxicity'] for x in samples)
    clip_alias = ROOT / 'openai/clip-vit-large-patch14-336'
    clip_alias.parent.mkdir(exist_ok=True)
    if not clip_alias.exists():
        clip_alias.symlink_to(ROOT / 'models/clip-vit-large-patch14-336', target_is_directory=True)
    original_loader = module.load_pretrained_model
    original_cosine = module.N.cosine_similarity
    signals, forward_records, model_info = [], [], {}

    def forward_hook(model, positional, keyword, outputs):
        ids = positional[0] if positional else keyword['input_ids']
        assert keyword.get('images') is None
        assert len(outputs.hidden_states) == 33
        forward_records.append({'input_length': ids.shape[-1],
                                'input_ids_sha256': hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest(),
                                'hidden_state_count': len(outputs.hidden_states),
                                'last_hidden_shape': list(outputs.hidden_states[-1].shape)})

    def load_and_observe(*positional, **keyword):
        started = time.monotonic()
        tokenizer, model, processor, context = original_loader(*positional, **keyword)
        assert model.config.vocab_size == 32000 and model.config.num_hidden_layers == 32
        assert model.lm_head.out_features == 32000 and not model.training
        meta_names = [name for name, value in model.named_parameters() if value.is_meta]
        assert not meta_names, 'Unmaterialized parameters: ' + str(meta_names[:5])
        from safetensors import safe_open
        checkpoint_dir = ROOT / 'models/llava-v1.6-vicuna-7b'
        weight_map = json.loads((checkpoint_dir / 'model.safetensors.index.json').read_text())['weight_map']
        vision_keys = sorted(name for name in weight_map if 'vision_tower' in name)
        assert vision_keys, 'Expected integrated vision weights in LLaVA checkpoint'
        checked = []
        for name in (vision_keys[0], vision_keys[len(vision_keys) // 2], vision_keys[-1]):
            parameter = model.state_dict()[name]
            with safe_open(checkpoint_dir / weight_map[name], framework='pt', device='cpu') as checkpoint:
                assert torch.equal(parameter.detach().cpu(), checkpoint.get_tensor(name).to(parameter.dtype))
            checked.append({'name': name, 'device': str(parameter.device), 'dtype': str(parameter.dtype),
                            'matches_integrated_llava_checkpoint': True})
        model.register_forward_hook(forward_hook, with_kwargs=True)
        encodings = {word: tokenizer.encode(word, add_special_tokens=False) for word in module.refusal_lst}
        model_info.update({'load_seconds': time.monotonic() - started,
                           'dtype': str(next(model.parameters()).dtype),
                           'device': str(next(model.parameters()).device),
                           'training': model.training, 'context_length': context,
                           'vocab_size': model.config.vocab_size, 'tokenizer_length': len(tokenizer),
                           'refusal_encodings': encodings,
                           'refusal_unique_first_token_ids': sorted({x[0] for x in encodings.values()}),
                           'meta_parameter_count': len(meta_names),
                           'integrated_vision_weight_count': len(vision_keys),
                           'vision_checkpoint_checks': checked,
                           'template': 'llava_v1',
                           'template_sha256': hashlib.sha256(str(module.conv_templates['llava_v1'].dict()).encode()).hexdigest(),
                           'local_vision_path_alias': 'openai/clip-vit-large-patch14-336 -> models/clip-vit-large-patch14-336'})
        print('MODEL_LOADED', json.dumps({k: model_info[k] for k in ['load_seconds', 'dtype', 'device', 'vocab_size']}), flush=True)
        return tokenizer, model, processor, context

    def observe_cosine(*positional, **keyword):
        result = original_cosine(*positional, **keyword)
        if positional[0].shape[-1] == 32000 and result.numel() == 1:
            signals.append(float(result.item()))
            if len(signals) % 32 == 0:
                print('SCORED', len(signals) // 32, '/', len(samples), flush=True)
        return result

    module.load_pretrained_model = load_and_observe
    module.N.cosine_similarity = observe_cosine
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        labels, scores = module.test(samples, str(ROOT / 'models/llava-v1.6-vicuna-7b'), s=16, e=29)
    finally:
        module.load_pretrained_model = original_loader
        module.N.cosine_similarity = original_cosine
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert len(signals) == 32 * len(samples) and len(forward_records) == len(samples)
    traces = [signals[i:i + 32] for i in range(0, len(signals), 32)]
    validate_traces(traces, scores)
    assert labels == [x['label'] for x in selected]
    result_rows = [{**chosen, 'F': trace, 'score': float(score), **forward}
                   for chosen, trace, score, forward in zip(selected, traces, scores, forward_records)]
    output = ROOT / args.output
    output.parent.mkdir(exist_ok=True)
    with output.with_suffix('.jsonl').open('w') as stream:
        for row in result_rows:
            stream.write(json.dumps(row) + '\n')
    summary = {'status': 'completed', 'purpose': 'text pipeline verification; not multimodal reproduction',
               'evaluation_scope': manifest.get('scope', '8-sample smoke only; no performance claim'),
               'author_commit': commit, 'manifest': args.manifest,
               'torch': torch.__version__, 'transformers': transformers.__version__,
               'gpu': torch.cuda.get_device_name(), 'model': model_info,
               'precision': 'float16', 'quantization': False, 'seed': 0,
               'interval_zero_based_inclusive': [16, 29], 'positive_label': 'unsafe=1',
               'samples': len(samples), 'wall_seconds_including_model_load': elapsed,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
               'trace_checks': '32 finite layer signals per sample; recomputed trapezoid equals author score',
               'author_code_modified': False, 'generation_executed': False,
               'observer_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if len(samples) > 8:
        summary['auroc'] = float(module.evaluate_AUROC(labels, scores))
        summary['auprc_trapezoid'] = float(module.evaluate_AUPRC(labels, scores))
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('COMPLETED', json.dumps({k: v for k, v in summary.items() if k != 'model'}), flush=True)

if __name__ == '__main__':
    main()
