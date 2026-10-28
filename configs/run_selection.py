"""Run the author's 12-query multimodal FDV calibration, then apply Eq.4."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def select_range(fdv):
    if not fdv or not all(math.isfinite(x) for x in fdv):
        raise ValueError('Empty/nonfinite FDV')
    eligible = [i for i, x in enumerate(fdv) if x > fdv[-1]]
    if not eligible:
        raise ValueError('Paper Eq.4 defines no range for this FDV')
    return min(eligible), max(eligible), eligible

def require_author_commit(repository, expected):
    actual = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError('Author repository revision changed: ' + actual)
    return actual

def main():
    os.chdir(ROOT)
    sys.dont_write_bytecode = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(0)
    np.random.seed(0)
    baseline = json.loads((ROOT / 'results/xstest450-default-s16e29.json').read_text())
    assert baseline['status'] == 'completed' and baseline['model']['meta_parameter_count'] == 0
    author = ROOT / 'source/HiddenDetect'
    commit = require_author_commit(author, '2fd161b1fc47c99b36fe0a299ec64ff0b816f0bb')
    assert commit == baseline['author_commit']
    assert not subprocess.check_output(['git', '-C', str(author), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    with zipfile.ZipFile(author / 'data/few_shot.zip') as archive:
        for member in archive.infolist():
            target = (ROOT / 'data' / member.filename).resolve()
            if not target.is_relative_to((ROOT / 'data').resolve()):
                raise ValueError('Unsafe archive path')
        archive.extractall(ROOT / 'data')
    calibration_path = ROOT / 'data/few_shot/few_shot.json'
    calibration = json.loads(calibration_path.read_text())
    ordered = [x for x in calibration if x.get('toxicity', 0) != 1] + [x for x in calibration if x.get('toxicity', 0) == 1]
    assert len(ordered) == 12 and sum(x.get('toxicity') == 1 for x in ordered) == 6
    sys.path.insert(0, str(author / 'code'))
    spec = importlib.util.spec_from_file_location('author_layer_selection', author / 'code/safety_aware_layers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_loader = module.load_pretrained_model
    original_cosine = module.N.cosine_similarity
    signals, vision_info, forwards = [], {}, []

    def observe_forward(model, positional, keyword, outputs):
        assert len(outputs.hidden_states) == 33
        image = keyword.get('images')
        forwards.append({'has_image': image is not None,
                         'image_tensor_shape': list(image.shape) if image is not None else None,
                         'language_sequence_length': outputs.hidden_states[-1].shape[1]})

    def load_and_check(*positional, **keyword):
        tokenizer, model, processor, context = original_loader(*positional, **keyword)
        assert not any(x.is_meta for x in model.parameters()), 'Unmaterialized parameters'
        tower = model.get_vision_tower()
        assert tower.is_loaded and tower.device.type == 'cuda'
        from safetensors import safe_open
        checkpoint_dir = ROOT / 'models/llava-v1.6-vicuna-7b'
        weight_map = json.loads((checkpoint_dir / 'model.safetensors.index.json').read_text())['weight_map']
        for checked in baseline['model']['vision_checkpoint_checks']:
            name = checked['name']
            actual = model.state_dict()[name]
            with safe_open(checkpoint_dir / weight_map[name], framework='pt', device='cpu') as checkpoint:
                assert torch.equal(actual.detach().cpu(), checkpoint.get_tensor(name).to(actual.dtype))
        vision_info.update({'device': str(tower.device), 'dtype': str(tower.dtype),
                            'meta_parameter_count': sum(x.is_meta for x in model.parameters()),
                            'loaded': tower.is_loaded, 'checkpoint_tensor_checks': 3})
        model.register_forward_hook(observe_forward, with_kwargs=True)
        return tokenizer, model, processor, context

    def observe_cosine(*positional, **keyword):
        result = original_cosine(*positional, **keyword)
        if positional[0].shape[-1] == 32000 and result.numel() == 1:
            signals.append(float(result.item()))
            if len(signals) % 32 == 0:
                print('CALIBRATED', len(signals) // 32, '/ 12', flush=True)
        return result

    module.load_pretrained_model = load_and_check
    module.N.cosine_similarity = observe_cosine
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        fdv = module.locate_most_safety_aware_layers(str(ROOT / 'models/llava-v1.6-vicuna-7b'))
    finally:
        module.load_pretrained_model = original_loader
        module.N.cosine_similarity = original_cosine
    torch.cuda.synchronize()
    assert len(signals) == 12 * 32 and len(forwards) == 12
    traces = np.asarray(signals, dtype=np.float64).reshape(12, 32)
    assert np.isfinite(traces).all()
    safe_mean, unsafe_mean = traces[:6].mean(0), traces[6:].mean(0)
    assert np.allclose(unsafe_mean - safe_mean, fdv, atol=1e-12, rtol=1e-12)
    s, e, eligible = select_range(fdv)
    result = {'status': 'completed', 'purpose': 'original multimodal calibration; text test scoring only',
              'author_commit': commit, 'seed': 0,
              'calibration_json_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
              'samples': 12, 'safe': 6, 'unsafe': 6, 'image_queries': sum(x['has_image'] for x in forwards),
              'F_safe_mean': safe_mean.tolist(), 'F_unsafe_mean': unsafe_mean.tolist(), 'FDV': fdv,
              'reference_last_layer': fdv[-1], 'eligible_zero_based_layers': eligible,
              'interval_zero_based_inclusive': [s, e],
              'interval_one_based_decoder_blocks': [s + 1, e + 1],
              'vision': vision_info, 'wall_seconds_including_model_load': time.monotonic() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'author_code_modified': False,
              'normalization': 'author behavior preserved, including repeated norm of final hidden state',
              'observer_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    cached = [json.loads(line) for line in (ROOT / 'results/xstest450-default-s16e29.jsonl').read_text().splitlines()]
    labels = np.asarray([x['label'] for x in cached])
    scores = [float(np.trapz(np.asarray(x['F'])[s:e + 1])) for x in cached]
    result['xstest450_cached_auroc_selected_interval'] = float(roc_auc_score(labels, scores))
    result['xstest450_cached_auroc_default_interval'] = baseline['auroc']
    result['test_scope'] = '450 original CSV prompts include 8 smoke prompts. Layer selection uses only the separate author calibration set; no test-driven range tuning.'
    sample_rows = []
    for sample, trace, forward in zip(ordered, traces.tolist(), forwards):
        image = sample.get('img')
        sample_rows.append({'prompt_sha256': hashlib.sha256(sample['txt'].encode()).hexdigest(),
                            'label': sample.get('toxicity', 0), 'image_path': image,
                            'image_sha256': hashlib.sha256((ROOT / image).read_bytes()).hexdigest() if image not in (None, 'null') else None,
                            'F': trace, **forward})
    (ROOT / 'results/original-layer-selection.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'results/original-calibration-traces.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in sample_rows))
    with (ROOT / 'results/xstest450-selected-scores.jsonl').open('w') as stream:
        for row, score in zip(cached, scores):
            stream.write(json.dumps({'id': row['id'], 'label': row['label'], 'score': score}) + '\n')
    print('SELECTION_COMPLETED', json.dumps({k: result[k] for k in ['interval_zero_based_inclusive', 'xstest450_cached_auroc_selected_interval', 'wall_seconds_including_model_load', 'peak_allocated_bytes']}), flush=True)

if __name__ == '__main__':
    main()
